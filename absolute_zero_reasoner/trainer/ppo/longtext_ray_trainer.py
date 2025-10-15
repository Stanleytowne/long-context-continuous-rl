"""
Long-Text Continuous Learning PPO Trainer.

This module implements the trainer for long-text continuous learning using
the three-phase training loop (Proposer, Solver, Judge).
"""

import os
import gc
import ray
import torch
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from absolute_zero_reasoner.trainer.ppo.reason_rl_ray_trainer import ReasonRLRayPPOTrainer
from absolute_zero_reasoner.data_construction.constructor import (
    get_gen_longtext_qa_data,
    get_pred_longtext_qa_data,
    get_judge_longtext_qa_data,
    chunk_long_text,
)
from absolute_zero_reasoner.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from absolute_zero_reasoner.utils.logging_utils.stdout import PrettyPrinter
from absolute_zero_reasoner.trainer.ppo.azr_ray_trainer import DatasetManager


class LongTextQARayPPOTrainer(ReasonRLRayPPOTrainer):
    """
    PPO Trainer for long-text continuous learning.
    
    Training Flow:
    1. SFT Warmup: Pre-train model on long text
    2. Proposer Phase: Generate QA pairs from text segments
    3. Solver Phase: Answer questions without seeing text
    4. (Optional) Judge Phase: Train judge to evaluate answers
    """
    
    def __init__(
        self,
        long_text_path: str = None,
        past_epoch_window: int = 10,
        *args,
        **kwargs
    ):
        """
        Initialize the long-text QA trainer.
        
        Args:
            long_text_path: Path to the long text file to learn from
            past_epoch_window: Window for tracking recent data
            *args, **kwargs: Additional arguments for parent class
        """
        super().__init__(*args, **kwargs)
        
        self._past_epoch_window = past_epoch_window
        self.dataset_manager = DatasetManager.remote()
        self._last_cleanup_step = 0
        self._cleanup_frequency = self.config.azr.get('executor_cleanup_frequency', 5)
        
        # Load and process long text
        self.long_text_path = long_text_path or self.config.azr.long_text.get('path')
        if not self.long_text_path:
            raise ValueError("long_text_path must be specified either in config or as argument")
        
        print(f"[INFO] Loading long text from: {self.long_text_path}")
        self.long_text = self._load_long_text(self.long_text_path)
        print(f"[INFO] Long text loaded: {len(self.long_text)} characters")
        
        # Chunk configuration
        self.use_chunking = self.config.azr.long_text.get('use_chunking', True)
        self.chunk_size = self.config.azr.long_text.get('chunk_size', 2048)
        self.overlap = self.config.azr.long_text.get('overlap', 256)
        self.chunk_sampling_strategy = self.config.azr.long_text.get(
            'chunk_sampling_strategy', 'random'
        )
        
        # Process text into chunks
        if self.use_chunking:
            print(f"[INFO] Chunking text: chunk_size={self.chunk_size}, overlap={self.overlap}")
            self.text_chunks = chunk_long_text(
                self.long_text,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
                tokenizer=self.tokenizer
            )
            print(f"[INFO] Created {len(self.text_chunks)} text chunks")
        else:
            self.text_chunks = [{'text': self.long_text, 'chunk_id': 0}]
            print(f"[INFO] Using full text without chunking")
        
        # Initialize prompt manager
        try:
            from absolute_zero_reasoner.utils.prompt_manager import PromptManager
            self.prompt_manager = PromptManager(
                config=self.config,
                output_dir=self.config.trainer.default_local_dir + "/prompt_history"
            )
            print("[INFO] PromptManager initialized")
        except Exception as e:
            print(f"[WARNING] Could not initialize PromptManager: {e}")
            self.prompt_manager = None
        
        # Set prompt manager for reward function
        if hasattr(self.reward_fn, 'set_prompt_manager') and self.prompt_manager:
            self.reward_fn.set_prompt_manager(self.prompt_manager)
            print("[INFO] Set prompt_manager for reward_fn")
        
        print(f"[INFO] LongTextQARayPPOTrainer initialized successfully")
    
    def _create_dataloader(self):
        """
        Override parent's _create_dataloader to skip file-based dataset loading.
        
        For longtext_qa tasks, data is generated dynamically during training,
        so we don't need to load from train_files. Instead, we create a dummy
        dataloader that yields empty batches for the parent's fit method.
        """
        # For longtext_qa, we don't use pre-existing training files
        # Data is generated dynamically in each training iteration
        print("[INFO] Skipping file-based dataloader creation for longtext_qa task")
        print("[INFO] Data will be generated dynamically during training")
        
        # Set placeholders to avoid errors in parent class
        self.train_dataset = None
        self.val_dataset = None
        self.val_dataloader = None
        
        # Create a dummy dataloader that yields one batch per epoch
        # The parent's fit method will iterate over this
        class DummyDataLoader:
            def __init__(self, total_epochs):
                self.total_epochs = total_epochs
            
            def __iter__(self):
                # Yield one empty dict per epoch
                # We'll override fit() to handle actual data generation
                for _ in range(self.total_epochs):
                    yield {}
            
            def __len__(self):
                return self.total_epochs
        
        self.train_dataloader = DummyDataLoader(self.config.trainer.total_epochs)
        
        # Set total_training_steps (required by parent class)
        # For longtext_qa, we use the configured total_training_steps
        if self.config.trainer.total_training_steps is not None and self.config.trainer.total_training_steps > 0:
            total_training_steps = self.config.trainer.total_training_steps
        else:
            # If not specified, use total_epochs as a fallback
            # Since data is generated dynamically, we treat each epoch as 1 step
            total_training_steps = self.config.trainer.total_epochs
        
        self.total_training_steps = total_training_steps
        print(f'[INFO] Total training steps: {self.total_training_steps}')
        
        # Inject total_training_steps into optim config (same as parent class)
        from omegaconf import OmegaConf, open_dict
        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            if hasattr(self.config, 'critic'):
                self.config.critic.optim.total_training_steps = total_training_steps
    
    def _load_long_text(self, file_path: str) -> str:
        """
        Load long text from file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            Text content as string
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Long text file not found: {file_path}")
        
        # Determine file encoding and format
        _, ext = os.path.splitext(file_path)
        
        try:
            if ext in ['.txt', '.md', '.rst']:
                # Plain text files
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif ext == '.pdf':
                # PDF files (requires additional library)
                raise NotImplementedError(
                    "PDF support not yet implemented. Please convert to .txt first."
                )
            elif ext in ['.json', '.jsonl']:
                # JSON files
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    if ext == '.jsonl':
                        # JSONL: each line is a JSON object
                        lines = [json.loads(line) for line in f]
                        # Concatenate text fields
                        text = '\n\n'.join([
                            line.get('text', '') or line.get('content', '') or str(line)
                            for line in lines
                        ])
                    else:
                        # Single JSON object or array
                        data = json.load(f)
                        if isinstance(data, dict):
                            text = data.get('text', '') or data.get('content', '') or str(data)
                        elif isinstance(data, list):
                            text = '\n\n'.join([
                                item.get('text', '') or item.get('content', '') or str(item)
                                for item in data
                            ])
                        else:
                            text = str(data)
            else:
                # Default: try reading as plain text
                print(f"[WARNING] Unknown file extension {ext}, treating as plain text")
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            # Basic cleaning
            text = text.strip()
            
            if not text:
                raise ValueError(f"Loaded text is empty from {file_path}")
            
            return text
            
        except Exception as e:
            raise RuntimeError(f"Failed to load long text from {file_path}: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        gc.collect()
    
    def _is_longtext_task(self) -> bool:
        """Check if current task type is longtext_qa."""
        return getattr(self.config.azr, 'task_type', 'code') == 'longtext_qa'
    
    def _create_train_gen_dataloader(
        self,
        problem_type: str,
        data_len: int,
        dataset_key: str = None,
        seeding: bool = False,
    ) -> DataLoader:
        """
        Create dataloader for proposer (generation) phase.
        
        Model sees text segments and generates QA pairs.
        
        Args:
            problem_type: Should be 'longtext_qa'
            data_len: Number of samples to generate
            dataset_key: Key for dataset management
            seeding: Whether in seeding phase
            
        Returns:
            DataLoader iterator
        """
        if problem_type != 'longtext_qa':
            raise ValueError(f"Invalid problem type for longtext trainer: {problem_type}")
        
        # Get reference QA pairs from dataset if available
        if dataset_key is None:
            dataset_key = "longtext_qa"
        
        try:
            reference_qa_pairs = ray.get(self.dataset_manager.get_dataset.remote(dataset_key))
        except Exception:
            reference_qa_pairs = []
        
        # Create output path
        parquet_path = (self._code_dir / f'train_gen_{problem_type}.parquet').as_posix()
        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
        
        # Configure reference inclusion
        include_references = self.config.azr.reward.generation_reward_config.get(
            'include_references', 0.5
        )
        
        # Generate proposer data
        PrettyPrinter.section_header(f"Creating Proposer Training Data")
        print(f"[INFO] Generating {data_len} proposer samples")
        print(f"[INFO] Reference QA pairs available: {len(reference_qa_pairs)}")
        print(f"[INFO] Include references probability: {include_references}")
        
        get_gen_longtext_qa_data(
            long_text=self.long_text,
            target_data_len=data_len,
            content_max_length=self.config.azr.data_selection_strategy.content_max_length,
            output_path=parquet_path,
            split='train',
            tokenizer=self.tokenizer,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            use_chunking=self.use_chunking,
            chunk_sampling_strategy=self.chunk_sampling_strategy,
            reference_qa_pairs=reference_qa_pairs if reference_qa_pairs else None,
            include_references=include_references,
            prompt_manager=self.prompt_manager,
        )
        
        # Create dataset
        gen_train_dataset = RLHFDataset(
            parquet_files=parquet_path,
            tokenizer=self.tokenizer,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get('return_raw_chat', False),
            truncation='error',
            extra_source_key=f"gen_{problem_type}_train"
        )
        
        # Create sampler
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(gen_train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(gen_train_dataset)
        
        return iter(DataLoader(
            dataset=gen_train_dataset,
            batch_size=self.config.data.train_batch_size,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler
        ))
    
    def _create_train_pred_dataloader(
        self,
        problem_type: str,
        data_len: int
    ) -> DataLoader:
        """
        Create dataloader for solver (prediction) phase.
        
        Model answers questions WITHOUT seeing the text.
        
        Args:
            problem_type: Should be 'longtext_qa'
            data_len: Number of samples to generate
            
        Returns:
            DataLoader iterator
        """
        if problem_type != 'longtext_qa':
            raise ValueError(f"Invalid problem type for longtext trainer: {problem_type}")
        
        dataset_key = "longtext_qa"
        
        # Get QA pairs from dataset
        try:
            full_dataset = ray.get(self.dataset_manager.get_dataset.remote(dataset_key))
        except Exception:
            print("[WARNING] No QA pairs in dataset yet, creating empty dataset")
            full_dataset = []
        
        if not full_dataset:
            print("[WARNING] No QA pairs available for solver phase")
            # Return empty dataloader or skip this phase
            return iter([])
        
        # Sample data based on strategy
        strategy = self.config.azr.pred_data_mix_strategy
        
        PrettyPrinter.section_header(f"Creating Solver Training Data")
        print(f"[INFO] Total QA pairs in dataset: {len(full_dataset)}")
        print(f"[INFO] Sampling strategy: {strategy}")
        
        if strategy == "uniform_total":
            selected_data = random.sample(full_dataset, min(len(full_dataset), data_len))
        elif strategy == "max_new":
            # Prioritize recently generated QA pairs
            total_recent = ray.get(self.dataset_manager.get_recent_additions.remote(
                dataset_key, self.global_steps, self._past_epoch_window
            ))
            new_qa_pairs = full_dataset[-total_recent:] if total_recent > 0 else []
            new_samples = random.sample(new_qa_pairs, min(len(new_qa_pairs), data_len))
            remaining = data_len - len(new_samples)
            if remaining > 0:
                old_samples = random.sample(full_dataset, min(len(full_dataset), remaining))
                selected_data = new_samples + old_samples
            else:
                selected_data = new_samples
        elif strategy == "half_new":
            # 50-50 mix of new and old
            total_recent = ray.get(self.dataset_manager.get_recent_additions.remote(
                dataset_key, self.global_steps, self._past_epoch_window
            ))
            new_qa_pairs = full_dataset[-total_recent:] if total_recent > 0 else []
            new_count = min(len(new_qa_pairs), data_len // 2)
            base_count = data_len - new_count
            selected_data = (
                random.sample(new_qa_pairs, new_count) +
                random.sample(full_dataset, min(len(full_dataset), base_count))
            )
        else:
            # Default: uniform sampling
            selected_data = random.sample(full_dataset, min(len(full_dataset), data_len))
        
        print(f"[INFO] Selected {len(selected_data)} QA pairs for solver training")
        
        # Create output path
        parquet_path = (self._code_dir / f'train_pred_{problem_type}.parquet').as_posix()
        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
        
        # Generate solver data
        get_pred_longtext_qa_data(
            qa_pairs=selected_data,
            target_data_len=data_len,
            content_max_length=self.config.azr.data_selection_strategy.content_max_length,
            output_path=parquet_path,
            split='train',
            tokenizer=self.tokenizer,
            prompt_manager=self.prompt_manager,
        )
        
        # Create dataset
        pred_train_dataset = RLHFDataset(
            parquet_files=parquet_path,
            tokenizer=self.tokenizer,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get('return_raw_chat', False),
            truncation='error',
            extra_source_key=f"pred_{problem_type}_train"
        )
        
        # Create sampler
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(pred_train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(pred_train_dataset)
        
        return iter(DataLoader(
            dataset=pred_train_dataset,
            batch_size=self.config.data.train_batch_size,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler
        ))
    
    def _create_train_judge_dataloader(
        self,
        problem_type: str,
        data_len: int
    ) -> DataLoader:
        """
        Create dataloader for judge training phase (optional).
        
        Model learns to evaluate answer quality.
        
        Args:
            problem_type: Should be 'longtext_qa'
            data_len: Number of samples to generate
            
        Returns:
            DataLoader iterator
        """
        if problem_type != 'longtext_qa':
            raise ValueError(f"Invalid problem type for longtext trainer: {problem_type}")
        
        # Check if judge training is enabled
        train_judge = self.config.azr.get('train_judge', False)
        if not train_judge:
            print("[INFO] Judge training disabled, skipping judge dataloader")
            return iter([])
        
        # For judge training, we need QA pairs with both ground truth and generated answers
        # This would typically come from solver phase outputs
        # For now, return empty as judge training is optional
        print("[INFO] Judge training not yet fully implemented")
        return iter([])
    
    def _update_dataset_with_valid_data(
        self,
        valid_data: List[Dict],
        problem_type: str
    ):
        """
        Update the dataset manager with validated QA pairs.
        
        Args:
            valid_data: List of validated QA pair dictionaries
            problem_type: Problem type identifier
        """
        if not valid_data:
            return
        
        dataset_key = "longtext_qa"
        
        # Add data to dataset manager
        ray.get(self.dataset_manager.add_data.remote(
            dataset_key=dataset_key,
            new_data=valid_data,
            step=self.global_steps
        ))
        
        print(f"[INFO] Added {len(valid_data)} validated QA pairs to dataset")
        
        # Log dataset statistics
        total_size = len(ray.get(self.dataset_manager.get_dataset.remote(dataset_key)))
        print(f"[INFO] Total QA pairs in dataset: {total_size}")
    
    def _save_checkpoint(self):
        """Save checkpoint including datasets."""
        super()._save_checkpoint()
        # Save datasets
        save_dir = Path(self.config.trainer.default_local_dir) / 'datasets'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all datasets
        import pickle
        datasets_with_types = ray.get(self.dataset_manager.get_all_data_with_type_counters.remote())
        
        # Save datasets
        pickle.dump(datasets_with_types, open(save_dir / 'datasets.pkl', 'wb'))
        PrettyPrinter.status("SAVE", f"Saved datasets to {save_dir}", "success")
    
    def _load_checkpoint(self):
        """Load checkpoint including datasets."""
        super()._load_checkpoint()
        
        # Initialize loaded_datasets flag
        self.loaded_datasets = False
        
        if self.global_steps == 0:
            PrettyPrinter.section_header(f"Training from scratch")
        else:
            PrettyPrinter.section_header(f"Resuming training from checkpoint, step {self.global_steps}")
        
        # Load datasets if they exist
        datasets_path = Path(self.config.trainer.default_local_dir) / 'datasets' / 'datasets.pkl'
        if self.config.trainer.resume_mode == 'auto' and datasets_path.exists():
            import pickle
            from collections import defaultdict
            
            datasets_with_types = pickle.load(open(datasets_path, 'rb'))
            
            # Filter datasets based on global step
            if self.global_steps > 0:
                for dataset_key in ['general', 'general_pair']:
                    steps_key = f"{dataset_key}_steps"
                    if steps_key in datasets_with_types and dataset_key in datasets_with_types:
                        filtered_data = []
                        filtered_steps = []
                        
                        for entry, step in zip(datasets_with_types[dataset_key], datasets_with_types[steps_key]):
                            if step <= self.global_steps:
                                filtered_data.append(entry)
                                filtered_steps.append(step)
                        
                        datasets_with_types[dataset_key] = filtered_data
                        datasets_with_types[steps_key] = filtered_steps
                        
                        # Filter step counter
                        counter_key = f"{dataset_key}_steps_counter"
                        if counter_key in datasets_with_types:
                            filtered_counter = defaultdict(int)
                            for step, count in datasets_with_types[counter_key].items():
                                if step <= self.global_steps:
                                    filtered_counter[step] = count
                            datasets_with_types[counter_key] = filtered_counter
            
            ray.get(self.dataset_manager.full_load_data_with_type_counters.remote(datasets_with_types))
            PrettyPrinter.status("LOAD", f"Loaded datasets from {datasets_path}", "success")
            self.loaded_datasets = True
    
    def _compute_batch(self, batch, metrics: dict, timing_raw: dict, problem_type: str):
        """
        Compute batch for longtext QA task (similar to GeneralIORayPPOTrainer).
        
        Args:
            batch: Input batch (DataProto)
            metrics: Metrics dictionary to update
            timing_raw: Timing dictionary
            problem_type: Problem type (e.g., 'gen_longtext_qa', 'pred_longtext_qa')
            
        Returns:
            Tuple of (processed batch, updated metrics)
        """
        from verl.utils.debug import marked_timer
        from verl.protocol import DataProto
        from verl.trainer.ppo.ray_trainer import (
            compute_response_mask,
            compute_advantage,
            apply_kl_penalty,
        )
        from verl.trainer.ppo.metric_utils import _compute_response_info
        import uuid
        import numpy as np
        
        PrettyPrinter.section_header(f"Computing batch for {problem_type}")
        
        # Pop keys for generation
        gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
        
        # Generate sequences
        with marked_timer(f'gen/{problem_type}', timing_raw):
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
        
        # Add UIDs
        batch.non_tensor_batch['uid'] = np.array(
            [str(uuid.uuid4()) for _ in range(len(batch.batch))],
            dtype=object
        )
        
        # Repeat to align with repeated responses in rollout
        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        batch = batch.union(gen_batch_output)
        
        batch.batch["response_mask"] = compute_response_mask(batch)
        
        # Balance the batch
        self._balance_batch(batch, metrics=metrics)
        
        # Compute global valid tokens
        batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()
        
        # Recompute old_log_probs
        with marked_timer(f'old_log_prob/{problem_type}', timing_raw):
            from verl.trainer.ppo.ray_trainer import agg_loss
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
            metrics.update(old_log_prob_metrics)
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)
        
        # Compute reference policy if enabled
        if self.use_reference_policy:
            with marked_timer(f'ref/{problem_type}', timing_raw):
                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)
        
        # Compute values if using critic
        if self.use_critic:
            with marked_timer(f'values/{problem_type}', timing_raw):
                values = self.critic_wg.compute_values(batch)
                batch = batch.union(values)
        
        # Compute rewards and advantages
        with marked_timer(f'adv/{problem_type}', timing_raw):
            if self.use_rm:
                reward_tensor = self.rm_wg.compute_rm_score(batch)
                batch = batch.union(reward_tensor)
            
            # Call reward function
            reward_fn_kwargs = {
                'data': batch,
                'problem_type': problem_type,
            }
            
            with marked_timer(f'reward_fn/{problem_type}', timing_raw):
                PrettyPrinter.status("REWARD", f"Computing rewards for {problem_type}...", "info")
                reward_tensor, train_metrics, valid_data = self.reward_fn(**reward_fn_kwargs)
                PrettyPrinter.status("REWARD", f"Found {len(valid_data) if valid_data else 0} valid data for {problem_type}", "success")
            
            # Log new QA pairs if available
            if valid_data and self.config.azr.random_print_max_programs > 0 and problem_type.startswith('gen'):
                PrettyPrinter.section_header(f"New {problem_type} QA Pairs")
                max_print = min(self.config.azr.random_print_max_programs, len(valid_data))
                for qa_pair in random.sample(valid_data, max_print):
                    PrettyPrinter.status("QUESTION", qa_pair.get('question', 'N/A'), "info")
                    PrettyPrinter.status("ANSWER", qa_pair.get('answer', 'N/A'), "info")
                    if 'thought' in qa_pair:
                        PrettyPrinter.status("THOUGHT", qa_pair['thought'], "info")
                    print("\n" + "-"*80 + "\n")
            
            # Add valid data to dataset manager
            if problem_type.startswith('gen'):
                if valid_data:
                    ray.get(self.dataset_manager.add_general_batch.remote(valid_data, self.global_steps))
            elif problem_type.startswith('pred'):
                if valid_data:
                    ray.get(self.dataset_manager.add_general_pair_batch.remote(valid_data, self.global_steps))
            
            # Update metrics
            train_metrics = {f'{problem_type}/{k}': np.mean(v) if isinstance(v, list) else v 
                           for k, v in train_metrics.items()}
            
            # Log number of valid pairs
            if problem_type.startswith('gen'):
                dataset_key = 'general'  # Using 'general' dataset for longtext QA pairs
                train_metrics[f'{problem_type}/num_valid_questions'] = ray.get(
                    self.dataset_manager.get_recent_additions.remote(
                        dataset_key, self.global_steps, self._past_epoch_window
                    )
                )
            elif problem_type.startswith('pred'):
                dataset_key = 'general_pair'
                train_metrics[f'{problem_type}/num_valid_pairs'] = ray.get(
                    self.dataset_manager.get_recent_additions.remote(
                        dataset_key, self.global_steps, self._past_epoch_window
                    )
                )
            
            metrics.update(train_metrics)
            batch.batch['token_level_scores'] = reward_tensor
            
            # Apply KL penalty if configured
            if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                if self.config.algorithm.use_kl_in_reward and hasattr(self, 'kl_ctrl_in_reward'):
                    batch, kl_metrics = apply_kl_penalty(
                        batch,
                        kl_ctrl=self.kl_ctrl_in_reward,
                        kl_penalty=self.config.algorithm.kl_penalty
                    )
                    metrics.update(kl_metrics)
                else:
                    batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
            else:
                batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
            
            # Compute advantages
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                config=self.config.algorithm
            )
        
        gc.collect()
        return batch, metrics
    
    def fit(self):
        """
        Complete training loop for longtext QA tasks.
        
        Follows the same structure as GeneralIORayPPOTrainer.fit() but
        adapted for longtext QA with proposer/solver phases.
        """
        from omegaconf import OmegaConf
        from absolute_zero_reasoner.utils.tracking import ReasonRLTracking
        from verl.protocol import DataProto
        from verl.trainer.ppo.ray_trainer import reduce_metrics, compute_timing_metrics
        from verl.utils.debug import marked_timer
        import numpy as np
        
        # Initialize logger
        logger = ReasonRLTracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
            tags=self.config.trainer.wandb_tags,
            resume="must" if self.config.trainer.resume_mode == 'auto' and \
                self.config.trainer.wandb_run_id is not None else False,
            run_id=self.config.trainer.wandb_run_id \
                if self.config.trainer.wandb_run_id is not None else None
        )
        
        self.global_steps = 0
        
        # Load checkpoint before doing anything
        self._load_checkpoint()
        
        # Base model chat template
        if self.config.actor_rollout_ref.model.pretrained_tokenizer:
            self.tokenizer.chat_template = "{%- for message in messages -%}{{- '\n' if not loop.first -}}{{- message['content'] -}}{%- endfor -%}"
        
        # Validation before training
        if self.config.trainer.get('val_before_train', True) and (self.global_steps == 0 or self.config.trainer.get('val_only', False)):
            PrettyPrinter.status("INFO", "Skipping initial validation for longtext QA", "info")
            if self.config.trainer.get('val_only', False):
                return
        
        if self.config.trainer.val_only:
            PrettyPrinter.status("INFO", "Validation only mode enabled, exiting after validation", "info")
            return
        
        # Initialize code directory for storing generated datasets
        code_dir = Path(self.config.trainer.default_local_dir) / 'code'
        self._code_dir = code_dir
        code_dir.mkdir(parents=True, exist_ok=True)
        PrettyPrinter.status("Directory", f"Using code directory at {code_dir}", "info")
        
        # Initialize seed dataset if needed
        if not self.loaded_datasets:
            PrettyPrinter.section_header(f"Initializing LongText QA Training")
            # For longtext QA, we don't need a seed dataset as we generate from text
            # Just make sure dataset manager is ready
            PrettyPrinter.status("INIT", "Dataset manager ready for longtext QA", "success")
        
        # Start training from step 1
        self.global_steps += 1
        if self.config.azr.pretrain_pred_steps > 0 and self.global_steps <= self.config.azr.pretrain_pred_steps:
            self.pretrain_pred = True
        else:
            self.pretrain_pred = False
        
        # Main training loop
        while self.global_steps < self.total_training_steps:
            PrettyPrinter.section_header(f"Training Step {self.global_steps}")
            
            PrettyPrinter.progress_bar(
                current=self.global_steps,
                total=self.total_training_steps,
                label="Training Progress"
            )
            
            # Calculate data length for this step
            data_len = self.config.data.train_batch_size * self.config.azr.data_selection_strategy.get('update_iteration', 1)
            
            # Create dataloaders for this step
            if 'longtext_qa' in self.config.azr.problem_types:
                # Create proposer dataloader (generate questions from text)
                gen_longtext_dataloader = self._create_train_gen_dataloader(
                    problem_type='longtext_qa',
                    data_len=data_len,
                    dataset_key='general',  # Use 'general' dataset for longtext QA
                )
                
                # Create solver dataloader (answer questions)
                pred_longtext_dataloader = self._create_train_pred_dataloader(
                    problem_type='longtext_qa',
                    data_len=data_len,
                )
                
                # Check if we have enough data for the iteration
                try:
                    # Test if we can get at least one batch
                    test_batch = next(gen_longtext_dataloader)
                except StopIteration:
                    PrettyPrinter.status("ERROR", "No data generated for proposer phase", "error")
                    break
            
            # Inner loop: update_iteration times per global step
            for _ in range(self.config.azr.data_selection_strategy.get('update_iteration', 1)):
                metrics = {}
                timing_raw = {}
                batches = {}
                
                with marked_timer('step', timing_raw):
                    # Periodic cleanup
                    if self.global_steps - self._last_cleanup_step >= self._cleanup_frequency:
                        PrettyPrinter.section_header("Periodic Cleanup")
                        with marked_timer('cleanup', timing_raw):
                            self.cleanup()
                        self._last_cleanup_step = self.global_steps
                    
                    if 'longtext_qa' in self.config.azr.problem_types:
                        # Proposer phase: generate questions from text
                        if not self.pretrain_pred and self.config.azr.train_propose:
                            try:
                                batch_dict = next(gen_longtext_dataloader)
                            except StopIteration:
                                # Regenerate dataloader if exhausted
                                gen_longtext_dataloader = self._create_train_gen_dataloader(
                                    problem_type='longtext_qa',
                                    data_len=data_len,
                                    dataset_key='general',
                                )
                                batch_dict = next(gen_longtext_dataloader)
                            
                            gen_batch: DataProto = DataProto.from_single_dict(batch_dict)
                            gen_batch, metrics = self._compute_batch(
                                gen_batch, metrics, timing_raw, 
                                problem_type='gen_longtext_qa'
                            )
                            batches[f'gen_longtext_qa'] = gen_batch
                        
                        # Solver phase: answer questions
                        if self.config.azr.train_solve:
                            try:
                                batch_dict = next(pred_longtext_dataloader)
                                pred_batch: DataProto = DataProto.from_single_dict(batch_dict)
                                pred_batch, metrics = self._compute_batch(
                                    pred_batch, metrics, timing_raw,
                                    problem_type='pred_longtext_qa'
                                )
                                batches[f'pred_longtext_qa'] = pred_batch
                            except StopIteration:
                                PrettyPrinter.status("WARN", "No QA pairs available for solver phase yet", "warn")
                    
                    # Skip if no batches were created
                    if not batches:
                        PrettyPrinter.status("ERROR", "No batches available for training. Skipping this step.", "error")
                        continue
                    
                    # Concatenate batches
                    batch = DataProto.concat(list(batches.values()))
                    
                    PrettyPrinter.section_header(f"Starting Parameter Updates")
                    
                    # Update critic
                    if self.use_critic:
                        with marked_timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)
                    
                    # Update actor (implement critic warmup)
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)
                    
                    # Validation
                    PrettyPrinter.section_header(f"Starting Validation")
                    if self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                        with marked_timer('testing', timing_raw):
                            # For longtext QA, skip validation for now
                            # Can implement later with held-out questions
                            PrettyPrinter.status("INFO", "Validation not yet implemented for longtext QA", "info")
                            val_metrics = {}
                        metrics.update(val_metrics)
                    
                    # Print dataset statistics
                    if 'longtext_qa' in self.config.azr.problem_types:
                        try:
                            num_questions = ray.get(self.dataset_manager.get_dataset_size.remote('general'))
                            PrettyPrinter.status(
                                "DATA",
                                f"Number of generated QA pairs: {num_questions}",
                                "info"
                            )
                        except Exception as e:
                            PrettyPrinter.status("DATA", f"Could not get dataset size: {e}", "warn")
                    
                    # Save checkpoint
                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with marked_timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                
                # Collect metrics
                from absolute_zero_reasoner.trainer.ppo.azr_ray_trainer import compute_data_metrics
                all_types = []
                if 'longtext_qa' in self.config.azr.problem_types:
                    if not self.pretrain_pred and self.config.azr.train_propose:
                        all_types.append('gen_longtext_qa')
                    if self.config.azr.train_solve:
                        all_types.append('pred_longtext_qa')
                
                if all_types:
                    sep_batches = batch.chunk(len(all_types))
                    for sep_batch, problem_type in zip(sep_batches, all_types):
                        sep_metrics = compute_data_metrics(
                            batch=sep_batch,
                            use_critic=self.use_critic,
                            tokenizer=self.tokenizer
                        )
                        sep_metrics = {f'{problem_type}/{k}': v for k, v in sep_metrics.items()}
                        metrics.update(sep_metrics)
                
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                
                # Print metrics
                PrettyPrinter.table(
                    ["Category", "Value"],
                    [[k, v] for k, v in metrics.items()],
                    title="Step Metrics"
                )
                
                # Log metrics
                logger.log(data=metrics, step=self.global_steps)
                
                # Update pretrain_pred flag
                if self.global_steps >= self.config.azr.pretrain_pred_steps:
                    self.pretrain_pred = False
                
                self.global_steps += 1
                
                gc.collect()
                
                # Check if training is complete
                if self.global_steps >= self.total_training_steps:
                    # Final validation
                    if self.val_reward_fn is not None:
                        PrettyPrinter.section_header(f"Starting Final Validation")
                        PrettyPrinter.status("INFO", "Validation not yet implemented for longtext QA", "info")
                    
                    # Final checkpoint
                    if self.config.trainer.save_freq > 0 and \
                            (self.global_steps - 1) % self.config.trainer.save_freq != 0:
                        with marked_timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                    return

