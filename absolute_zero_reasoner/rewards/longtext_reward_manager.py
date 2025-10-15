"""
Reward Manager for Long-Text Continuous Learning Tasks.

This module implements reward computation for the proposer-solver adversarial loop:
- Proposer: Generates QA pairs from text, rewarded for making Solver fail
- Solver: Answers questions without text, rewarded for correctness
"""

from typing import Dict, List, Any, Tuple
import os
import torch
import pandas as pd
import numpy as np
from collections import defaultdict
from openai import OpenAI
from transformers import AutoTokenizer
import re
from fractions import Fraction

from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.protocol import DataProto
from verl.utils.model import update_model_config
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto


def model_prompting_with_client(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> str:
    """
    Prompt an external LLM (e.g., via NVIDIA API) for evaluation.
    
    Args:
        client: OpenAI client instance
        model: Model identifier
        prompt: Evaluation prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        
    Returns:
        Generated text response
    """
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"[ERROR] LLM prompting failed: {e}")
        return ""


class LongTextQARewardManager:
    """
    Reward manager for long-text continuous learning with adversarial training.
    
    Reward Design:
    - Solver: answer_correctness + format_reward
    - Proposer: (1 - solver_correctness) + format_reward
    
    This creates an adversarial loop where:
    - Proposer tries to generate hard questions (wants Solver to fail)
    - Solver tries to answer correctly (wants to succeed)
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        num_examine: int = 0,
        split: str = 'train',
        reward_fn_extraction_type: str = 'boxed',
        splitter: str = 'Assistant:',
        output_path: str = './',
        generation_reward_config: Dict[str, Any] = None,
        eval_reward_config: Dict[str, Any] = None,
        model_name: str = 'meta/llama-3.1-405b-instruct',
        max_prompt_length: int = 8192,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 0.95,
        stream: bool = True,
        judge_with_actor: bool = False,
        train_judge: bool = False,
        prompt_manager=None,
        api_key: str = None,
        api_base_url: str = "https://integrate.api.nvidia.com/v1",
        use_format_reward: bool = True,
        **kwargs
    ):
        """
        Initialize the reward manager.
        
        Args:
            tokenizer: Tokenizer for text processing
            num_examine: Number of examples to print for debugging
            split: Data split ('train', 'val', 'test')
            reward_fn_extraction_type: Type of answer extraction
            splitter: String to split response from prompt
            output_path: Directory for saving outputs
            generation_reward_config: Config for proposer rewards (not used in simplified design)
            eval_reward_config: Config for solver rewards
            model_name: External LLM model for judging
            max_prompt_length: Maximum prompt length
            temperature: LLM sampling temperature
            max_tokens: Maximum tokens for LLM generation
            top_p: Nucleus sampling parameter
            stream: Whether to use streaming
            judge_with_actor: Use actor model for judging (recommended)
            train_judge: Whether to train the judge
            prompt_manager: Manager for dynamic prompts
            api_key: API key for external LLM
            api_base_url: Base URL for external LLM API
            use_format_reward: Whether to add format compliance reward
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.split = split
        self.reward_fn_extraction_type = reward_fn_extraction_type
        self.splitter = splitter
        self.output_path = output_path
        self.max_prompt_length = max_prompt_length
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stream = stream
        self.judge_with_actor = judge_with_actor
        self.train_judge = train_judge
        self.prompt_manager = prompt_manager
        self.use_format_reward = use_format_reward
        
        # Reward configuration with defaults
        self.eval_reward_config = eval_reward_config or {
            'answer_weight': 1.0,  # Weight for answer correctness
            'format_weight': 0.1,  # Weight for format compliance
        }
        
        # Model name for external LLM
        self.model_name = model_name
        
        # Initialize external LLM client if not using actor
        if not judge_with_actor:
            self.client = OpenAI(
                base_url=api_base_url,
                api_key=api_key or "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"
            )
        else:
            self.client = None
    
    def set_prompt_manager(self, prompt_manager):
        """Set or update the prompt_manager."""
        self.prompt_manager = prompt_manager
    
    def extract_score_from_tags(self, text: str) -> List[float]:
        """
        Extract numerical scores from <score></score> tags.
        
        Args:
            text: Text containing score tags
            
        Returns:
            List of extracted scores as floats
        """
        score_tags = re.findall(r'<score>(.*?)</score>', text, re.IGNORECASE | re.DOTALL)
        
        extracted_scores = []
        for tag_content in score_tags:
            tag_content = tag_content.strip()
            score_value = None
            
            # Try to extract fraction (e.g., 8/10)
            fraction_match = re.search(r'\b(\d+)/(\d+)\b', tag_content)
            if fraction_match:
                numerator = int(fraction_match.group(1))
                denominator = int(fraction_match.group(2))
                if denominator != 0:
                    score_value = float(Fraction(numerator, denominator))
            else:
                # Try to extract float
                float_match = re.search(r'\b(\d+\.\d+)\b', tag_content)
                if float_match:
                    score_value = float(float_match.group(1))
                else:
                    # Try to extract integer
                    integer_match = re.search(r'\b(\d+)\b', tag_content)
                    if integer_match:
                        score_value = float(integer_match.group(1))
            
            if score_value is not None:
                extracted_scores.append(score_value)
        
        return extracted_scores
    
    def count_tags(self, text: str, tag: str) -> tuple:
        """Count opening and closing tags in text."""
        open_count = text.count(f'<{tag}>')
        close_count = text.count(f'</{tag}>')
        return open_count, close_count
    
    def check_format_compliance(self, text: str, task_type: str = 'solver') -> float:
        """
        Check if output follows the required format.
        
        Format requirements differ by task:
        - Proposer: <question>...</question> <answer>...</answer>
        - Solver: <answer>...</answer> (only answer, question is given)
        
        Args:
            text: Generated text
            task_type: 'proposer' or 'solver'
            
        Returns:
            Format score (0.0 to 1.0)
        """
        a_open, a_close = self.count_tags(text, 'answer')
        
        # Answer tag scoring (required for both tasks)
        a_score = 0.0
        if a_open == a_close == 1:
            a_score = 1.0
        elif a_open == a_close:
            if a_open > 0:
                a_score = 0.5
            else:
                a_score = 0.0
        else:
            a_score = 0.0
        
        if task_type == 'solver':
            # Solver only needs answer tag
            return a_score
        
        elif task_type == 'proposer':
            # Proposer needs both question and answer tags
            q_open, q_close = self.count_tags(text, 'question')
            
            q_score = 0.0
            if q_open == q_close == 1:
                q_score = 1.0
            elif q_open == q_close:
                if q_open > 0:
                    q_score = 0.5
                else:
                    q_score = 0.0
            else:
                q_score = 0.0
            
            # Both tags must be present, return average
            return (q_score + a_score) / 2
        
        else:
            raise ValueError(f"Invalid task_type: {task_type}. Must be 'solver' or 'proposer'")
    
    def rollout_with_actors(self, dataset_file: str, rollout_actor_wg) -> DataProto:
        """
        Use actor model to generate responses for evaluation.
        
        Args:
            dataset_file: Path to parquet file with prompts
            rollout_actor_wg: Actor worker group for generation
            
        Returns:
            DataProto with generated responses
        """
        dataset = RLHFDataset(
            parquet_files=dataset_file,
            tokenizer=self.tokenizer,
            prompt_key='prompt',
            max_prompt_length=self.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=False,
            truncation='error'
        )
        
        if os.path.exists(dataset_file):
            os.remove(dataset_file)
        
        sampler = torch.utils.data.SequentialSampler(data_source=dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=len(dataset),
            drop_last=False,
            shuffle=False,
            collate_fn=collate_fn,
            sampler=sampler,
        )
        
        data = next(iter(dataloader))
        batch = DataProto.from_single_dict(data)
        gen_batch = batch.pop(['input_ids', 'attention_mask', 'position_ids'])
        gen_batch.meta_info = {
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
            'recompute_log_prob': False,
            'do_sample': True,
            'validate': True,
        }
        
        gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, rollout_actor_wg.world_size)
        out_gen_batch_padded = rollout_actor_wg.generate_sequences(gen_batch_padded)
        out_gen_batch = unpad_dataproto(out_gen_batch_padded, pad_size=pad_size)
        batch = batch.union(out_gen_batch)
        
        return batch
    
    def _judge_answer_with_actor(
        self,
        questions: List[str],
        answers: List[str],
        ground_truths: List[str],
        uids: List[str],
        rollout_actor_wg,
    ) -> Dict[str, float]:
        """
        Use actor model to judge answer quality.
        
        Args:
            questions: List of questions
            answers: List of generated answers
            ground_truths: List of ground truth answers
            uids: List of unique identifiers
            rollout_actor_wg: Actor worker group
            
        Returns:
            Dictionary mapping uid to score (0-1)
        """
        from absolute_zero_reasoner.data_construction.prompts import get_longtext_judge_prompt
        
        # Build evaluation prompts
        eval_prompts = []
        for question, answer, gt, uid in zip(questions, answers, ground_truths, uids):
            eval_text = get_longtext_judge_prompt(
                question=question,
                ground_truth=gt,
                generated_answer=answer,
                prompt_manager=self.prompt_manager
            )
            eval_prompts.append({
                'prompt': [{'role': 'user', 'content': eval_text}],
                'uid': uid,
            })
        
        # Save to temporary file
        temp_judge_file = f'{self.output_path}/temp_longtext_judge.parquet'
        pd.DataFrame(eval_prompts).to_parquet(temp_judge_file)
        
        # Generate judgments with actor
        judge_batch = self.rollout_with_actors(
            dataset_file=temp_judge_file,
            rollout_actor_wg=rollout_actor_wg
        )
        
        # Collect scores
        uid_to_score = {}
        for jb in judge_batch:
            uid = jb.non_tensor_batch['uid']
            text = self.tokenizer.decode(jb.batch['responses'], skip_special_tokens=True)
            scores = self.extract_score_from_tags(text)
            
            if scores:
                # Normalize to 0-1 range (assuming 1-10 scale)
                score = (scores[0] - 1) / 9.0
                uid_to_score[uid] = min(1.0, max(0.0, score))
            else:
                print(f"[WARNING] No score found for uid {uid}, using neutral 0.5")
                uid_to_score[uid] = 0.5
        
        return uid_to_score
    
    def _judge_answer_with_llm(
        self,
        question: str,
        answer: str,
        ground_truth: str,
    ) -> float:
        """
        Use external LLM to judge answer quality.
        
        Args:
            question: The question
            answer: Generated answer
            ground_truth: Ground truth answer
            
        Returns:
            Score between 0 and 1
        """
        from absolute_zero_reasoner.data_construction.prompts import get_longtext_judge_prompt
        
        prompt = get_longtext_judge_prompt(
            question=question,
            ground_truth=ground_truth,
            generated_answer=answer,
            prompt_manager=self.prompt_manager
        )
        
        try:
            if self.client:
                response = model_prompting_with_client(
                    client=self.client,
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                
                # Extract score
                scores = self.extract_score_from_tags(response)
                if scores:
                    # Normalize to 0-1 range (assuming 1-10 scale)
                    return (scores[0] - 1) / 9.0
                else:
                    print(f"[WARNING] No score extracted from LLM response")
                    return 0.5
            else:
                print("[WARNING] No LLM client available")
                return 0.5
                
        except Exception as e:
            print(f"[ERROR] LLM judging failed: {e}")
            return 0.5
    
    def _compute_solver_rewards(
        self,
        data_dicts: List[Dict],
        rollout_actor_wg=None,
    ) -> Tuple[List[float], List[float]]:
        """
        Compute rewards for solver (pred_longtext_qa) phase.
        
        Reward = answer_weight * correctness + format_weight * format_score
        
        Args:
            data_dicts: List of data dictionaries with questions, answers, and ground truth
            rollout_actor_wg: Actor worker group for judging (if judge_with_actor=True)
            
        Returns:
            Tuple of (answer_scores, format_scores)
        """
        answer_scores = []
        format_scores = []
        
        config = self.eval_reward_config
        
        # Extract data
        questions = []
        answers = []
        ground_truths = []
        uids = []
        
        for i, data_dict in enumerate(data_dicts):
            question = data_dict.get('question', '')
            generated_answer = data_dict.get('response', '')
            ground_truth = data_dict.get('ground_truth', '')
            uid = data_dict.get('uid', f'sample_{i}')
            
            questions.append(question)
            answers.append(generated_answer)
            ground_truths.append(ground_truth)
            uids.append(uid)
            
            # Format score (solver only needs <answer> tag)
            if self.use_format_reward:
                format_score = self.check_format_compliance(generated_answer, task_type='solver')
                format_scores.append(format_score)
            else:
                format_scores.append(0.0)
        
        # Judge answers
        if self.judge_with_actor and rollout_actor_wg is not None:
            print(f"[INFO] Using actor to judge {len(data_dicts)} solver answers")
            uid_to_score = self._judge_answer_with_actor(
                questions=questions,
                answers=answers,
                ground_truths=ground_truths,
                uids=uids,
                rollout_actor_wg=rollout_actor_wg,
            )
            answer_scores = [uid_to_score.get(uid, 0.5) for uid in uids]
        else:
            print(f"[INFO] Using external LLM to judge {len(data_dicts)} solver answers")
            for question, answer, gt in zip(questions, answers, ground_truths):
                if not answer or not gt:
                    answer_scores.append(0.0)
                else:
                    score = self._judge_answer_with_llm(question, answer, gt)
                    answer_scores.append(score)
        
        return answer_scores, format_scores
    
    def _compute_proposer_rewards(
        self,
        data_dicts: List[Dict],
        solver_scores: List[float],
    ) -> Tuple[List[float], List[Dict]]:
        """
        Compute rewards for proposer (gen_longtext_qa) phase.
        
        Reward = (1 - solver_correctness) + format_weight * format_score
        
        The proposer is rewarded for generating questions that the solver fails to answer.
        This creates an adversarial loop where proposer generates hard questions.
        
        Args:
            data_dicts: List of data dictionaries
            solver_scores: Solver correctness scores (0-1) for each question
            
        Returns:
            Tuple of (rewards list, valid data list)
        """
        from absolute_zero_reasoner.data_construction.constructor import extract_qa_pair
        
        rewards = []
        valid_data = []
        format_scores = []
        
        for i, data_dict in enumerate(data_dicts):
            # Extract QA pair from model output
            response = data_dict.get('response', '')
            qa_pair = extract_qa_pair(response)
            
            question = qa_pair['question']
            answer = qa_pair['answer']
            
            # Format score (proposer needs both <question> and <answer> tags)
            if self.use_format_reward:
                format_score = self.check_format_compliance(response, task_type='proposer')
                format_scores.append(format_score)
            else:
                format_score = 0.0
                format_scores.append(0.0)
            
            if not qa_pair['extraction_success'] or not question or not answer:
                # Failed extraction - low reward
                rewards.append(0.0)
                continue
            
            # Adversarial reward: proposer rewarded for solver failure
            # If solver_scores is available, use it; otherwise use default
            if i < len(solver_scores):
                difficulty_reward = 1.0 - solver_scores[i]
            else:
                difficulty_reward = 0.5  # Neutral if no solver score
            
            # Combine rewards
            format_weight = self.eval_reward_config.get('format_weight', 0.1)
            reward = difficulty_reward + format_weight * format_score
            rewards.append(reward)
            
            # Save valid QA pairs for dataset
            if qa_pair['extraction_success']:
                valid_data.append({
                    'question': question,
                    'answer': answer,
                    'chunk_id': data_dict.get('chunk_id', -1),
                    'difficulty_reward': difficulty_reward,
                    'format_score': format_score,
                })
        
        return rewards, valid_data
    
    def __call__(
        self,
        data,  # DataProto object
        problem_type: str = None,
        executor=None,
        rollout_actor_wg=None,
        n_samples: int = 3,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict, List[Dict]]:
        """
        Main reward computation entry point.
        
        Args:
            data: DataProto object containing batch data
            problem_type: Type of task ('gen_longtext_qa', 'pred_longtext_qa')
            executor: Code executor (not used for long-text tasks)
            rollout_actor_wg: Actor worker group for self-judging
            n_samples: Number of samples for difficulty estimation
            
        Returns:
            Tuple of (reward_tensor, all_scores_dict, valid_data_list)
        """
        # Initialize reward tensor
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        all_scores = defaultdict(list)
        valid_data = []
        
        # Determine problem type if not specified
        if problem_type is None:
            problem_types = [
                d.non_tensor_batch['extra_info'].get('metric', 'pred_longtext_qa')
                for d in data
            ]
            problem_type = problem_types[0] if problem_types else 'pred_longtext_qa'
        
        # Build data dictionaries from DataProto
        data_dicts = []
        for i in range(len(data)):
            # Get response text
            response_ids = data.batch['responses'][i]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            
            # Get extra info
            extra_info = data[i].non_tensor_batch.get('extra_info', {})
            
            data_dict = {
                'response': response,
                'question': extra_info.get('question', ''),
                'ground_truth': extra_info.get('ground_truth', ''),
                'chunk_id': extra_info.get('chunk_id', -1),
                'valid_response_length': len(response_ids),
                'uid': extra_info.get('uid', f'sample_{i}'),
            }
            data_dicts.append(data_dict)
        
        # Compute rewards based on task type
        if problem_type == 'pred_longtext_qa':
            # Solver phase: reward correctness
            print(f"[INFO] Computing solver rewards for {len(data_dicts)} samples")
            answer_scores, format_scores = self._compute_solver_rewards(
                data_dicts, 
                rollout_actor_wg=rollout_actor_wg
            )
            
            # Combine rewards
            answer_weight = self.eval_reward_config.get('answer_weight', 1.0)
            format_weight = self.eval_reward_config.get('format_weight', 0.1)
            
            for i, (ans_score, fmt_score) in enumerate(zip(answer_scores, format_scores)):
                reward = answer_weight * ans_score + format_weight * fmt_score
                valid_length = data_dicts[i]['valid_response_length']
                reward_tensor[i, valid_length - 1] = reward
            
            all_scores['solver_answer_score'] = answer_scores
            all_scores['solver_format_score'] = format_scores
            all_scores['accuracy'] = answer_scores  # For compatibility
            
        elif problem_type == 'gen_longtext_qa':
            # Proposer phase: reward difficulty (1 - solver_correctness)
            print(f"[INFO] Computing proposer rewards for {len(data_dicts)} samples")
            
            # For proposer, we don't have solver scores yet
            # Use neutral 0.5 for now (in real training, this would come from previous solver rollouts)
            solver_scores = [0.5] * len(data_dicts)
            
            rewards, valid_data = self._compute_proposer_rewards(
                data_dicts, 
                solver_scores=solver_scores
            )
            
            # Assign rewards to last token of each sequence
            for i, reward in enumerate(rewards):
                valid_length = data_dicts[i]['valid_response_length']
                reward_tensor[i, valid_length - 1] = reward
            
            all_scores['proposer_reward'] = rewards
            
        else:
            # Unknown problem type - neutral rewards
            print(f"[WARNING] Unknown problem type: {problem_type}, using neutral rewards")
            for i in range(len(data_dicts)):
                valid_length = data_dicts[i]['valid_response_length']
                reward_tensor[i, valid_length - 1] = 0.5
            
            all_scores['accuracy'] = [0.5] * len(data_dicts)
        
        return reward_tensor, all_scores, valid_data
