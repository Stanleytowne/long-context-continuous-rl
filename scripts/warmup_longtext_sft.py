#!/usr/bin/env python3
"""
Long-Text SFT Warmup Script.

This script prepares a model for long-text continuous learning by:
1. Loading a long text document
2. Generating high-quality QA pairs using a strong model
3. Fine-tuning the target model on these QA pairs (SFT warmup)

This warmup phase ensures the model learns to:
- Answer questions based on the text content
- Maintain proper output format
- Build initial knowledge from the long text
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from omegaconf import DictConfig, OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from verl.trainer.fsdp_sft_trainer import run_sft
from verl.utils import hf_tokenizer
from absolute_zero_reasoner.data_construction.constructor import chunk_long_text, extract_qa_pair
from absolute_zero_reasoner.data_construction.prompts import get_longtext_proposer_prompt


def load_long_text(file_path: str) -> str:
    """
    Load long text from various file formats.
    
    Args:
        file_path: Path to text file
        
    Returns:
        Text content as string
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Long text file not found: {file_path}")
    
    _, ext = os.path.splitext(file_path)
    
    try:
        if ext in ['.txt', '.md', '.rst']:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
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
        elif ext == '.jsonl':
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [json.loads(line) for line in f]
                text = '\n\n'.join([
                    line.get('text', '') or line.get('content', '') or str(line)
                    for line in lines
                ])
        else:
            # Default: treat as plain text
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        
        return text.strip()
        
    except Exception as e:
        raise RuntimeError(f"Failed to load long text from {file_path}: {e}")


def generate_qa_pairs_with_local_model(
    text_chunks: List[Dict],
    model_path: str,
    tokenizer,
    num_pairs_per_chunk: int = 1,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    device: str = "cuda",
) -> List[Dict]:
    """
    Generate QA pairs from text chunks using a local model.
    
    Args:
        text_chunks: List of text chunk dictionaries
        model_path: Path to local model
        tokenizer: Tokenizer for the model
        num_pairs_per_chunk: Number of QA pairs to generate per chunk
        temperature: Sampling temperature
        max_tokens: Maximum tokens for generation
        device: Device to run model on
        
    Returns:
        List of QA pair dictionaries with 'question' and 'answer'
    """
    import torch
    from transformers import AutoModelForCausalLM
    
    print(f"[INFO] Generating QA pairs using local model: {model_path}")
    print(f"[INFO] Processing {len(text_chunks)} text chunks, {num_pairs_per_chunk} pairs per chunk...")
    
    # Load model
    print(f"[INFO] Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    
    all_qa_pairs = []
    
    for i, chunk_dict in enumerate(text_chunks):
        chunk_text = chunk_dict['text']
        chunk_id = chunk_dict.get('chunk_id', i)
        
        print(f"[INFO] Processing chunk {i+1}/{len(text_chunks)} (ID: {chunk_id})...")
        
        # Generate multiple QA pairs for this chunk
        for pair_idx in range(num_pairs_per_chunk):
            print(f"  Generating QA pair {pair_idx+1}/{num_pairs_per_chunk}...")
            
            # Create prompt for QA generation
            prompt = get_longtext_proposer_prompt(
                text_segment=chunk_text,
            )
            
            try:
                # Tokenize
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = tokenizer([text], return_tensors="pt")
                
                # Move to specified device
                if device != "auto":
                    model_inputs = model_inputs.to(device)
                
                # Generate
                with torch.no_grad():
                    generated_ids = model.generate(
                        model_inputs.input_ids,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=True,
                        top_p=0.95,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                
                # Decode
                generated_ids = [
                    output_ids[len(input_ids):] 
                    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # Extract QA pair
                qa_pair = extract_qa_pair(generated_text)
                
                if qa_pair['extraction_success']:
                    qa_pair['chunk_id'] = chunk_id
                    qa_pair['pair_idx'] = pair_idx
                    qa_pair['source_chunk'] = chunk_text[:200] + "..."  # Store snippet
                    all_qa_pairs.append(qa_pair)
                    print(f"    ✓ Generated QA pair {pair_idx+1} successfully")
                else:
                    print(f"    ✗ Failed to extract QA pair {pair_idx+1} from response")
                    print(f"    Generated text: {generated_text[:200]}...")
                    
            except Exception as e:
                print(f"    ✗ Error generating QA pair {pair_idx+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n[INFO] Generated {len(all_qa_pairs)} valid QA pairs total")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return all_qa_pairs


def generate_qa_pairs_with_llm(
    text_chunks: List[Dict],
    model_name: str = "meta/llama-3.1-405b-instruct",
    num_pairs_per_chunk: int = 3,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    api_key: str = None,
) -> List[Dict]:
    """
    Generate QA pairs from text chunks using an external LLM.
    
    Args:
        text_chunks: List of text chunk dictionaries
        model_name: Model to use for generation
        num_pairs_per_chunk: Number of QA pairs to generate per chunk
        temperature: Sampling temperature
        max_tokens: Maximum tokens for generation
        api_key: API key for external LLM
        
    Returns:
        List of QA pair dictionaries with 'question' and 'answer'
    """
    from openai import OpenAI
    
    print(f"[INFO] Generating QA pairs using {model_name}...")
    print(f"[INFO] Processing {len(text_chunks)} text chunks, {num_pairs_per_chunk} pairs per chunk...")
    
    # Initialize client
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key or "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"
    )
    
    all_qa_pairs = []
    
    for i, chunk_dict in enumerate(text_chunks):
        chunk_text = chunk_dict['text']
        chunk_id = chunk_dict.get('chunk_id', i)
        
        print(f"[INFO] Processing chunk {i+1}/{len(text_chunks)} (ID: {chunk_id})...")
        
        # Generate multiple QA pairs for this chunk
        for pair_idx in range(num_pairs_per_chunk):
            print(f"  Generating QA pair {pair_idx+1}/{num_pairs_per_chunk}...")
            
            # Create prompt for QA generation
            prompt = get_longtext_proposer_prompt(
                text_segment=chunk_text,
            )
            
            try:
                # Generate QA pairs
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=0.95,
                )
                
                generated_text = response.choices[0].message.content
                
                # Extract QA pair
                qa_pair = extract_qa_pair(generated_text)
                
                if qa_pair['extraction_success']:
                    qa_pair['chunk_id'] = chunk_id
                    qa_pair['pair_idx'] = pair_idx
                    qa_pair['source_chunk'] = chunk_text[:200] + "..."  # Store snippet
                    all_qa_pairs.append(qa_pair)
                    print(f"    ✓ Generated QA pair {pair_idx+1} successfully")
                else:
                    print(f"    ✗ Failed to extract QA pair {pair_idx+1} from response")
                    
            except Exception as e:
                print(f"    ✗ Error generating QA pair {pair_idx+1}: {e}")
                continue
    
    print(f"\n[INFO] Generated {len(all_qa_pairs)} valid QA pairs total")
    return all_qa_pairs


def create_sft_dataset_from_longtext(
    long_text_path: str,
    output_path: str,
    chunk_size: int = 2048,
    overlap: int = 256,
    num_pairs_per_chunk: int = 3,
    generation_model: str = "meta/llama-3.1-405b-instruct",
    temperature: float = 0.7,
    tokenizer = None,
    api_key: str = None,
    use_local_model: bool = False,
    local_model_path: str = None,
    local_model_device: str = "auto",
) -> Path:
    """
    Create SFT warmup dataset from long text.
    
    Strategy: Use a strong model to generate QA pairs from text chunks.
    
    Args:
        long_text_path: Path to long text file
        output_path: Path to save SFT dataset
        chunk_size: Maximum tokens per chunk
        overlap: Overlap between chunks
        num_pairs_per_chunk: QA pairs to generate per chunk
        generation_model: Model name for external LLM (if use_local_model=False)
        temperature: Sampling temperature
        tokenizer: Tokenizer for chunking
        api_key: API key for external LLM
        use_local_model: Whether to use local model instead of API
        local_model_path: Path to local model (if use_local_model=True)
        local_model_device: Device for local model ('auto', 'cuda', 'cuda:0', 'cpu', etc.)
        
    Returns:
        Path to saved dataset
    """
    print("="*80)
    print("Creating SFT warmup dataset from long text")
    print("="*80)
    
    # Load long text
    print(f"\n[1/4] Loading long text from: {long_text_path}")
    long_text = load_long_text(long_text_path)
    print(f"  Text length: {len(long_text)} characters")
    
    # Chunk text
    print(f"\n[2/4] Chunking text (chunk_size={chunk_size}, overlap={overlap})")
    text_chunks = chunk_long_text(
        text=long_text,
        chunk_size=chunk_size,
        overlap=overlap,
        tokenizer=tokenizer
    )
    print(f"  Created {len(text_chunks)} chunks")
    
    # Generate QA pairs
    if use_local_model:
        # Use local model
        if not local_model_path:
            raise ValueError("local_model_path must be provided when use_local_model=True")
        
        print(f"\n[3/4] Generating QA pairs with local model: {local_model_path}")
        print(f"  Device: {local_model_device}")
        qa_pairs = generate_qa_pairs_with_local_model(
            text_chunks=text_chunks,
            model_path=local_model_path,
            tokenizer=tokenizer,
            num_pairs_per_chunk=num_pairs_per_chunk,
            temperature=temperature,
            device=local_model_device,
        )
    else:
        # Use external LLM API
        print(f"\n[3/4] Generating QA pairs with external LLM: {generation_model}")
        qa_pairs = generate_qa_pairs_with_llm(
            text_chunks=text_chunks,
            model_name=generation_model,
            num_pairs_per_chunk=num_pairs_per_chunk,
            temperature=temperature,
            api_key=api_key,
        )
    
    if not qa_pairs:
        raise ValueError("Failed to generate any valid QA pairs")
    
    # Convert to dataframe
    print(f"\n[4/4] Saving dataset to: {output_path}")
    df = pd.DataFrame([
        {
            'prompt': qa['question'],
            'response': qa['answer'],
            'chunk_id': qa.get('chunk_id', -1),
        }
        for qa in qa_pairs
    ])
    
    # Save to parquet
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    print(f"  ✓ Saved {len(df)} QA pairs")
    print(f"\n{'='*80}")
    print("SFT warmup dataset created successfully!")
    print(f"{'='*80}\n")
    
    return output_path


def create_longtext_sft_config(
    model_path: str,
    train_data_path: str,
    val_data_path: Optional[str] = None,
    output_dir: str = "checkpoints/warmup_longtext_sft",
    epochs: int = 2,
    batch_size: int = 256,
    micro_batch_size: int = 4,
    learning_rate: float = 5e-6,
    max_length: int = 2048,
    save_freq: int = 100,
    test_freq: int = 50,
    n_gpus: int = 8,
    lora_rank: int = 0,
):
    """Create configuration for long-text SFT warmup training."""
    
    if val_data_path is None:
        val_data_path = train_data_path
    
    config = {
        'data': {
            'train_batch_size': batch_size,
            'micro_batch_size_per_gpu': micro_batch_size,
            'train_files': train_data_path,
            'val_files': val_data_path,
            'prompt_key': 'prompt',
            'response_key': 'response',
            'max_length': max_length,
            'truncation': 'error',
            'balance_dp_token': False,
            'chat_template': None,
            'custom_cls': {
                'path': None,
                'name': None,
            },
            'use_shm': False,
        },
        'model': {
            'partial_pretrain': model_path,
            'use_shm': False,
            'fsdp_config': {
                'model_dtype': 'bf16',
                'wrap_policy': {
                    'min_num_params': 0,
                },
                'cpu_offload': False,
                'offload_params': False,
            },
            'external_lib': None,
            'enable_gradient_checkpointing': True,
            'trust_remote_code': True,
            'lora_rank': lora_rank,
            'lora_alpha': 16,
            'target_modules': 'all-linear',
            'use_liger': False,
            'strategy': 'fsdp2',
        },
        'optim': {
            'lr': learning_rate,
            'betas': [0.9, 0.95],
            'weight_decay': 0.01,
            'warmup_steps_ratio': 0.1,
            'clip_grad': 1.0,
            'lr_scheduler': 'cosine',
        },
        'ulysses_sequence_parallel_size': 1,
        'use_remove_padding': False,
        'trainer': {
            'default_local_dir': output_dir,
            'default_hdfs_dir': None,
            'resume_path': None,
            'project_name': 'warmup-longtext-sft',
            'experiment_name': 'warmup-longtext',
            'total_epochs': epochs,
            'total_training_steps': None,
            'logger': ['console'],
            'seed': 42,
            'save_freq': save_freq,
            'test_freq': test_freq,
            'nnodes': 1,
            'n_gpus_per_node': n_gpus,
            'max_ckpt_to_keep': 3,
        },
    }
    
    return DictConfig(config)


def main():
    """
    Main entry point for long-text SFT warmup.
    
    Usage:
        python scripts/warmup_longtext_sft.py \
            --model_path Qwen/Qwen2.5-7B \
            --long_text_path /path/to/document.txt \
            --output_dir checkpoints/warmup_longtext_sft
    """
    import argparse
    parser = argparse.ArgumentParser(description="Long-Text SFT Warmup using VERL")
    
    # Model and data paths
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the pretrained model')
    parser.add_argument('--long_text_path', type=str, required=True,
                       help='Path to long text file (.txt, .md, .json, .jsonl)')
    parser.add_argument('--output_dir', type=str, default='checkpoints/warmup_longtext_sft',
                       help='Directory to save checkpoints and datasets')
    
    # Long-text processing parameters
    parser.add_argument('--chunk_size', type=int, default=2048,
                       help='Maximum tokens per chunk')
    parser.add_argument('--overlap', type=int, default=256,
                       help='Overlap between chunks')
    parser.add_argument('--num_pairs_per_chunk', type=int, default=3,
                       help='Number of QA pairs to generate per chunk')
    
    # QA generation options
    parser.add_argument('--use_local_model', action='store_true',
                       help='Use local model for QA generation instead of external API')
    parser.add_argument('--local_model_device', type=str, default='auto',
                       help='Device for local model (auto, cuda, cuda:0, cpu, etc.)')
    parser.add_argument('--generation_model', type=str, default='meta/llama-3.1-405b-instruct',
                       help='External LLM model name for QA generation (if not using local model)')
    parser.add_argument('--api_key', type=str, default=None,
                       help='API key for external LLM (if not using local model)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature for QA generation')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=2,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Total batch size')
    parser.add_argument('--micro_batch_size', type=int, default=4,
                       help='Micro batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=5e-6,
                       help='Learning rate')
    parser.add_argument('--max_length', type=int, default=2048,
                       help='Maximum sequence length')
    
    # Hardware configuration
    parser.add_argument('--n_gpus', type=int, default=8,
                       help='Number of GPUs to use')
    
    # Checkpointing
    parser.add_argument('--save_freq', type=int, default=100,
                       help='Save checkpoint every N steps')
    parser.add_argument('--test_freq', type=int, default=50,
                       help='Validate every N steps')
    
    # LoRA configuration (optional)
    parser.add_argument('--lora_rank', type=int, default=0,
                       help='LoRA rank (0 to disable, 32/64 for LoRA)')
    
    # Force regenerate dataset
    parser.add_argument('--force_regenerate', action='store_true',
                       help='Force regenerate QA dataset even if it exists')
    
    args = parser.parse_args()
    
    # Check if we're in distributed training
    import torch.distributed as dist
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    is_main_process = local_rank == 0
    
    # Check if running in distributed mode (will be initialized by torchrun/run_sft)
    # For dataset generation, we only care about local_rank
    is_distributed = 'LOCAL_RANK' in os.environ
    
    if is_main_process:
        print("\n" + "="*80)
        print("Long-Text SFT Warmup Script")
        print("="*80 + "\n")
    
    # Output paths
    warmup_data_dir = Path(args.output_dir) / "warmup_data"
    if is_main_process:
        warmup_data_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_path = warmup_data_dir / "longtext_warmup.parquet"
    
    # Synchronize all processes before continuing
    # Use file-based synchronization since distributed may not be initialized yet
    dataset_ready_marker = warmup_data_dir / ".dataset_ready"
    
    # Only main process generates the dataset
    if is_main_process:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = hf_tokenizer(args.model_path, trust_remote_code=True)
        print(f"  Tokenizer loaded: {args.model_path}")
        print()
        
        # Create SFT dataset from long text
        if not dataset_path.exists() or args.force_regenerate:
            if args.force_regenerate and dataset_path.exists():
                print(f"Force regenerating dataset (existing file will be overwritten)...")
                # Remove old marker file
                if dataset_ready_marker.exists():
                    dataset_ready_marker.unlink()
            print("Creating SFT warmup dataset...")
            print("[INFO] Running on main process (rank 0) only for dataset generation")
            create_sft_dataset_from_longtext(
                long_text_path=args.long_text_path,
                output_path=str(dataset_path),
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                num_pairs_per_chunk=args.num_pairs_per_chunk,
                generation_model=args.generation_model,
                temperature=args.temperature,
                tokenizer=tokenizer,
                api_key=args.api_key,
                use_local_model=args.use_local_model,
                local_model_path=args.model_path if args.use_local_model else None,
                local_model_device=args.local_model_device,
            )
        else:
            print(f"Using existing warmup dataset: {dataset_path}")
            df = pd.read_parquet(dataset_path)
            print(f"  Dataset size: {len(df)} examples")
            print()
        
        # Main process: create marker file after dataset is ready
        dataset_ready_marker.touch()
    
    if is_distributed and not is_main_process:
        # Other processes: wait for marker file
        print(f"[INFO] Rank {local_rank}: Waiting for main process to generate dataset...")
        import time
        max_wait_time = 3600  # 1 hour
        wait_interval = 5
        elapsed_time = 0
        while not dataset_ready_marker.exists() and elapsed_time < max_wait_time:
            time.sleep(wait_interval)
            elapsed_time += wait_interval
        
        if not dataset_ready_marker.exists():
            raise TimeoutError(f"Timeout waiting for dataset generation after {max_wait_time} seconds")
        
        print(f"[INFO] Rank {local_rank}: Dataset generation complete, proceeding to training...")
    
    # Verify dataset exists (all processes)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Main process may have failed to generate it.")
    
    if not is_main_process:
        # Load tokenizer for non-main processes (needed for config)
        tokenizer = hf_tokenizer(args.model_path, trust_remote_code=True)
    
    # Create configuration for SFT training
    config = create_longtext_sft_config(
        model_path=args.model_path,
        train_data_path=str(dataset_path),
        val_data_path=None,  # Will use train_data_path for validation
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        save_freq=args.save_freq,
        test_freq=args.test_freq,
        n_gpus=args.n_gpus,
        lora_rank=args.lora_rank,
    )
    
    # Print configuration (only main process)
    if is_main_process:
        print("\n" + "="*80)
        print("SFT Training Configuration")
        print("="*80)
        print(OmegaConf.to_yaml(config))
        print("="*80 + "\n")
        
        # Run SFT training
        print("\n" + "="*80)
        print("Starting SFT Training")
        print("="*80 + "\n")
    
    try:
        run_sft(config)
        if is_main_process:
            print("\n" + "="*80)
            print("SFT Warmup Completed Successfully!")
            print(f"Checkpoints saved to: {args.output_dir}")
            print("="*80 + "\n")
    except Exception as e:
        if is_main_process:
            print(f"\n[ERROR] SFT training failed: {e}")
            import traceback
            traceback.print_exc()
        raise


if __name__ == '__main__':
    main()

