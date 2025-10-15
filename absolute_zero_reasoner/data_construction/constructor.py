from typing import List, Dict

from numpy import random
import pandas as pd
from transformers import AutoTokenizer

from absolute_zero_reasoner.data_construction.prompts import get_code_problem_generator_prompt, get_code_problem_predictor_prompt, get_general_generator_prompt,get_general_generation_with_reference_prompt, get_general_predictor_prompt, get_general_judger_prompt
from absolute_zero_reasoner.data_construction.process_data import boxed_instruction, instruction_following
from absolute_zero_reasoner.utils.code_utils.parsers import replace_main_function_name

def extract_question(text: str) -> str:
    """
    Extract the question part from the text.
    Assumes the question is enclosed in <question> tags.
    """
    start = text.find('<question>') + len('<question>')
    end = text.find('</question>', start)
    return text[start:end].strip() if start != -1 and end != -1 else text.strip()

import numpy as np
import pandas as pd

def get_gen_general_io_data(
    io_data: List[Dict],
    target_data_len: int,
    content_max_length: int,
    io_n: int,
    output_path: str,
    split: str,
    tokenizer,  # 不强依赖类型声明，避免导入问题
    weights: List[float] = None,
    include_references: float = 1.0,
    with_answer_generation: bool = True,
    prompt_manager = None,  # Add prompt manager parameter
):
    return_io_data = []

    # 兜底：空数据直接写空表并返回
    if not io_data:
        pd.DataFrame(return_io_data).to_parquet(output_path)
        return

    # 概率分布
    if weights is None:
        probabilities = np.full(len(io_data), 1.0 / len(io_data))
    else:
        w = np.asarray(weights, dtype=float)
        s = w.sum()
        if s <= 0 or len(w) != len(io_data) or not np.isfinite(s):
            probabilities = np.full(len(io_data), 1.0 / len(io_data))
        else:
            probabilities = w / s

    # 解析 include_references 为概率 p 
    try:
        if isinstance(include_references, bool):
            p_include = 1.0 if include_references else 0.0
        else:
            p_include = float(include_references)
        # clamp 到 [0,1]
        p_include = max(0.0, min(1.0, p_include))
    except Exception:
        p_include = 1.0
    print(f"[DEBUG] get_gen_general_io_data: include_references probability = {p_include}")

    idx = 0
    max_attempts = max(5 * target_data_len, 100)  # 防止无限循环
    attempts = 0

    while len(return_io_data) < target_data_len and attempts < max_attempts:
        attempts += 1

        # 本轮是否包含参考：按概率决定
        include_refs_this_round = (np.random.rand() < p_include)

            # Use dynamic prompt if prompt_manager is available
        if prompt_manager:
            if include_refs_this_round:
                instruction_template = prompt_manager.get_proposer_instruction(ref=True, with_answer_generation=with_answer_generation)
            else:
                instruction_template = prompt_manager.get_proposer_instruction(ref=False, with_answer_generation=with_answer_generation)
            print(f"[DEBUG] get_gen_general_io_data: Using dynamic proposer instruction")
        else:
            instruction_template = '{}'
            print(f"[DEBUG] get_gen_general_io_data: Using default instruction template")

        if not include_refs_this_round:
            chosen_references = []
        else:
            k = min(io_n, len(io_data))
            # 用 numpy 的 choice，并转成 Python list
            chosen_indices = np.random.choice(len(io_data), size=k, replace=False, p=probabilities)
            chosen_references = [io_data[i] for i in chosen_indices]
        if not chosen_references:
            if prompt_manager:
                # Use the enhanced proposer instruction directly from prompt_manager
                io_prompt = instruction_template
            else:
                io_prompt = instruction_template.format(
                    get_general_generator_prompt(reference_questions=chosen_references)
                )
        else:
            if prompt_manager:
                # Use the enhanced proposer instruction directly from prompt_manager
                # But we need to add reference questions to it
                base_instruction = instruction_template
                reference_section = get_general_generation_with_reference_prompt(reference_questions=chosen_references)
                # Extract the reference questions part from the reference_section
                if "### Reference Questions:" in reference_section:
                    ref_part = reference_section.split("### Reference Questions:")[1]
                    io_prompt = base_instruction + "\n### Reference Questions:" + ref_part + "\n### Reference Questions Ends Here"
                else:
                    io_prompt = base_instruction
            else:
                io_prompt = instruction_template.format(
                    get_general_generation_with_reference_prompt(reference_questions=chosen_references)
                )
        # 提取 question
        # question = extract_question(io_prompt.split('### Your Task:')[1].strip())
        question = io_prompt
        # if not question:
        #     # print("No question found in the generated prompt, skipping this item.")
        #     continue

        # 显示给actor的prompt日志
        from absolute_zero_reasoner.utils.logging_utils.stdout import PrettyPrinter
        PrettyPrinter.section_header(f"🤖 Gen_General Proposer Prompt for Actor (Item {idx+1})")
        PrettyPrinter.code_block(f"Prompt Content:\n{io_prompt}")
        print(f"[GEN_GENERAL_LOG] Prompt length: {len(tokenizer(io_prompt)['input_ids'])} tokens")
        print(f"[GEN_GENERAL_LOG] Using prompt_manager: {prompt_manager is not None}")
        if prompt_manager:
            print(f"[GEN_GENERAL_LOG] Enhanced proposer instruction with question-answer verification enabled")
        
        # 过滤过长样本
        if len(tokenizer(io_prompt)['input_ids']) <= content_max_length:
            io_item = {
                "data_source": 'gen_general',
                "prompt": [{"role": "user", "content": io_prompt}],
                "question": question,
                "answer": "",
                "ability": "general",
                "reward_model": {"style": "rule", "ground_truth": ''},
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'metric': 'gen_general',
                    # 确保可序列化
                    'chosen_references': chosen_references,
                }
            }
            return_io_data.append(io_item)
            idx += 1

    # 不足就上采样补齐（前提：io_data 非空）
    while len(return_io_data) < target_data_len:
        j = np.random.randint(0, len(io_data))  # 上界开区间，不会越界
        return_io_data.append(io_data[j])

    # 输出到 parquet
    pd.DataFrame(return_io_data).to_parquet(output_path)

def get_pred_general_io_data(
    io_data: List[Dict],
    target_data_len: int,
    content_max_length: int,
    output_path: str,
    split: str,
    tokenizer: AutoTokenizer,
    prompt_manager = None,  # Add prompt manager parameter
):
    return_io_data = []
    
    # Use dynamic prompt if prompt_manager is available
    if prompt_manager:
        instruction_template = prompt_manager.get_solver_instruction("{}")
        print(f"[DEBUG] get_pred_general_io_data: Using dynamic solver instruction")
    else:
        instruction_template = '{}'
        print(f"[DEBUG] get_pred_general_io_data: Using default instruction template")

    for idx, io_item in enumerate(io_data):
        if prompt_manager:
            # Use prompt manager to get enhanced solver instruction
            io_prompt = prompt_manager.get_solver_instruction(io_item['question'])
        else:
            # Use traditional template approach
            io_prompt = instruction_template.format(
                get_general_predictor_prompt(
                    question=io_item['question'],
                )
            )
        print(f"Generated prompt: {io_prompt}")
        # since we have abundant judge data, we can afford to filter out some data
        if len(tokenizer(io_prompt)['input_ids']) <= content_max_length:
            output_io_item = {
                "data_source": 'pred_general',
                "prompt": [{
                    "role": "user",
                    "content": io_prompt,
                }],
                "question": io_item['question'],
                "answer": "",
                "ability": "general",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": io_item.get('answer',''),
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'metric': 'pred_general',
                }
            }
            return_io_data.append(output_io_item)

        if len(return_io_data) >= target_data_len:
            break

    # if io_data is not full, we sample upsample random data
    while len(return_io_data) < target_data_len:
        io_item = return_io_data[random.randint(0, len(return_io_data))]
        return_io_data.append(io_item)

    # output to parquet
    df = pd.DataFrame(return_io_data)
    df.to_parquet(output_path)

def get_judge_general_io_data(
    io_data: List[Dict],
    target_data_len: int,
    content_max_length: int,
    output_path: str,
    split: str,
    tokenizer: AutoTokenizer,
    prompt_manager=None,
):
    return_io_data = []
    instruction_template = '{}'

    for idx, io_item in enumerate(io_data):
        io_prompt = instruction_template.format(
            get_general_judger_prompt(
                question=io_item['question'],
                answer=io_item['answer'],
                prompt_manager=prompt_manager,
            )
        )
        print(f"Generated prompt: {io_prompt}")
        # since we have abundant judge data, we can afford to filter out some data
        if len(tokenizer(io_prompt)['input_ids']) <= content_max_length:
            output_io_item = {
                "data_source": 'judge_general',
                "prompt": [{
                    "role": "user",
                    "content": io_prompt,
                }],
                "question": io_item['question'],
                "answer": io_item['answer'],
                "ability": "general",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": io_item['reward_model']['ground_truth'],
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'metric': 'judge_general',
                }
            }
            return_io_data.append(output_io_item)

        if len(return_io_data) >= target_data_len:
            break

    # if io_data is not full, we sample upsample random data
    while len(return_io_data) < target_data_len:
        io_item = return_io_data[random.randint(0, len(return_io_data))]
        return_io_data.append(io_item)

    # output to parquet
    df = pd.DataFrame(return_io_data)
    df.to_parquet(output_path)

def get_gen_code_io_data(
    io_data: List[Dict],
    target_data_len: int,
    problem_type: str,
    instruction_type: str,
    content_max_length: int,
    io_n: int,
    output_path: str,
    split: str,
    tokenizer: AutoTokenizer,
    banned_keywords: List[str],
    banned_assertion_keywords: List[str],
    weights: List[float] = None,
    enable_composite_function: bool = False,
    composite_function_n_min: int = -1,
    composite_function_n_max: int = -1,
    composite_chance: float = 0.5,
    remove_after_return: bool = False,
    num_inputs: int = 10,
    remove_input_from_snippet: bool = False,
    include_references: bool = True,
):
    return_io_data = []
    if instruction_type.startswith('boxed'):
        instruction_template = boxed_instruction
    elif instruction_type.startswith('answer'):
        instruction_template = instruction_following
    elif instruction_type.startswith('none'):
        instruction_template = '{}'
    else:
        raise ValueError(f"Invalid instruction type: {instruction_type}")

    if weights is None:
        probabilities = [1.0 / len(io_data)] * len(io_data)
    else:
        # Normalize weights to form a probability distribution
        probabilities = [float(w)/sum(weights) for w in weights]
    
    idx = 0

    while len(return_io_data) < target_data_len:
        if not include_references and problem_type != 'code_f':
            chosen_references = []
        else:
            chosen_references = random.choice(io_data, size=min(io_n, len(io_data)), replace=False, p=probabilities)
        # composite functions is not used for code_f problem type
        if problem_type != 'code_f' and composite_function_n_max > 0 and enable_composite_function and random.random() <= composite_chance and len(chosen_references) > composite_function_n_max:
            # TODO: we only allow composite to sample from code snippets without composite functions
            io_without_composite_function_indices = [i for i in range(len(io_data)) if not io_data[i]['composite_functions']]
            io_without_composite_function_data = [io_data[i] for i in io_without_composite_function_indices]
            io_without_composite_function_weights = [probabilities[i] for i in io_without_composite_function_indices]
            # normalize the weights
            io_without_composite_function_probabilities = [w / sum(io_without_composite_function_weights) for w in io_without_composite_function_weights]
            # number of composite functions to sample is either fixed or random
            composite_function_n = composite_function_n_min if composite_function_n_min == composite_function_n_max else random.randint(composite_function_n_min, composite_function_n_max)
            composite_functions = random.choice(io_without_composite_function_data, size=composite_function_n, replace=False, p=io_without_composite_function_probabilities)
            for i, composite_function in enumerate(composite_functions):
                # TODO: need to also replace recursively called composite functions, ignore functions that have f as the last letter, only for function call f()
                composite_functions[i]['snippet'] = replace_main_function_name(composite_function['snippet'], 'f', f'g_{i}')
            imports = []
        else:
            composite_functions = []
            if include_references:
                imports = chosen_references[0]['imports']
            else:
                imports = []
        io_prompt = instruction_template.format(
            get_code_problem_generator_prompt(
                problem_type=problem_type,
                reference_snippets=chosen_references,
                banned_keywords=banned_keywords,
                banned_assertion_keywords=banned_assertion_keywords,
                composite_functions=composite_functions,
                remove_after_return=remove_after_return,
                num_inputs=num_inputs,
                remove_input_from_snippet=remove_input_from_snippet,
            )
        )
        # since we have abundant judge data, we can afford to filter out some data
        if len(tokenizer(io_prompt)['input_ids']) <= content_max_length:
            io_item = {
                "data_source": 'gen_' + problem_type,
                "prompt": [{
                    "role": "user",
                    "content": io_prompt,
                }],
                "problem": '',
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": '',
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'metric': 'gen_' + problem_type,
                    'chosen_references': chosen_references,
                    'composite_functions': composite_functions,
                    'imports': imports,
                }
            }
            return_io_data.append(io_item)
            idx += 1

        if len(return_io_data) >= target_data_len:
            break

    # if io_data is not full, we sample upsample random data
    while len(return_io_data) < target_data_len:
        io_item = io_data[random.randint(0, len(io_data))]
        return_io_data.append(io_item)

    # output to parquet
    df = pd.DataFrame(return_io_data)
    df.to_parquet(output_path)


def get_pred_code_io_data(
    io_data: List[Dict],
    target_data_len: int,
    problem_type: str,
    instruction_type: str,
    content_max_length: int,
    output_path: str,
    split: str,
    tokenizer: AutoTokenizer,
):
    return_io_data = []
    if instruction_type.startswith('boxed'):
        instruction_template = boxed_instruction
    elif instruction_type.startswith('answer'):
        instruction_template = instruction_following
    elif instruction_type.startswith('none'):
        instruction_template = '{}'
    else:
        raise ValueError(f"Invalid instruction type: {instruction_type}")

    for idx, io_item in enumerate(io_data):
        if problem_type == 'code_i':
            ground_truth = io_item['input']
        elif problem_type == 'code_o':
            ground_truth = io_item['output']
        elif problem_type == 'code_e':
            ground_truth = io_item['output']
        elif problem_type == 'code_f':
            ground_truth = io_item['snippet']
        else:
            raise ValueError(f"Invalid problem type: {problem_type}")
        if problem_type == 'code_f':
            num_given_inputs = len(io_item['inputs']) // 2
            num_given_outputs = len(io_item['outputs']) // 2
            given_inputs = list(io_item['inputs'][:num_given_inputs])
            given_outputs = list(io_item['outputs'][:num_given_outputs])
            hidden_inputs = list(io_item['inputs'][num_given_inputs:])
            hidden_outputs = list(io_item['outputs'][num_given_outputs:])
            io_prompt = instruction_template.format(
                get_code_problem_predictor_prompt(
                    problem_type=problem_type,
                    snippet=io_item['snippet'],
                    message=io_item['message'],
                    input_output_pairs=zip(given_inputs, given_outputs),
                )
            )
        else:
            io_prompt = instruction_template.format(
                get_code_problem_predictor_prompt(
                    problem_type=problem_type,
                    snippet=io_item['snippet'],
                    input_args=io_item['input'],
                    output=io_item['output'],
                )
            )
        # since we have abundant judge data, we can afford to filter out some data
        if len(tokenizer(io_prompt)['input_ids']) <= content_max_length:
            output_io_item = {
                "data_source": 'pred_' + problem_type,
                "prompt": [{
                    "role": "user",
                    "content": io_prompt,
                }],
                "problem": io_item['snippet'],
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth,
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'metric': 'pred_' + problem_type,
                    'imports': io_item['imports'],
                }
            }
            if problem_type == 'code_f': # for code_f, we need to split the inputs and outputs into given and hidden, only show part of the inputs and outputs to the model
                output_io_item['extra_info']['given_inputs'] = given_inputs
                output_io_item['extra_info']['given_outputs'] = given_outputs
                output_io_item['extra_info']['hidden_inputs'] = hidden_inputs
                output_io_item['extra_info']['hidden_outputs'] = hidden_outputs
                output_io_item['extra_info']['message'] = io_item['message']
            else:
                output_io_item['extra_info']['input'] = io_item['input']
                output_io_item['extra_info']['output'] = io_item['output']
            return_io_data.append(output_io_item)

        if len(return_io_data) >= target_data_len:
            break

    # if io_data is not full, we sample upsample random data
    while len(return_io_data) < target_data_len:
        io_item = return_io_data[random.randint(0, len(return_io_data))]
        return_io_data.append(io_item)

    # output to parquet
    df = pd.DataFrame(return_io_data)
    df.to_parquet(output_path)


"""
Long-Text Continuous Learning Data Construction
"""

def chunk_long_text(
    text: str,
    chunk_size: int = 2048,
    overlap: int = 256,
    tokenizer = None,
) -> List[Dict[str, any]]:
    """
    Split long text into overlapping chunks for processing.
    
    Args:
        text: The long text to be chunked
        chunk_size: Maximum number of tokens per chunk
        overlap: Number of overlapping tokens between consecutive chunks
        tokenizer: Tokenizer to count tokens (if None, use character approximation)
    
    Returns:
        List of dictionaries containing chunk info:
        - 'text': chunk text
        - 'start_idx': start position in original text
        - 'end_idx': end position in original text
        - 'chunk_id': sequential chunk identifier
    """
    chunks = []
    
    # If no tokenizer, use character-based approximation (4 chars ≈ 1 token)
    if tokenizer is None:
        char_per_token = 4
        chunk_size_chars = chunk_size * char_per_token
        overlap_chars = overlap * char_per_token
        
        start = 0
        chunk_id = 0
        while start < len(text):
            end = min(start + chunk_size_chars, len(text))
            chunk_text = text[start:end]
            
            chunks.append({
                'text': chunk_text,
                'start_idx': start,
                'end_idx': end,
                'chunk_id': chunk_id,
                'token_count': len(chunk_text) // char_per_token,
            })
            
            chunk_id += 1
            start += chunk_size_chars - overlap_chars
            
            if start >= len(text):
                break
    else:
        # Use tokenizer for accurate chunking
        tokens = tokenizer.encode(text)
        start_token = 0
        chunk_id = 0
        
        while start_token < len(tokens):
            end_token = min(start_token + chunk_size, len(tokens))
            chunk_tokens = tokens[start_token:end_token]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            chunks.append({
                'text': chunk_text,
                'start_token_idx': start_token,
                'end_token_idx': end_token,
                'chunk_id': chunk_id,
                'token_count': len(chunk_tokens),
            })
            
            chunk_id += 1
            start_token += chunk_size - overlap
            
            if start_token >= len(tokens):
                break
    
    print(f"[INFO] Chunked long text into {len(chunks)} chunks")
    return chunks


def extract_qa_pair(text: str) -> Dict[str, str]:
    """
    Extract question and answer from model-generated text.
    Expected format: <question>...</question><answer>...</answer>
    
    Args:
        text: Model-generated text containing question and answer tags
    
    Returns:
        Dictionary with 'question' and 'answer' keys, or None if extraction fails
    """
    import re
    
    # Try to extract question
    question_match = re.search(r'<question>(.*?)</question>', text, re.DOTALL | re.IGNORECASE)
    if question_match:
        question = question_match.group(1).strip()
    else:
        # Fallback: look for alternative markers
        question = None
    
    # Try to extract answer
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        # Fallback: look for alternative markers
        answer = None
    
    if question and answer:
        return {
            'question': question,
            'answer': answer,
            'extraction_success': True
        }
    else:
        # If tags not found, try to infer from structure
        # This is a fallback for robustness
        return {
            'question': text[:len(text)//2].strip() if not question else question,
            'answer': text[len(text)//2:].strip() if not answer else answer,
            'extraction_success': False
        }


def get_gen_longtext_qa_data(
    long_text: str,
    target_data_len: int,
    content_max_length: int,
    output_path: str,
    split: str,
    tokenizer,
    chunk_size: int = 2048,
    overlap: int = 256,
    use_chunking: bool = True,
    chunk_sampling_strategy: str = 'random',  # 'random', 'sequential', 'weighted'
    reference_qa_pairs: List[Dict] = None,
    include_references: float = 0.5,
    weights: List[float] = None,
    prompt_manager = None,
):
    """
    Generate question-answer pairs from long text for the Proposer phase.
    The model sees the text and generates QA pairs that require text knowledge.
    
    Args:
        long_text: The source long text to learn from
        target_data_len: Number of QA pairs to generate
        content_max_length: Maximum token length for prompts
        output_path: Path to save the generated parquet file
        split: Data split identifier ('train', 'val', 'test')
        tokenizer: Tokenizer for text processing
        chunk_size: Size of text chunks in tokens
        overlap: Overlap between chunks in tokens
        use_chunking: Whether to chunk the text (True for long texts)
        chunk_sampling_strategy: How to sample chunks
        reference_qa_pairs: Optional existing QA pairs for reference
        include_references: Probability of including reference QA pairs (0.0-1.0)
        weights: Optional sampling weights for chunks
        prompt_manager: Optional prompt manager for dynamic prompts
    
    Returns:
        None (saves data to parquet file)
    """
    from absolute_zero_reasoner.data_construction.prompts import (
        get_longtext_proposer_prompt,
        get_longtext_proposer_with_reference_prompt
    )
    from absolute_zero_reasoner.utils.logging_utils.stdout import PrettyPrinter
    
    return_io_data = []
    
    # Chunk the text if needed
    if use_chunking:
        chunks = chunk_long_text(long_text, chunk_size, overlap, tokenizer)
    else:
        chunks = [{'text': long_text, 'chunk_id': 0, 'token_count': len(tokenizer.encode(long_text))}]
    
    # Initialize sampling probabilities
    if weights is None or len(weights) != len(chunks):
        probabilities = np.full(len(chunks), 1.0 / len(chunks))
    else:
        w = np.asarray(weights, dtype=float)
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            probabilities = np.full(len(chunks), 1.0 / len(chunks))
        else:
            probabilities = w / s
    
    # Parse include_references probability
    try:
        if isinstance(include_references, bool):
            p_include = 1.0 if include_references else 0.0
        else:
            p_include = float(include_references)
        p_include = max(0.0, min(1.0, p_include))
    except Exception:
        p_include = 0.5
    
    print(f"[DEBUG] get_gen_longtext_qa_data: include_references probability = {p_include}")
    print(f"[DEBUG] Processing {len(chunks)} text chunks")
    
    idx = 0
    chunk_idx = 0
    max_attempts = max(5 * target_data_len, 100)
    attempts = 0
    
    while len(return_io_data) < target_data_len and attempts < max_attempts:
        attempts += 1
        
        # Sample a chunk based on strategy
        if chunk_sampling_strategy == 'random':
            chunk = chunks[np.random.choice(len(chunks), p=probabilities)]
        elif chunk_sampling_strategy == 'sequential':
            chunk = chunks[chunk_idx % len(chunks)]
            chunk_idx += 1
        else:  # weighted
            chunk = chunks[np.random.choice(len(chunks), p=probabilities)]
        
        # Decide whether to include reference QA pairs
        include_refs_this_round = (np.random.rand() < p_include) and (reference_qa_pairs is not None) and (len(reference_qa_pairs) > 0)
        
        # Build the prompt
        if prompt_manager:
            if include_refs_this_round:
                instruction_template = prompt_manager.get_proposer_instruction(ref=True, with_answer_generation=True)
            else:
                instruction_template = prompt_manager.get_proposer_instruction(ref=False, with_answer_generation=True)
        else:
            instruction_template = '{}'
        
        # Select reference QA pairs if needed
        if include_refs_this_round:
            k = min(3, len(reference_qa_pairs))  # Up to 3 reference QA pairs
            chosen_refs = np.random.choice(reference_qa_pairs, size=k, replace=False)
            if prompt_manager:
                io_prompt = instruction_template + "\n\n" + get_longtext_proposer_with_reference_prompt(
                    text_segment=chunk['text'],
                    reference_qa_pairs=chosen_refs
                )
            else:
                io_prompt = instruction_template.format(
                    get_longtext_proposer_with_reference_prompt(
                        text_segment=chunk['text'],
                        reference_qa_pairs=chosen_refs
                    )
                )
        else:
            if prompt_manager:
                io_prompt = instruction_template + "\n\n" + get_longtext_proposer_prompt(
                    text_segment=chunk['text']
                )
            else:
                io_prompt = instruction_template.format(
                    get_longtext_proposer_prompt(text_segment=chunk['text'])
                )
        
        # Log prompt for debugging
        PrettyPrinter.section_header(f"🤖 Gen_LongText Proposer Prompt (Item {idx+1})")
        PrettyPrinter.code_block(f"Text segment (first 200 chars): {chunk['text'][:200]}...")
        print(f"[GEN_LONGTEXT_LOG] Prompt length: {len(tokenizer(io_prompt)['input_ids'])} tokens")
        print(f"[GEN_LONGTEXT_LOG] Chunk ID: {chunk['chunk_id']}, Chunk tokens: {chunk['token_count']}")
        
        # Filter out prompts that are too long
        if len(tokenizer(io_prompt)['input_ids']) <= content_max_length:
            io_item = {
                "data_source": 'gen_longtext_qa',
                "prompt": [{"role": "user", "content": io_prompt}],
                "text_segment": chunk['text'],
                "chunk_id": chunk['chunk_id'],
                "question": "",  # To be filled by model
                "answer": "",    # To be filled by model
                "ability": "longtext_qa",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": '',
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'metric': 'gen_longtext_qa',
                    'chunk_id': chunk['chunk_id'],
                    'chunk_token_count': chunk['token_count'],
                    'has_references': include_refs_this_round,
                }
            }
            return_io_data.append(io_item)
            idx += 1
    
    # Upsample if not enough data
    while len(return_io_data) < target_data_len:
        io_item = return_io_data[np.random.randint(0, len(return_io_data))]
        return_io_data.append(io_item)
    
    # Save to parquet
    pd.DataFrame(return_io_data).to_parquet(output_path)
    print(f"[INFO] Saved {len(return_io_data)} gen_longtext_qa items to {output_path}")


def get_pred_longtext_qa_data(
    qa_pairs: List[Dict],
    target_data_len: int,
    content_max_length: int,
    output_path: str,
    split: str,
    tokenizer,
    prompt_manager = None,
):
    """
    Generate prediction tasks from QA pairs for the Solver phase.
    The model does NOT see the text and must answer based on learned knowledge.
    
    Args:
        qa_pairs: List of question-answer pairs generated by proposer
        target_data_len: Number of prediction tasks to create
        content_max_length: Maximum token length for prompts
        output_path: Path to save the generated parquet file
        split: Data split identifier
        tokenizer: Tokenizer for text processing
        prompt_manager: Optional prompt manager for dynamic prompts
    
    Returns:
        None (saves data to parquet file)
    """
    from absolute_zero_reasoner.data_construction.prompts import get_longtext_solver_prompt
    
    return_io_data = []
    
    # Use dynamic prompt if prompt_manager is available
    if prompt_manager:
        instruction_template = prompt_manager.get_solver_instruction("{}")
        print(f"[DEBUG] get_pred_longtext_qa_data: Using dynamic solver instruction")
    else:
        instruction_template = '{}'
        print(f"[DEBUG] get_pred_longtext_qa_data: Using default instruction template")
    
    for idx, qa_pair in enumerate(qa_pairs):
        question = qa_pair.get('question', '')
        ground_truth_answer = qa_pair.get('answer', '')
        
        if not question:
            continue
        
        # Build the prompt (question only, no text)
        if prompt_manager:
            io_prompt = prompt_manager.get_solver_instruction(question)
        else:
            io_prompt = instruction_template.format(
                get_longtext_solver_prompt(question=question)
            )
        
        # Filter out prompts that are too long
        if len(tokenizer(io_prompt)['input_ids']) <= content_max_length:
            output_io_item = {
                "data_source": 'pred_longtext_qa',
                "prompt": [{
                    "role": "user",
                    "content": io_prompt,
                }],
                "question": question,
                "answer": "",  # To be filled by model
                "ability": "longtext_qa",
                "reward_model": {
                    "style": "llm_judge",
                    "ground_truth": ground_truth_answer,
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'metric': 'pred_longtext_qa',
                    'original_chunk_id': qa_pair.get('chunk_id', -1),
                }
            }
            return_io_data.append(output_io_item)
        
        if len(return_io_data) >= target_data_len:
            break
    
    # Upsample if not enough data
    while len(return_io_data) < target_data_len and len(return_io_data) > 0:
        io_item = return_io_data[random.randint(0, len(return_io_data))]
        return_io_data.append(io_item)
    
    # Save to parquet
    df = pd.DataFrame(return_io_data)
    df.to_parquet(output_path)
    print(f"[INFO] Saved {len(return_io_data)} pred_longtext_qa items to {output_path}")


def get_judge_longtext_qa_data(
    qa_pairs_with_answers: List[Dict],
    target_data_len: int,
    content_max_length: int,
    output_path: str,
    split: str,
    tokenizer,
    prompt_manager = None,
):
    """
    Generate judge evaluation tasks for the Judge phase.
    The model compares solver's answer with ground truth.
    
    Args:
        qa_pairs_with_answers: List of QA pairs with both ground truth and solver's answers
        target_data_len: Number of judge tasks to create
        content_max_length: Maximum token length for prompts
        output_path: Path to save the generated parquet file
        split: Data split identifier
        tokenizer: Tokenizer for text processing
        prompt_manager: Optional prompt manager for dynamic prompts
    
    Returns:
        None (saves data to parquet file)
    """
    from absolute_zero_reasoner.data_construction.prompts import get_longtext_judge_prompt
    
    return_io_data = []
    instruction_template = '{}'
    
    for idx, item in enumerate(qa_pairs_with_answers):
        question = item.get('question', '')
        ground_truth = item.get('ground_truth_answer', '')
        generated_answer = item.get('generated_answer', '')
        
        if not question or not ground_truth:
            continue
        
        # Build the judge prompt
        if prompt_manager:
            io_prompt = get_longtext_judge_prompt(
                question=question,
                ground_truth=ground_truth,
                generated_answer=generated_answer,
                prompt_manager=prompt_manager,
            )
        else:
            io_prompt = instruction_template.format(
                get_longtext_judge_prompt(
                    question=question,
                    ground_truth=ground_truth,
                    generated_answer=generated_answer,
                )
            )
        
        # Filter out prompts that are too long
        if len(tokenizer(io_prompt)['input_ids']) <= content_max_length:
            output_io_item = {
                "data_source": 'judge_longtext_qa',
                "prompt": [{
                    "role": "user",
                    "content": io_prompt,
                }],
                "question": question,
                "ground_truth_answer": ground_truth,
                "generated_answer": generated_answer,
                "ability": "longtext_qa",
                "reward_model": {
                    "style": "format",
                    "ground_truth": '',  # Judge just needs to produce a valid score
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'metric': 'judge_longtext_qa',
                }
            }
            return_io_data.append(output_io_item)
        
        if len(return_io_data) >= target_data_len:
            break
    
    # Upsample if not enough data
    while len(return_io_data) < target_data_len and len(return_io_data) > 0:
        io_item = return_io_data[random.randint(0, len(return_io_data))]
        return_io_data.append(io_item)
    
    # Save to parquet
    df = pd.DataFrame(return_io_data)
    df.to_parquet(output_path)
    print(f"[INFO] Saved {len(return_io_data)} judge_longtext_qa items to {output_path}")