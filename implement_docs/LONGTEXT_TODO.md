# Long-Text Continuous Learning Implementation TODO List

## Phase 1: Data Construction Layer ✅ COMPLETE

### 1.1 Constructor Module (`absolute_zero_reasoner/data_construction/constructor.py`) ✅
- [x] Add `get_gen_longtext_qa_data()` - Generate QA pairs from long text (Proposer phase)
- [x] Add `get_pred_longtext_qa_data()` - Create prediction tasks without text (Solver phase)
- [x] Add `get_judge_longtext_qa_data()` - Create judge evaluation tasks
- [x] Add helper function `chunk_long_text()` - Split long text into manageable chunks
- [x] Add helper function `extract_qa_pair()` - Extract question and answer from model output

### 1.2 Prompts Module (`absolute_zero_reasoner/data_construction/prompts.py`) ✅
- [x] Add `longtext_proposer_prompt` - Prompt for generating QA from text
- [x] Add `longtext_proposer_with_reference_prompt` - Proposer with reference QA pairs
- [x] Add `longtext_solver_prompt` - Prompt for answering questions without text
- [x] Add `longtext_judge_prompt` - Prompt for judging answer quality
- [x] Add getter functions: `get_longtext_proposer_prompt()`, `get_longtext_solver_prompt()`, `get_longtext_judge_prompt()`

## Phase 2: Reward Management Layer ✅ COMPLETE

### 2.1 Reward Manager (`absolute_zero_reasoner/rewards/longtext_reward_manager.py`) - NEW FILE ✅
- [x] Create `LongTextQARewardManager` class
- [x] Implement `__init__()` - Initialize with tokenizer, LLM client, config
- [x] Implement `__call__()` - Main reward computation entry point
- [x] Implement `_compute_proposer_rewards()` - Evaluate QA generation quality
  - [x] Question quality scoring (requires text knowledge)
  - [x] Question difficulty estimation (via solver success rate)
  - [x] Answer quality scoring (completeness, accuracy)
- [x] Implement `_compute_solver_rewards()` - Evaluate answer correctness
  - [x] Semantic similarity with ground truth
  - [x] LLM judge scoring
- [x] Implement `_judge_with_llm()` - Use LLM to judge answers
- [x] Implement `_estimate_difficulty()` - Estimate question difficulty
- [x] Implement `_compute_judge_rewards()` - Judge training rewards
- [x] Add support for both external LLM judge and self-judge modes
- [x] Add `extract_score_from_tags()` - Parse LLM scores

### 2.2 Update Reward Managers Registry (`absolute_zero_reasoner/rewards/reward_managers.py`)
- [x] No modification needed - Can be imported directly from new file

## Phase 3: Trainer Layer ✅ COMPLETE

### 3.1 Long-Text Trainer (`absolute_zero_reasoner/trainer/ppo/longtext_ray_trainer.py`) - NEW FILE ✅
- [x] Create `LongTextQARayPPOTrainer` class inheriting from `ReasonRLRayPPOTrainer`
- [x] Implement `__init__()` - Initialize with long text path and config
- [x] Implement `_load_long_text()` - Load and preprocess long text (supports .txt, .md, .json, .jsonl)
- [x] Text chunking integrated in `__init__()` - Uses `chunk_long_text()` from constructor
- [x] Implement `_create_train_gen_dataloader()` - Proposer training data
- [x] Implement `_create_train_pred_dataloader()` - Solver training data
- [x] Implement `_create_train_judge_dataloader()` - Judge training data (optional stub)
- [x] Text chunk management via `chunk_sampling_strategy` config
- [x] Implement `_update_dataset_with_valid_data()` - Dataset management
- [x] Implement `_is_longtext_task()` - Task type detection
- [x] Implement `cleanup()` - Resource cleanup

### 3.2 Update Trainer Registry (`absolute_zero_reasoner/trainer/ppo/reason_rl_ray_trainer.py`)
- [x] Method `_is_longtext_task()` implemented in trainer itself

## Phase 4: Configuration Layer ✅ COMPLETE

### 4.1 Config File (`absolute_zero_reasoner/configs/azr_ppo_trainer_longtext.yaml`) - NEW FILE ✅
- [x] Create base configuration inheriting from general config
- [x] Add `azr.task_type: longtext_qa`
- [x] Add `azr.long_text` section:
  - [x] `path`: Path to long text file
  - [x] `chunk_size`: Maximum tokens per chunk (2048)
  - [x] `overlap`: Overlap between chunks (256)
  - [x] `use_chunking`: Enable/disable chunking
  - [x] `chunk_sampling_strategy`: random/sequential/weighted
- [x] Add `azr.reward.generation_reward_config` (proposer):
  - [x] `question_quality_weight` (0.3)
  - [x] `question_difficulty_weight` (0.3)
  - [x] `answer_quality_weight` (0.4)
  - [x] `min_difficulty_threshold` (0.2)
  - [x] `max_difficulty_threshold` (0.8)
  - [x] `include_references` (0.5)
- [x] Add `azr.reward.eval_reward_config` (solver):
  - [x] `exact_match_weight` (0.0)
  - [x] `semantic_similarity_weight` (1.0)
  - [x] `llm_judge_weight` (1.0)
- [x] Configure data batch sizes (32/64) and lengths (8096/4096)
- [x] Configure PPO hyperparameters (lr=1e-6, clip_ratio=0.2, etc.)
- [x] Add reward_fn LLM judge configuration
- [x] Add trainer configuration (50 epochs, save_freq=20, etc.)

## Phase 5: Main Entry Point ✅ COMPLETE

### 5.1 Main Script (`absolute_zero_reasoner/main_azr_ppo.py`) ✅
- [x] Add `task_type == 'longtext_qa'` branch
- [x] Long text loading handled by trainer (in __init__)
- [x] Initialize `LongTextQARewardManager`
- [x] Initialize `LongTextQARayPPOTrainer` with long_text_path
- [x] Add validation for long text file existence (in trainer __init__)
- [x] Add logging for long text statistics (in trainer __init__: length, chunks, etc.)
- [x] Add wandb tags for tracking
- [x] Import new modules

## Phase 6: SFT Warmup Layer

### 6.1 Warmup Script (`scripts/warmup_longtext_sft.py`) - NEW FILE
- [ ] Create `create_sft_dataset_from_longtext()` function
  - [ ] Strategy 1: Use strong model to generate QA pairs
  - [ ] Strategy 2: Text chunking + summarization
  - [ ] Strategy 3: Key information extraction
- [ ] Implement `run_sft_training()` - Standard SFT training
- [ ] Add configuration options for warmup strategies
- [ ] Add data quality filtering
- [ ] Support multiple warmup epochs

### 6.2 Warmup Shell Script (`scripts/run_warmup_longtext_sft.sh`) - NEW FILE
- [ ] Create bash script for easy warmup execution
- [ ] Add model path configuration
- [ ] Add long text path configuration
- [ ] Add training hyperparameters
- [ ] Add logging and checkpoint settings

## Phase 7: Training Scripts

### 7.1 Self-Play Script (`scripts/selfplay/longtext.sh`) - NEW FILE
- [ ] Create training script for long-text QA
- [ ] Configure model paths
- [ ] Configure long text data path
- [ ] Set batch sizes and training parameters
- [ ] Configure reward weights
- [ ] Add wandb logging settings

### 7.2 Evaluation Script (`scripts/evaluation/eval_longtext.sh`) - NEW FILE
- [ ] Create evaluation script
- [ ] Add test set configuration
- [ ] Add metrics calculation
- [ ] Support multiple evaluation benchmarks

## Phase 8: Utilities and Helpers

### 8.1 Text Processing Utils (`absolute_zero_reasoner/utils/text_utils.py`) - NEW FILE
- [ ] `load_text_file()` - Load text from various formats
- [ ] `clean_text()` - Text preprocessing
- [ ] `chunk_text()` - Smart text chunking with overlap
- [ ] `extract_key_segments()` - Extract important text segments
- [ ] `compute_text_statistics()` - Length, vocabulary, etc.

### 8.2 QA Extraction Utils (`absolute_zero_reasoner/utils/qa_utils.py`) - NEW FILE
- [ ] `extract_question_from_tags()` - Parse <question> tags
- [ ] `extract_answer_from_tags()` - Parse <answer> tags
- [ ] `validate_qa_pair()` - Check QA pair quality
- [ ] `compute_semantic_similarity()` - Compare answers

### 8.3 Evaluation Utils (`absolute_zero_reasoner/utils/eval_utils.py`)
- [ ] Add `evaluate_longtext_knowledge()` - Measure knowledge retention
- [ ] Add `evaluate_longtext_qa()` - QA accuracy metrics
- [ ] Add comparison with RAG baseline

## Phase 9: Documentation

### 9.1 README Updates (`README.md`)
- [ ] Add long-text continuous learning section
- [ ] Add usage examples
- [ ] Add configuration guide
- [ ] Add evaluation instructions

### 9.2 Example Documentation (`docs/longtext_example.md`) - NEW FILE
- [ ] Step-by-step tutorial
- [ ] Example long text file
- [ ] Expected results
- [ ] Troubleshooting guide

## Phase 10: Testing and Validation

### 10.1 Unit Tests
- [ ] Test text chunking functionality
- [ ] Test QA extraction
- [ ] Test reward computation
- [ ] Test data construction

### 10.2 Integration Tests
- [ ] Test full training pipeline
- [ ] Test checkpoint saving/loading
- [ ] Test with different text lengths

### 10.3 Validation
- [ ] Validate against simple synthetic long text
- [ ] Compare knowledge retention vs baseline
- [ ] Validate reward signals are meaningful

---

## Implementation Priority Order

### Critical Path (Must Complete First):
1. ✅ **Phase 1**: Data Construction (constructor.py, prompts.py)
2. **Phase 2**: Reward Management (longtext_reward_manager.py)
3. **Phase 3**: Trainer (longtext_ray_trainer.py)
4. **Phase 4**: Configuration (azr_ppo_trainer_longtext.yaml)
5. **Phase 5**: Main Entry Point (main_azr_ppo.py)

### Secondary Path (Core Functionality):
6. **Phase 6**: SFT Warmup
7. **Phase 7**: Training Scripts
8. **Phase 8**: Utilities

### Optional Path (Enhancement):
9. **Phase 9**: Documentation
10. **Phase 10**: Testing

---

## Current Status
- **Phase 1.1**: In Progress - Modifying constructor.py
- **Total Tasks**: ~90
- **Completed**: 0
- **In Progress**: 1
- **Remaining**: 89

