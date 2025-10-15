# Phase 2: Reward Manager Completion Report

## ✅ Completed Tasks

### File Created: `absolute_zero_reasoner/rewards/longtext_reward_manager.py`

A comprehensive reward manager for long-text continuous learning tasks implementing the three-phase training loop.

---

## 📋 Core Components

### 1. **LongTextQARewardManager Class** (Lines 58-632)

Main reward manager supporting three task types:
- **gen_longtext_qa**: Proposer generates QA pairs from text
- **pred_longtext_qa**: Solver answers questions without text
- **judge_longtext_qa**: Judge evaluates answer quality (optional)

#### Key Initialization Parameters:
```python
- tokenizer: Text processing
- model_name: External LLM for judging (e.g., 'meta/llama-3.1-405b-instruct')
- generation_reward_config: Proposer reward weights
- eval_reward_config: Solver reward weights
- judge_with_actor: Use trained model for judging (vs external LLM)
- train_judge: Whether to train the judge component
- prompt_manager: Dynamic prompt management
- api_key: API key for external LLM
```

---

## 🔧 Core Methods

### 2. **`__call__()`** - Main Entry Point (Lines 541-632)
Dispatches to appropriate reward computation based on task type.

**Flow:**
1. Parse DataProto to extract responses and metadata
2. Determine task type (gen/pred/judge)
3. Call specialized reward computation
4. Return reward tensor, scores dict, and valid data

**Returns:**
- `reward_tensor`: PyTorch tensor with rewards for each token
- `all_scores`: Dictionary with detailed metrics
- `valid_data`: List of validated QA pairs for dataset

---

### 3. **`_compute_proposer_rewards()`** - Proposer Phase (Lines 321-408)

Evaluates QA pair generation quality.

**Reward Formula:**
```
reward = α * question_quality + β * difficulty + γ * answer_quality
```

**Components:**
1. **Question Quality (α = 0.3)**
   - Uses LLM to judge if question requires text knowledge
   - Evaluates clarity, specificity, reasoning depth

2. **Difficulty Score (β = 0.3)**
   - Estimates how challenging the question is
   - Rewards medium difficulty (curriculum learning)
   - Shaped to avoid too-easy or too-hard questions
   - Thresholds: min=0.2, max=0.8

3. **Answer Quality (γ = 0.4)**
   - Judges answer completeness and correctness
   - Checks reasoning and explanations
   - Must be accurate based on text

**Output:**
- Rewards list (one per sample)
- Valid QA pairs for dataset expansion

---

### 4. **`_compute_solver_rewards()`** - Solver Phase (Lines 410-450)

Evaluates answer correctness without seeing the text.

**Reward Formula:**
```
reward = semantic_similarity_weight * llm_judge_score
```

**Process:**
1. Extract generated answer from response
2. Compare with ground truth using LLM judge
3. Weight and normalize score

**Key Feature:** Tests knowledge retention - model must answer correctly based on learned knowledge from SFT warmup, not from seeing the text.

---

### 5. **`_compute_judge_rewards()`** - Judge Phase (Lines 452-481)

Rewards for training the judge itself (optional).

**Reward Criteria:**
- Valid format: Contains `<score></score>` tags
- Reasonable range: Score between 1-10
- High reward (1.0) for correct format
- Low reward (0.1) for invalid format

**Purpose:** Enables self-judging without external LLM dependency.

---

### 6. **`_judge_with_llm()`** - LLM Evaluation (Lines 198-278)

Core judging functionality using external LLM or actor model.

**Supported Judge Types:**
- **'question'**: Evaluate question quality
  - Clarity, specificity
  - Requires text context
  - Has clear answer

- **'answer'**: Evaluate answer quality
  - Correctness vs ground truth
  - Completeness (covers all key points)
  - Clarity and structure
  - Reasoning quality

**Process:**
1. Build evaluation prompt using template
2. Call LLM (external API or actor model)
3. Extract score from `<score></score>` tags
4. Normalize to 0-1 range

**Robustness:**
- Fallback to default score (0.5) if LLM fails
- Handles various score formats (int, float, fraction)
- Error logging for debugging

---

### 7. **`_estimate_difficulty()`** - Difficulty Estimation (Lines 280-319)

Estimates question difficulty for curriculum learning.

**Current Heuristic:**
```python
difficulty = 0.5 * length_factor + 0.5 * reasoning_factor

where:
  length_factor = (answer_length + question_length) / 100.0
  reasoning_factor = count(reasoning_keywords) / 3.0
```

**Reasoning Keywords:**
- 'because', 'therefore', 'thus'
- 'however', 'although', 'consequently'

**Future Enhancement:** Sample actual solver attempts and measure success rate (1 - average_success = difficulty).

**Range:** Clamped between 0.1 and 0.9

---

### 8. **`extract_score_from_tags()`** - Score Parsing (Lines 160-196)

Robust extraction of numerical scores from LLM responses.

**Supported Formats:**
1. **Fractions**: `<score>8/10</score>` → 0.8
2. **Floats**: `<score>7.5</score>` → 7.5
3. **Integers**: `<score>9</score>` → 9.0

**Features:**
- Case-insensitive tag matching
- Handles multiple scores in one response
- Priority: fraction > float > integer
- Robust to malformed output

---

## 🎯 Design Decisions

### 1. **Reward Configuration**

**Default Proposer Config:**
```python
{
    'question_quality_weight': 0.3,
    'question_difficulty_weight': 0.3,
    'answer_quality_weight': 0.4,
    'min_difficulty_threshold': 0.2,
    'max_difficulty_threshold': 0.8,
}
```
- Balanced weights between quality dimensions
- Difficulty shaping encourages medium-hard questions
- Answer quality weighted highest (ensures correctness)

**Default Solver Config:**
```python
{
    'exact_match_weight': 0.0,      # Not used (too strict)
    'semantic_similarity_weight': 1.0,
    'llm_judge_weight': 1.0,
}
```
- Focuses on semantic correctness, not exact wording
- LLM judge provides nuanced evaluation

---

### 2. **Difficulty Shaping**

Implements **curriculum learning** by rewarding medium-difficulty questions:

```
if difficulty < 0.2:  → Too easy → Low reward (0.5)
if 0.2 ≤ difficulty ≤ 0.8:  → Sweet spot → High reward (1.0)  
if difficulty > 0.8:  → Too hard → Low reward (0.5)
```

**Rationale:**
- Too-easy questions don't teach much
- Too-hard questions frustrate learning
- Medium difficulty maximizes learning efficiency

---

### 3. **External LLM Integration**

Uses NVIDIA NIM API for high-quality judging:
- Model: `meta/llama-3.1-405b-instruct` (default)
- Provides consistent, reliable evaluations
- Can be replaced with self-judging (actor model) after training

**Fallback Strategy:**
```python
if not self.client:
    return 0.5  # Neutral score
```

---

### 4. **Token-Level Reward Assignment**

Rewards assigned to **last token** of each sequence:
```python
reward_tensor[i, valid_response_length - 1] = reward
```

**Why:** PPO training uses advantage estimation from final token to propagate credit throughout sequence.

---

## 📊 Reward Examples

### Example 1: High-Quality Proposer Output

**Question:** "What are the three main factors that contributed to system performance improvement, and how do they interact?"

**Answer:** "The three factors are: (1) memory management, (2) parallel processing, (3) optimized algorithms. They interact synergistically..."

**Scoring:**
- Question Quality: 0.9 (specific, requires text knowledge)
- Difficulty: 0.7 (medium-hard, well-shaped)
- Answer Quality: 0.95 (complete, accurate, well-reasoned)

**Total Reward:** 0.3×0.9 + 0.3×1.0 + 0.4×0.95 = 0.27 + 0.30 + 0.38 = **0.95**

---

### Example 2: Poor Proposer Output

**Question:** "What is the system?"

**Answer:** "It's a system."

**Scoring:**
- Question Quality: 0.2 (vague, not specific)
- Difficulty: 0.1 (too easy, low shaped score = 0.25)
- Answer Quality: 0.15 (circular, uninformative)

**Total Reward:** 0.3×0.2 + 0.3×0.25 + 0.4×0.15 = 0.06 + 0.075 + 0.06 = **0.195**

---

### Example 3: High-Quality Solver Output

**Question:** "Explain the main benefit of the proposed architecture."

**Generated:** "The main benefit is improved memory efficiency through dynamic allocation..."

**Ground Truth:** "The architecture improves memory efficiency via dynamic allocation and reduces overhead..."

**LLM Judge Score:** 8.5/10 = 0.85

**Total Reward:** 1.0 × 0.85 = **0.85**

---

## 🔗 Integration Points

### With Data Construction:
```python
from absolute_zero_reasoner.data_construction.constructor import extract_qa_pair
```
- Uses `extract_qa_pair()` to parse model outputs
- Validates QA pair extraction success

### With Prompts:
```python
from absolute_zero_reasoner.data_construction.prompts import get_longtext_judge_prompt
```
- Uses judge prompts for LLM evaluation
- Supports dynamic prompt management

### With Trainer:
- Called by `LongTextQARayPPOTrainer` during rollout evaluation
- Returns rewards for PPO advantage computation
- Provides valid data for dataset expansion

---

## 📈 Code Statistics

- **Total Lines:** ~632 lines
- **Main Class:** 1 (LongTextQARewardManager)
- **Core Methods:** 8
  - `__call__`: Main entry
  - `_compute_proposer_rewards`: Proposer evaluation
  - `_compute_solver_rewards`: Solver evaluation  
  - `_compute_judge_rewards`: Judge training
  - `_judge_with_llm`: LLM-based judging
  - `_estimate_difficulty`: Difficulty estimation
  - `extract_score_from_tags`: Score parsing
  - `set_prompt_manager`: Dynamic prompt support

---

## 🎓 Key Features

### 1. **Multi-Phase Support**
Unified interface for three training phases with specialized reward logic for each.

### 2. **Flexible Judging**
Supports both external LLM (NVIDIA API) and self-judging (actor model).

### 3. **Curriculum Learning**
Difficulty shaping encourages progressively challenging questions.

### 4. **Robust Parsing**
Handles various output formats, falls back gracefully on errors.

### 5. **Configurable Weights**
Easy tuning of reward components via config dictionaries.

### 6. **Comprehensive Logging**
Debug prints for monitoring reward computation.

---

## 🚀 Next Steps

After Phase 2 completion:

### **Phase 3: Trainer Implementation**
Create `LongTextQARayPPOTrainer` that:
- Loads long text and manages chunks
- Creates dataloaders using constructor functions
- Calls this reward manager during training
- Implements the full training loop

---

## ✅ Summary

✅ **Phase 2 COMPLETE**
- Full reward manager implementation
- Support for all three training phases
- Flexible LLM judging infrastructure
- Curriculum learning via difficulty shaping
- Production-ready error handling

**Progress: 18/90 tasks complete (20%)**

**Files Modified:** 1 new file created

**Lines Added:** ~632 lines

Ready for Phase 3: Trainer Implementation!

