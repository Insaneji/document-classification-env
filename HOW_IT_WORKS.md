# 🎯 HOW THE OPENENV ENVIRONMENT WORKS

## Executive Overview

This is a **document classification and routing environment** where AI agents learn to classify incoming documents and route them to the correct department. It's like a smart mail sorter that learns to categorize letters!

---

## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    AGENT INTERACTION LOOP                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────┐
    │  1. Agent receives OBSERVATION      │
    │     (document content + features)   │
    └─────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────┐
    │  2. Agent makes DECISION            │
    │     (picks a category)              │
    └─────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────┐
    │  3. Environment steps (REWARD)      │
    │     • Accuracy bonus/penalty        │
    │     • Speed bonus                   │
    │     • Episode progress              │
    └─────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────┐
    │  4. Repeat until episode ends       │
    │     (all documents classified)      │
    └─────────────────────────────────────┘
```

---

## 📋 Complete Workflow

### Phase 1: Initialize Environment
```python
from environment import DocumentClassificationEnv

# Create an easy environment
env = DocumentClassificationEnv(task_difficulty="easy")

# Returns:
# - Action space: Discrete(5) = 5 categories
# - Observation space: Dict with document features
```

### Phase 2: Reset for Episode
```python
observation, info = env.reset()

# Returns observation dict:
# {
#   "document_id": "doc_000000",
#   "content": "My invoice shows an incorrect amount...",
#   "word_count": [45],
#   "has_urgency_markers": [0],
#   "features": [0.12, -0.34, 0.56, ...],  # 100-dim vector
#   "document_index": [0],
#   "total_documents": [100]
# }
```

### Phase 3: Agent Decision
```python
# Agent analyzes the document
action = agent.decide(observation)
# Returns: 1 (Billing category)
```

### Phase 4: Environment Steps
```python
obs, reward, done, truncated, info = env.step(action)

# Returns:
# - obs: Next document observation
# - reward: 1.0 (correct) or -0.5 (incorrect) + speed bonus
# - done: False (more docs) or True (episode complete)
# - info: {
#     "is_correct": True,
#     "true_category": "Billing",
#     "predicted_category": "Billing",
#     "episode_accuracy": 0.75,
#     "processing_time_ms": 45.2
#   }
```

### Phase 5: Episode Summary
```
When done=True, info includes:
{
  "episode_summary": {
    "total_reward": 75.3,
    "accuracy": 0.87,
    "average_reward": 0.873,
    "average_processing_time_ms": 50.1,
    "total_episode_time_seconds": 5.2
  }
}
```

---

## 🎲 Task Details

### Easy Task
```
Goal: Classify 100 documents into 5 categories
Categories: [General, Billing, Support, Technical, HR]
Challenge: Low - documents are pre-processed
Time Limit: None
Baseline Score: 0.78
Perfect Score: 1.0
```

**Example Document**:
```
Content: "I need to update my billing address."
True Label: Billing (category index 1)
Features: 100-dimensional TF-IDF vector
Reward if correct: +1.0
Reward if incorrect: -0.5
```

### Medium Task
```
Goal: Classify 500 documents into 10 categories
Categories: General, Billing, Billing-Dispute, Support, Technical,
           Technical-Bug, HR-Payroll, HR-Benefits, Legal, Executive
Challenge: Medium - more categories, time constraints
Time Limit: 2 seconds per decision
Baseline Score: 0.65
```

### Hard Task
```
Goal: Classify 1000 documents into 20 categories
Challenge: High - fine-grained, speed-accuracy tradeoff
Time Limit: 1 second per decision
Baseline Score: 0.52
Speed-Accuracy Tradeoff: Agents must balance speed vs accuracy
```

---

## 🧠 How Features Work

### Feature Extraction
```python
# Text → TF-IDF Vector (100 dimensions)
document = "I was overcharged on my last invoice"
features = tfidf_vectorizer.transform([document]).toarray()
# Output: [0.45, -0.23, 0.67, 0.12, ..., -0.34]  (100 values)
```

### Feature Interpretation
```
Positive values: Word appears frequently and is important
Negative values: Word is common (low importance - IDF weighted)
Magnitude: How important this word is to the document

Example:
  "invoice" → high positive (rare, important word)
  "the" → near zero (common, low importance)
  "billing" → very high positive (rare AND diagnostic)
```

---

## 💰 Reward Function

### Accuracy Component
```
If agent classifies CORRECTLY:
  reward += 1.0  ✓

If agent classifies INCORRECTLY:
  reward -= 0.5  ✗
```

### Speed Bonus Component
```
Easy Task:
  Process < 100ms → reward += 0.1

Medium Task:
  Process < 200ms → reward += 0.15
  Process < 500ms → reward += 0.05

Hard Task (1s limit):
  Process < 100ms → reward += 0.2
  Process < 300ms → reward += 0.1
```

### Total Reward Example
```
Correct + Fast (medium):
  reward = 1.0 (correct) + 0.15 (speed) = 1.15

Incorrect + Fast (medium):
  reward = -0.5 (incorrect) + 0.15 (speed) = -0.35

Correct + Slow (hard):
  reward = 1.0 (correct) + 0.0 (too slow) = 1.0
```

---

## 📊 Episode Flow

```
Episode Start (100 documents for Easy)
│
├─ Doc 1: "My billing address..."
│  ├─ Agent predicts: Billing ✓
│  ├─ Reward: +1.0 + 0.1 (speed) = +1.1
│  └─ Accuracy: 1/1 = 100%
│
├─ Doc 2: "I found a bug..."
│  ├─ Agent predicts: Support ✗ (should be Technical)
│  ├─ Reward: -0.5 + 0.1 (speed) = -0.4
│  └─ Accuracy: 1/2 = 50%
│
├─ Doc 3: "Can I file a complaint?"
│  ├─ Agent predicts: HR ✓
│  ├─ Reward: +1.0 + 0.1 (speed) = +1.1
│  └─ Accuracy: 2/3 = 67%
│
...
│
└─ Doc 100: (last document)
   ├─ Agent predicts: General ✓
   ├─ Reward: +1.0
   └─ EPISODE COMPLETE
      Total Reward: 87.3
      Final Accuracy: 87%
      Avg Time: 50ms
```

---

## 🤖 Agent Decision Process

### Simple Baseline Agent (Rule-Based)
```python
def baseline_agent(observation):
    content = observation['content'].lower()
    
    # Check for keywords
    if "billing" in content or "invoice" in content:
        return 1  # Billing category
    elif "bug" in content or "error" in content:
        return 3  # Technical category
    elif "payroll" in content or "salary" in content:
        return 4  # HR category
    else:
        return 0  # Default: General
```

### Advanced ML Agent (Trained)
```python
def ml_agent(observation):
    # Use pre-trained neural network
    features = torch.tensor(observation['features'])
    logits = model(features)
    action = torch.argmax(logits).item()
    return action
```

---

## 🏅 Grading System

### How Agents Are Scored

```
Grading Process:
1. Run agent on test set (100/500/1000 documents)
2. Collect predictions and ground truths
3. Calculate metrics:
   - Accuracy = correct / total
   - Precision = TP / (TP + FP) per category
   - Recall = TP / (TP + FN) per category
   - F1 Score = harmonic mean(precision, recall)
4. Scale to difficulty level
5. Return final score (0.0 - 1.0)
```

### Difficulty-Aware Scoring

```
EASY:
  score = accuracy
  (No speed penalty)

MEDIUM:
  score = 0.8 × accuracy + 0.2 × speed_bonus
  (Balanced approach)

HARD:
  score = 0.75 × accuracy + 0.25 × speed_bonus
  (Emphasizes accuracy, rewards speed)
```

### Example Grading

```
Agent Performance on Easy:
├─ Correct: 87/100 documents
├─ Accuracy: 0.87
├─ Avg Speed: 45ms
├─ Speed Bonus: 0.1 (< 100ms)
└─ Final Score: 0.87

Agent Performance on Hard:
├─ Correct: 730/1000 documents
├─ Accuracy: 0.73
├─ Avg Speed: 95ms
├─ Speed Bonus: 0.2 (< 100ms)
└─ Final Score: 0.75 × 0.73 + 0.25 × 0.2 = 0.595
```

---

## 🔄 State API

### Accessing Environment State
```python
state = env.state()

# Returns:
{
  "current_observation": {...},  # Current document
  "current_document_index": 47,  # Which document we're on
  "total_documents": 100,        # Total in episode
  "episode_reward_total": 42.7,  # Sum of all rewards
  "episode_accuracy": 0.894,     # Accuracy so far
  "average_processing_time_ms": 48.3,
  "task_difficulty": "easy"
}
```

---

## 📈 Training an Agent

### Learning Curve Example
```
Episode 1: Score 0.35 (Random decisions)
Episode 5: Score 0.52 (Learning keywords)
Episode 10: Score 0.68 (Pattern recognition)
Episode 20: Score 0.79 (Specialized learning)
Episode 50: Score 0.91 (Near optimal)
```

---

## 🔐 Reproducibility

### Seed Control
```python
# Same seed = identical data and scores
env1 = DocumentClassificationEnv("easy", seed=42)
env2 = DocumentClassificationEnv("easy", seed=42)

obs1, _ = env1.reset(seed=42)
obs2, _ = env2.reset(seed=42)

# obs1['features'] == obs2['features']  ✓
```

### Fixed Test Sets
```
Test Set is FIXED:
- Easy: Same 100 documents every evaluation
- Medium: Same 500 documents every evaluation
- Hard: Same 1000 documents every evaluation

This ensures:
✓ Agent scores are comparable across runs
✓ Different agents can be fairly compared
✓ Reproducible baseline metrics
```

---

## 🎯 Categories Explained

### Easy (5 categories)
```
0: General        → Default/uncategorized
1: Billing        → Payment, invoice, charges
2: Support        → Help, assistance, how-to
3: Technical      → Technical issues, bugs
4: HR             → HR, payroll, benefits
```

### Medium (10 categories)
```
0: General
1: Billing        → Payments
2: Billing-Dispute → Overcharge disputes
3: Support        → General support
4: Technical      → General technical
5: Technical-Bug  → Specific bugs/errors
6: HR-Payroll     → Salary/payroll
7: HR-Benefits    → Benefits/insurance
8: Legal          → Legal matters
9: Executive      → Management level
```

### Hard (20 categories)
```
0-3: Billing variants
4-5: Support urgent/normal
6-8: Technical variants
9-11: HR variants
12-14: Legal variants
15-16: Executive variants
17: Finance
18: Marketing
19: Operations
```

---

## 💡 Real-World Use Case

```
A company receives 1000 emails/documents per day:

┌─────────────────────┐
│ Email comes in:     │
│ "I was charged     │
│ twice for order    │
│ #5432"             │
└─────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ Environment processes:      │
│ - Extract features (TF-IDF) │
│ - Normalize                 │
│ - Present to agent          │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ Agent decides: Category 2   │
│ (Billing-Dispute)          │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ Route to:                   │
│ Billing Department          │
│ Dispute Resolution Team     │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ Get reward:                 │
│ +1.0 (correct routing)      │
│ Learn from this decision    │
└─────────────────────────────┘
```

---

## 🚀 Running the Environment

### Local Execution
```bash
pip install -r requirements.txt
python test_environment.py          # Verify everything works
python baseline_inference.py        # See baseline performance
python app.py                       # Interactive web interface
```

### Docker Execution
```bash
docker build -t doc-classifier .
docker run -p 7860:7860 doc-classifier
# Access at http://localhost:7860
```

---

## 📊 Performance Interpretation

### Baseline Scores
```
Easy: 0.78
├─ Simple keyword matching strategy
├─ Achievable by any RL agent with basic features
└─ Gap to perfect (1.0) = 0.22 (22%)

Medium: 0.65
├─ Extended keywords + category variants
├─ Requires learning category distinctions
└─ Gap to target (0.85) = 0.20 (20%)

Hard: 0.52
├─ Limited by heuristic approach
├─ Needs ML model for fine-grained classification
└─ Gap to target (0.75) = 0.23 (23%)
```

### What Scores Mean
```
0.95+: Expert-level performance
0.85+: Strong RL agent
0.75+: Well-trained model
0.65+: Basic learning
0.50+: Better than random
0.00 : Completely random guessing
```

---

## ✨ Key Insights

1. **Real-World Problem**: Document routing is a genuine business challenge
2. **Progressive Difficulty**: Tasks scale from simple to complex
3. **Speed-Accuracy Tradeoff**: Hard task forces intelligent decisions
4. **Reproducible**: Same seed = same results every time
5. **Deterministic Grading**: Fair comparison between agents
6. **Extensible**: Easy to add custom agents

---

## 🎓 Summary

The environment teaches agents to:
✅ Understand document content (NLP)
✅ Extract meaningful patterns (feature importance)
✅ Make fast decisions (speed optimization)
✅ Balance accuracy vs speed (tradeoff learning)
✅ Generalize to new documents (test set performance)

This mirrors real-world AI deployment challenges!

---

**Ready to try it?** Start with:
```bash
pip install -r requirements.txt
python test_environment.py
python app.py
```

Then explore the code and try building your own agent!
