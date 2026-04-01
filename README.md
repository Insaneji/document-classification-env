---
title: Document Classification Env
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
- openenv
- document-classification
- reinforcement-learning
- gymnasium
---

# 📄 Document Classification Environment

**OpenEnv submission for Meta x PyTorch Hackathon**

An interactive Gymnasium-compatible environment where AI agents learn to classify and route customer support tickets to the correct department.

---

## 🎯 Real-World Task

Customer support teams receive hundreds of tickets daily — billing issues, technical bugs, HR complaints, legal queries. This environment simulates that routing challenge, training agents to read a document and instantly decide which department should handle it.

**Why this matters:** Misrouted tickets waste time and frustrate customers. A well-trained agent can reduce misrouting by 80%+.

---

## 🏗️ Environment Design

```
DocumentClassificationEnv(task_difficulty="hard", seed=42)
├── observation_space: Dict
│   ├── content: Text (the document)
│   ├── document_id: Text
│   ├── word_count: Box(1,)
│   ├── has_urgency_markers: MultiBinary(1)
│   ├── features: Box(100,)  ← TF-IDF features
│   ├── document_index: Box(1,)
│   └── total_documents: Box(1,)
└── action_space: Discrete(N)  ← N = num categories
```

### API

```python
from environment import DocumentClassificationEnv

env = DocumentClassificationEnv(task_difficulty="hard", seed=42)
obs, info = env.reset()

while True:
    action = your_agent(obs)          # int: category index
    obs, reward, done, _, info = env.step(action)
    print(f"Reward: {reward:.3f}, Correct: {info['is_correct']}")
    if done:
        print(info["episode_summary"])
        break
```

---

## 📊 Three Progressive Tasks

| Task | Categories | Documents | Time Limit | Target Score |
|------|-----------|-----------|------------|--------------|
| **Easy** | 5 | 100 | None | 0.85 |
| **Medium** | 10 | 500 | 2 sec | 0.75 |
| **Hard** | 22 | 1000 | 1 sec | 0.75 |

### Categories (Hard Mode — 22 total)
`General` `Billing` `Billing-Dispute` `Billing-Refund` `Support` `Support-Urgent` `Support-Normal` `Technical` `Technical-Bug` `Technical-Feature` `HR` `HR-Payroll` `HR-Benefits` `HR-Complaint` `Legal` `Legal-Contract` `Legal-Compliance` `Executive` `Executive-Strategic` `Finance` `Marketing` `Operations`

---

## 🏆 Reward Function

```python
reward = accuracy_reward + speed_bonus

# accuracy_reward:
#   +1.0 for correct classification
#   -0.5 for wrong classification

# speed_bonus (difficulty-dependent):
#   Easy:   +0.10 if < 100ms
#   Medium: +0.15 if < 200ms, +0.10 if < 500ms
#   Hard:   +0.20 if < 100ms, +0.10 if < 300ms
```

Partial credit via speed bonus encourages efficient inference, not just accuracy.

---

## 📈 Baseline Results (Keyword Agent)

| Task | Score | Notes |
|------|-------|-------|
| EASY | **0.77** | Simple keyword matching |
| MEDIUM | **0.89** | Priority-ordered keywords |
| HARD | **0.165** | 22 categories, harder to distinguish |

Better agents (TF-IDF similarity, fine-tuned LLM) can significantly beat baseline.

---

## 🚀 Quick Start

```bash
git clone https://huggingface.co/spaces/TanujInsane/document-classification-env
cd document-classification-env
pip install -r requirements.txt

# Run baseline
python baseline_inference.py --task all --output results.json

# Launch UI
python app.py
```

---

## 📁 File Structure

```
├── environment.py          # Main Gymnasium environment
├── tasks.py               # Document generation + TF-IDF features
├── grading.py             # Scoring logic (0.0 - 1.0)
├── baseline_inference.py  # Keyword-based baseline agent
├── app.py                 # Gradio interactive demo
├── test_environment.py    # 5 unit tests (all passing ✅)
├── openenv.yaml           # OpenEnv specification
├── Dockerfile             # Container deployment
└── requirements.txt       # Dependencies
```

---

## 🔬 Reproducibility

- Seed-controlled document generation
- Fixed test sets for fair grading
- Deterministic reward calculation
- Docker containerization for consistent deployment

---

## 💡 Improving Beyond Baseline

```python
# Example: TF-IDF similarity agent (beats keyword matching)
from sklearn.metrics.pairwise import cosine_similarity

class TFIDFAgent:
    def __init__(self, difficulty):
        self.env = DocumentClassificationEnv(difficulty)
        # Pre-compute category centroid vectors
        # Use cosine similarity at inference time
        ...
```

---

*Built for Meta x PyTorch OpenEnv Hackathon | Gymnasium-compatible | Docker deployed*
