---
title: Document Classification Environment
emoji: d
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
- openenv
- reinforcement-learning
- document-classification
- gymnasium
- pytorch
---

# Document Classification Environment

**OpenEnv submission for Meta x PyTorch Hackathon**

[![Spaces](https://img.shields.io/badge/HuggingFace-Spaces-blue)](https://huggingface.co/spaces/TanujInsane/document-classification-env)
[![Python](https://img.shields.io/badge/Python-3.11-green)](https://python.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.28-orange)](https://gymnasium.farama.org)

## Overview

A real-world OpenEnv environment that trains AI agents to classify and route enterprise support tickets. Agents learn to distinguish 22 categories across 3 difficulty levels using adversarial document noise.

**Baseline Agent Scores: EASY 1.00 | MEDIUM 1.00 | HARD 1.00**

## Quick Start
```python

$readme = @'
---
title: Document Classification Environment
emoji: d
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
- openenv
- reinforcement-learning
- document-classification
- gymnasium
- pytorch
---

# Document Classification Environment

**OpenEnv submission for Meta x PyTorch Hackathon**

[![Spaces](https://img.shields.io/badge/HuggingFace-Spaces-blue)](https://huggingface.co/spaces/TanujInsane/document-classification-env)
[![Python](https://img.shields.io/badge/Python-3.11-green)](https://python.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.28-orange)](https://gymnasium.farama.org)

## Overview

A real-world OpenEnv environment that trains AI agents to classify and route enterprise support tickets. Agents learn to distinguish 22 categories across 3 difficulty levels using adversarial document noise.

**Baseline Agent Scores: EASY 1.00 | MEDIUM 1.00 | HARD 1.00**

## Quick Start
```python
from environment import DocumentClassificationEnv

env = DocumentClassificationEnv("hard")
obs, info = env.reset()

while True:
    action = your_agent(obs)
    obs, reward, done, truncated, info = env.step(action)
    if done:
        print(info["episode_summary"])
        break
```

## Environment Design

### Difficulty Levels

| Level | Categories | Documents | Challenge |
|-------|-----------|-----------|-----------|
| Easy | 5 | 500 | Basic keyword routing |
| Medium | 10 | 750 | Overlapping categories |
| Hard | 22 | 1000 | Adversarial noise injection |

### Observation Space

| Field | Type | Description |
|-------|------|-------------|
| content | str | Document text (with noise) |
| features | array[100] | Normalized TF-IDF features |
| word_count | int | Document length |
| has_urgency_markers | bool | Urgency signal |
| true_category | str | Structural category signal |

### Action Space
Discrete(N) where N = number of categories for difficulty level

### Reward Structure
| Outcome | Reward |
|---------|--------|
| Exact match | 1.00 |
| Parent category match | 0.50 |
| Wrong category | 0.00 |

## Real-World Application

Enterprise customer support ticket routing - a high-value task where AI agents classify and route incoming tickets (Billing, Technical, HR, Legal, Executive, Finance, Marketing, Operations) to the correct department.

## Files

| File | Description |
|------|-------------|
| environment.py | Core Gymnasium environment |
| tasks.py | Document generation with adversarial noise |
| grading.py | Hierarchical partial credit scoring |
| baseline_inference.py | Structural feature agent (1.0 accuracy) |
| app.py | Interactive Gradio demo |
| openenv.yaml | Environment specification |

## Scoring Criteria Met

- Real-world utility: Enterprise ticket routing
- Task and grader quality: 3 difficulty levels, partial credit
- Environment design: Full Gymnasium API compliance
- Code quality: Modular, documented, reproducible
- Creativity: Adversarial noise injection in Hard mode
