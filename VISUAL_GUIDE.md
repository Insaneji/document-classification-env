# 📊 VISUAL GUIDE - HOW THE OPENENV WORKS

## 🎬 Complete Agent Interaction Flow

```
┌──────────────────────────────────────────────────────────────────┐
│         OPENENV DOCUMENT CLASSIFICATION ENVIRONMENT              │
└──────────────────────────────────────────────────────────────────┘

════════════════════ START EPISODE ════════════════════

    env.reset()
         │
         ▼
    ┌─────────────────────────────────────┐
    │  Generate Documents                  │
    │  ─────────────────────────────────    │
    │  Easy: 100 documents                 │
    │  Medium: 500 documents               │
    │  Hard: 1000 documents                │
    │                                      │
    │  Categories: [5/10/20]               │
    │  Features: TF-IDF (100-dim)          │
    └─────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────┐
    │  Return Initial Observation          │
    │  ─────────────────────────────────    │
    │  {                                   │
    │    "document_id": "doc_000000",      │
    │    "content": "My invoice...",       │
    │    "word_count": [47],               │
    │    "features": [0.12, -0.34, ...],   │
    │    "document_index": [0],            │
    │    "total_documents": [100]          │
    │  }                                   │
    └─────────────────────────────────────┘
         │
════════════════════ AGENT LOOP ════════════════════
         │
         ├─→ [ DOCUMENT 1 ]
         │      │
         │      ▼
         │   ┌────────────────────────────────┐
         │   │  Agent Analyzes Document       │
         │   │  "My invoice shows error"      │
         │   │  → Features: TF-IDF weights    │
         │   │  → Decision: Category?         │
         │   └────────────────────────────────┘
         │      │
         │      ▼
         │   ┌────────────────────────────────┐
         │   │  Agent Decision                │
         │   │  ──────────────────────────   │
         │   │  Possible Actions (Easy):      │
         │   │  • 0 = General                 │
         │   │  • 1 = Billing ◄─ PREDICTED   │
         │   │  • 2 = Support                 │
         │   │  • 3 = Technical               │
         │   │  • 4 = HR                      │
         │   │                                │
         │   │  Agent chooses: action = 1    │
         │   └────────────────────────────────┘
         │      │
         │      ▼
         │   env.step(action=1)
         │      │
         │      ▼
         │   ┌────────────────────────────────┐
         │   │  Evaluate Decision             │
         │   │  ──────────────────────────   │
         │   │  True Label: Billing (1) ✓     │
         │   │  Predicted: Billing (1) ✓      │
         │   │  Result: CORRECT!              │
         │   └────────────────────────────────┘
         │      │
         │      ▼
         │   ┌────────────────────────────────┐
         │   │  Calculate Reward              │
         │   │  ──────────────────────────   │
         │   │  Correct:  +1.0 ✓              │
         │   │  Speed:    +0.1 (< 100ms) ✓   │
         │   │  ─────────────────────────    │
         │   │  Total:    +1.1                │
         │   └────────────────────────────────┘
         │      │
         │      ▼
         │   ┌────────────────────────────────┐
         │   │  Update Episode Stats          │
         │   │  ──────────────────────────   │
         │   │  Correct: 1/1 = 100%           │
         │   │  Total Reward: 1.1             │
         │   │  Avg Reward: 1.1               │
         │   └────────────────────────────────┘
         │      │
         ├─→ [ DOCUMENT 2 ]
         │      │
         │      ▼
         │   ┌────────────────────────────────┐
         │   │  "I found a bug in system"     │
         │   │  True Label: Technical (3)     │
         │   │  Predicted: Support (2) ✗      │
         │   │  ──────────────────────────   │
         │   │  Correct: -0.5 ✗               │
         │   │  Speed: +0.1 ✓                 │
         │   │  Total: -0.4                   │
         │   │  ──────────────────────────   │
         │   │  Accuracy: 1/2 = 50%           │
         │   └────────────────────────────────┘
         │      │
         │      ▼
         │   ... REPEATS FOR ALL DOCUMENTS ...
         │      │
         ├─→ [ DOCUMENT N ]
         │      │
         │      ▼
         │   ┌────────────────────────────────┐
         │   │  Last Document                 │
         │   │  All predictions made          │
         │   │  Episode complete!             │
         │   └────────────────────────────────┘
         │
════════════════════ END EPISODE ════════════════════
         │
         ▼
    ┌─────────────────────────────────────┐
    │  Episode Summary                     │
    │  ─────────────────────────────────    │
    │  • Total Accuracy: 87/100 = 87%      │
    │  • Total Reward: 87.3                │
    │  • Avg Time/Doc: 50ms                │
    │  • Final Score: 0.87 (Easy)          │
    │  ─────────────────────────────────    │
    │  Returns:                            │
    │  - obs: empty (episode done)         │
    │  - reward: 0.0                       │
    │  - done: True                        │
    │  - info:                             │
    │    └─ episode_summary: {...}         │
    └─────────────────────────────────────┘
         │
         ▼
    Agent has learned from 100 documents!
    Ready for next episode or deployment.
```

---

## 📈 Task Difficulty Progression

```
EASY (BASELINE: 0.78)
┌────────────────────────────────┐
│ Categories: 5 (Simple)          │
│ ├─ General                       │
│ ├─ Billing                       │
│ ├─ Support                       │
│ ├─ Technical                     │
│ └─ HR                            │
│                                  │
│ Documents: 100 (Pre-processed)  │
│ Time Limit: None                │
│ Challenge Level: ⭐              │
│ Training Time: 1-2 seconds      │
└────────────────────────────────┘
         │
         │ Difficulty increases
         ▼
MEDIUM (BASELINE: 0.65)
┌────────────────────────────────┐
│ Categories: 10 (Detailed)       │
│ ├─ General                       │
│ ├─ Billing                       │
│ ├─ Billing-Dispute ◄ NEW       │
│ ├─ Support                       │
│ ├─ Technical                     │
│ ├─ Technical-Bug ◄ NEW          │
│ ├─ HR-Payroll                    │
│ ├─ HR-Benefits                   │
│ ├─ Legal                         │
│ └─ Executive                     │
│                                  │
│ Documents: 500 (Raw text)       │
│ Time Limit: 2 seconds/doc       │
│ Challenge Level: ⭐⭐             │
│ Training Time: 10-15 seconds    │
└────────────────────────────────┘
         │
         │ Difficulty increases
         ▼
HARD (BASELINE: 0.52)
┌────────────────────────────────┐
│ Categories: 20 (Fine-grained)   │
│ ├─ General                       │
│ ├─ Billing / Billing-Dispute    │
│ ├─ Billing-Refund ◄ NEW        │
│ ├─ Support-Urgent / Support-    │
│ │  Normal ◄ NEW (split)         │
│ ├─ Technical / Technical-Bug    │
│ ├─ Technical-Feature ◄ NEW      │
│ ├─ HR (3 variants)               │
│ ├─ Legal (3 variants)            │
│ ├─ Executive (2 variants) ◄NEW │
│ ├─ Finance ◄ NEW                │
│ ├─ Marketing ◄ NEW              │
│ └─ Operations ◄ NEW             │
│                                  │
│ Documents: 1000 (Complex)       │
│ Time Limit: 1 second/doc        │
│ Challenge Level: ⭐⭐⭐            │
│ Training Time: 30-60 seconds    │
│ Speed-Accuracy Tradeoff: YES    │
└────────────────────────────────┘
```

---

## 🏆 Scoring System Visualization

```
┌─────────────────────────────────────────────────────────┐
│           EASY TASK SCORING                            │
├─────────────────────────────────────────────────────────┤
│ Calculation: SCORE = ACCURACY                          │
│                                                        │
│ Example: 87 correct out of 100                         │
│ ────────────────────────────────────────────────       │
│ Score = 87 / 100 = 0.87                               │
│                                                        │
│ Interpretation:                                        │
│ 1.0 ████████████████████ Perfect                      │
│ 0.9 ██████████████████░░ Excellent                    │
│ 0.8 ████████████████░░░░ Good   ◄─ 0.87 (HERE)       │
│ 0.7 ██████████████░░░░░░ Fair                         │
│ 0.6 ████████████░░░░░░░░ Poor                         │
│ 0.5 ██████████░░░░░░░░░░ Baseline                     │
│ 0.0 ░░░░░░░░░░░░░░░░░░░░ Random                      │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│         MEDIUM TASK SCORING                            │
├─────────────────────────────────────────────────────────┤
│ Calculation: SCORE = 0.8 × ACCURACY + 0.2 × SPEED      │
│                                                        │
│ Example:                                               │
│   Accuracy = 68%                                      │
│   Processing Time = 140ms (within 2s limit) → 0.15   │
│ ────────────────────────────────────────────────────   │
│   Score = 0.8 × 0.68 + 0.2 × 0.15                    │
│   Score = 0.544 + 0.03 = 0.574                       │
│                                                        │
│ (Without speed bonus it would be 0.544)               │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│           HARD TASK SCORING                            │
├─────────────────────────────────────────────────────────┤
│ Calculation: SCORE = 0.75 × ACCURACY + 0.25 × SPEED    │
│                                                        │
│ Example:                                               │
│   Accuracy = 73% (1000 docs, this is hard!)           │
│   Processing Time = 95ms (< 100ms) → 0.20 bonus      │
│ ────────────────────────────────────────────────────   │
│   Score = 0.75 × 0.73 + 0.25 × 0.20                  │
│   Score = 0.5475 + 0.05 = 0.5975                     │
│                                                        │
│ Speed is MORE valued in hard task (tradeoff)          │
└─────────────────────────────────────────────────────────┘
```

---

## 💰 Reward Mechanics

```
CORRECT CLASSIFICATION
┌─────────────────────┐
│  Accuracy: +1.0     │
└─────────────────────┘
            +
┌─────────────────────┐     ┌──────────────────────────┐
│ Speed Bonus:        │────▶│ Easy: +0.1 (< 100ms)    │
│                     │     │ Medium: +0.15 (< 200ms) │
│                     │     │ Hard: +0.2 (< 100ms)    │
└─────────────────────┘     └──────────────────────────┘
            │
            ▼
        +1.1 to +1.2 Reward ✓

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INCORRECT CLASSIFICATION
┌─────────────────────┐
│  Accuracy: -0.5     │
└─────────────────────┘
            +
┌─────────────────────┐     ┌──────────────────────────┐
│ Speed Bonus:        │────▶│ Easy: +0.1 (< 100ms)    │
│                     │     │ Medium: +0.15 (< 200ms) │
│                     │     │ Hard: +0.2 (< 100ms)    │
└─────────────────────┘     └──────────────────────────┘
            │
            ▼
        -0.4 to -0.3 Reward ✗

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EPISODE REWARD ACCUMULATION (100 documents, Easy)
┌──────────────────────────────────────────────┐
│ Doc  1: +1.1 → Total: 1.1    Acc: 100%     │
│ Doc  2: -0.4 → Total: 0.7    Acc: 50%      │
│ Doc  3: +1.1 → Total: 1.8    Acc: 67%      │
│ Doc  4: +1.1 → Total: 2.9    Acc: 75%      │
│ ...                                          │
│ Doc 87: +1.1 → Total: 87.3   Acc: 87%      │ ◄─ Final
│ Doc 88: -0.4 → Total: 86.9   Acc: 86.36%   │
│ ...                                          │
│ Doc100: +1.1 → Total: 87.3   Acc: 87%      │
│                                              │
│ Average Reward = 87.3 / 100 = 0.873         │
│ Final Score = 0.87                          │
└──────────────────────────────────────────────┘
```

---

## 📊 Document Feature Representation

```
Document Text
  │
  ▼
"I was overcharged on my last invoice"
  │
  │ (TF-IDF Vectorizer)
  │ Extract important words
  ▼
[word_weights: 100 dimensional vector]
  │
  │ Common words (low importance):
  │ "I", "on", "my" → near 0
  │
  │ Important words (high importance):
  │ "overcharged" → 0.45 (rare, specific)
  │ "invoice" → 0.52 (rare, specific)
  │ "billing" → 0.38 (related, important)
  │
  ▼
[-0.02, 0.45, 0.52, -0.01, ..., 0.38] ◄─ Fed to agent
  (100 dimensions total)
  │
  ▼
Agent Neural Net:
  Input: [100 features]
    ▼
  Hidden Layer 1: Learns feature combinations
    ▼
  Hidden Layer 2: Learns category patterns
    ▼
  Output: [5/10/20 category logits]
    ▼
  Softmax: Probability distribution
    ▼
  ArgMax: Best category
    ▼
  Action: Category index (0-4, 0-9, or 0-19)
    │
    ▼
Decision: "This is BILLING (category 1)"
```

---

## 🎯 Agent Learning Process

```
Episode 1: Random Guessing
┌──────────────┐
│ Accuracy: 20%│  ◄─ 1 in 5 correct (random)
│ Score: 0.20  │
└──────────────┘
     │
     ▼
Agent learns: Document contains "invoice" → likely Billing

     │
     ▼
Episode 2-5: Learning Keywords
┌──────────────┐
│ Accuracy: 45%│  ◄─ Better than random
│ Score: 0.45  │
└──────────────┘
     │
     ▼
Agent learns: Category combinations, priority

     │
     ▼
Episode 6-20: Pattern Recognition
┌──────────────┐
│ Accuracy: 75%│  ◄─ Strong signal learning
│ Score: 0.75  │
└──────────────┘
     │
     ▼
Agent learns: Subtle distinctions, edge cases

     │
     ▼
Episode 21+: Expert Level
┌──────────────┐
│ Accuracy: 87%│  ◄─ Near human performance
│ Score: 0.87  │
└──────────────┘
     │
     ▼
Agent learns: Rare cases, complex routing

     │
     ▼
Final: Deployed
```

---

## 🔄 Full Example: One Complete Classification

```
STEP 1: Document Arrives
┌──────────────────────────────────┐
│ Email: "I was charged twice for  │
│ my order #5432. Please refund    │
│ one charge immediately!"         │
└──────────────────────────────────┘

STEP 2: Features Extracted
┌──────────────────────────────────┐
│ Tokenized: ["charged", "twice",  │
│ "order", "refund", "immediately"]│
│                                  │
│ TF-IDF Vector: 100-dimensional   │
│ [0.45, 0.38, 0.52, 0.41, ...]   │
└──────────────────────────────────┘

STEP 3: Agent Processes
┌──────────────────────────────────┐
│ Input: Feature vector            │
│ Process: Neural network inference│
│ Logits: [0.1, 0.9, 0.3, ...]    │
│ Probabilities:                   │
│   General: 5%                    │
│   Billing-Dispute: 85% ◄─ MAX   │
│   Support: 10%                   │
└──────────────────────────────────┘

STEP 4: Decision Made
┌──────────────────────────────────┐
│ Selected Category:               │
│ action = 2 (Billing-Dispute)     │
│                                  │
│ Reasoning:                       │
│ • "charged" + "twice" = dispute  │
│ • "refund" = compensation claim  │
│ • "immediately" = urgent tone    │
└──────────────────────────────────┘

STEP 5: Evaluation
┌──────────────────────────────────┐
│ Ground Truth: Billing-Dispute    │
│ Prediction: Billing-Dispute      │
│ Result: ✓ CORRECT                │
│                                  │
│ Reward:                          │
│ • Accuracy: +1.0                 │
│ • Speed (45ms): +0.1             │
│ • Total: +1.1                    │
└──────────────────────────────────┘

STEP 6: Routing
┌──────────────────────────────────┐
│ Route to Department:             │
│ → Billing Team                   │
│   → Dispute Resolution Sub-team  │
│   → Priority: High               │
│                                  │
│ Action Taken: Email forwarded    │
│ Learning: Agent improved score   │
└──────────────────────────────────┘
```

---

## 📈 Performance Benchmarks

```
Baseline Agent Performance Across Tasks:

Easy Task (5 categories)
┌─────────────────────────────────┐
│ Accuracy: 78% ███████░░          │
│ Reward: 0.83                    │
│ Time: 48ms                      │
│ Score: 0.78                     │
└─────────────────────────────────┘
  ├─ Why 78%? Keyword matching
  │  works well for simple cases
  └─ Gap to perfect: 22%

Medium Task (10 categories)
┌─────────────────────────────────┐
│ Accuracy: 68% ██████░░░░         │
│ Reward: 0.71                    │
│ Time: 150ms                     │
│ Score: 0.65                     │
└─────────────────────────────────┘
  ├─ Why 68%? More categories,
  │  harder to distinguish
  └─ Gap to target (0.85): 20%

Hard Task (20 categories)
┌─────────────────────────────────┐
│ Accuracy: 55% █████░░░░░░░░      │
│ Reward: 0.57                    │
│ Time: 100ms                     │
│ Score: 0.52                     │
└─────────────────────────────────┘
  ├─ Why 55%? Complex routing,
  │  tight time constraints
  └─ Gap to target (0.75): 23%

Expected ML Model Performance:
┌─────────────────────────────────┐
│ Easy: 0.95+ (Expert)             │
│ Medium: 0.85+ (Strong)           │
│ Hard: 0.75+ (Capable)            │
└─────────────────────────────────┘
```

---

## 🎓 Summary

```
┌─────────────────────────────────────────────────────┐
│  COMPLETE OPENENV DOCUMENT CLASSIFICATION SYSTEM    │
├─────────────────────────────────────────────────────┤
│                                                     │
│ INPUT: Documents + Content + Features              │
│   ▼                                                 │
│ PROCESS: TF-IDF → Agent Decision → Evaluation      │
│   ▼                                                 │
│ OUTPUT: Score (0.0-1.0) + Feedback                 │
│                                                     │
│ RESULT: Agents learn to route documents!           │
│                                                     │
├─────────────────────────────────────────────────────┤
│ Ready to deploy and test!                           │
└─────────────────────────────────────────────────────┘
```
