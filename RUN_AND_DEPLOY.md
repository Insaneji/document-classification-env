# 🚀 DEPLOYMENT & TESTING GUIDE

## Installation & Setup

### Step 1: Install Dependencies
```bash
cd c:\Users\91748\Desktop\metax

# For Windows:
python -m pip install --upgrade pip
pip install -r requirements.txt

# This installs:
# - gymnasium (OpenAI Gym modern replacement)
# - numpy, pandas, scikit-learn (data processing)
# - pyyaml (configuration)
# - flask (web framework)
# - gradio (web interface)
# - huggingface-hub (integration)
```

### Step 2: Verify Installation
```bash
python -c "import gymnasium; print('✓ Gymnasium installed')"
python test_environment.py
```

---

## Running the Environment

### Option A: Quick Test (2 minutes)
```bash
python test_environment.py
```

**Expected Output**:
```
============================================================
Document Classification Environment - Test Suite
============================================================

Testing environment creation...
✓ easy environment created successfully
✓ medium environment created successfully
✓ hard environment created successfully

Testing step function...
✓ Step 1: reward=0.850, accuracy=1.000
✓ Step 2: reward=-0.400, accuracy=0.500
✓ Step 3: reward=1.100, accuracy=0.667
✓ Step 4: reward=0.750, accuracy=0.750
✓ Step 5: reward=1.050, accuracy=0.800
✓ Step function working correctly

[... more tests ...]

TEST SUMMARY
============================================================
✓ PASS - Environment Creation
✓ PASS - Step Function
✓ PASS - State Function
✓ PASS - Baseline Agent
✓ PASS - Grading System

Total: 5/5 tests passed
============================================================
```

### Option B: Baseline Evaluation (5 minutes)
```bash
# Test specific task
python baseline_inference.py --task easy

# Test all tasks
python baseline_inference.py --task all

# Test with verbose output
python baseline_inference.py --task hard --verbose
```

**Expected Output**:
```
======================================================================
Document Classification Environment - Baseline Evaluation
======================================================================

[easy] Starting evaluation...

============================================================
Task: EASY
============================================================
Accuracy: 0.7800
Correct Classifications: 78/100
Average Reward: 0.8350
Total Reward: 83.5000
Average Processing Time: 48.23ms

Final Score: 0.7800
============================================================

✓ EASY - Score: 0.7800

[medium] Starting evaluation...
✓ MEDIUM - Score: 0.6500

[hard] Starting evaluation...
✓ HARD - Score: 0.5200

======================================================================
BASELINE PERFORMANCE SUMMARY
======================================================================
EASY     - Overall Score: 0.7800  |  Accuracy: 0.7800
MEDIUM   - Overall Score: 0.6500  |  Accuracy: 0.6800
HARD     - Overall Score: 0.5200  |  Accuracy: 0.5500
======================================================================

Results saved to: baseline_results.json
```

### Option C: Interactive Demo (5 minutes)
```bash
python app.py
```

**What happens**:
1. Gradio web server starts on `http://localhost:7860`
2. Browser opens automatically (or visit manually)
3. Four tabs available:
   - **Interactive Demo**: Try classifying documents in real-time
   - **Environment Info**: Learn about the task
   - **Baseline Evaluation**: See baseline scores
   - **OpenEnv Spec**: View the specification

**Demo Steps**:
- Select difficulty (easy/medium/hard)
- Click "Create Environment"
- Click "Reset Episode"
- Choose a category
- Click "Classify Document"
- See the result (correct/incorrect)
- Try more documents

### Option D: Run Examples (10 minutes)
```bash
python example_usage.py
```

**What runs**:
1. Basic environment usage
2. Environment state inspection
3. Baseline agent performance
4. Agent grading system
5. Difficulty comparison
6. Reproducibility with seeds
7. Episode summaries

---

## How It Works - Simple Explanation

### The Loop
```
1. Create Environment
   ↓
2. Reset Episode
   ├─ Generates 100/500/1000 documents
   ├─ Each document has text + features
   └─ Tracks progress
   ↓
3. Agent Receives Document
   ├─ Sees document content
   ├─ Sees 100-dimensional feature vector
   └─ Must decide: which category?
   ↓
4. Environment Rewards Agent
   ├─ +1.0 if correct classification
   ├─ -0.5 if incorrect
   ├─ +0.1 to +0.2 bonus if fast
   └─ Returns next document
   ↓
5. Repeat Until Done
   └─ Episode ends when all documents classified
   ↓
6. Get Final Score
   ├─ Accuracy
   ├─ Total Reward
   ├─ Average Processing Time
   └─ Difficulty-weighted Score (0.0-1.0)
```

---

## Understanding the Output

### Key Metrics

**Accuracy**
- What % of documents were classified correctly?
- Easy: 78% (baseline)
- Medium: 68% (harder)
- Hard: 55% (hardest)

**Reward**
- +1.0: Correct classification
- -0.5: Wrong classification
- +0.1 to +0.2: Speed bonus

**Processing Time**
- How fast did the agent decide?
- Easy: No time limit (average 50ms)
- Medium: 2 seconds per decision (average 150ms)
- Hard: 1 second per decision (average 100ms)

**Score**
- 0.0-1.0 final rating
- Easy: Accuracy alone
- Medium: 80% accuracy + 20% speed
- Hard: 75% accuracy + 25% speed

---

## Example: Running Your First Classification

### Step-by-Step

**1. Create Environment**
```python
from environment import DocumentClassificationEnv
env = DocumentClassificationEnv("easy")
```

**2. Reset Episode**
```python
obs, info = env.reset()
print(obs['content'])
# Output: "My invoice shows an incorrect amount. Please review."
print(f"Words: {obs['word_count'][0]}")
# Output: Words: 9
```

**3. Make Decision**
```python
# Easy categories: [General, Billing, Support, Technical, HR]
action = 1  # Choose "Billing"
```

**4. Step Environment**
```python
obs, reward, done, _, info = env.step(action)

print(f"Reward: {reward}")
# Output: Reward: 1.1 (correct + speed bonus)
print(f"Accuracy: {info['episode_accuracy']}")
# Output: Accuracy: 1.0 (1 correct out of 1)
```

**5. Repeat**
```python
while not done:
    action = agent.decide(obs)
    obs, reward, done, _, info = env.step(action)
    
print(info['episode_summary'])
# {
#   'accuracy': 0.87,
#   'total_reward': 87.3,
#   'average_reward': 0.873,
#   'total_documents_classified': 100
# }
```

---

## Docker Deployment

### Building Docker Image
```bash
# In project directory
docker build -t doc-classifier:latest .

# Monitor build
# Takes 2-3 minutes
# Downloads Python base image
# Installs dependencies
# Creates non-root user
# Sets up health check
```

### Running Container
```bash
# Run with port mapping
docker run -p 7860:7860 doc-classifier:latest

# Run with environment variable
docker run -e TASK=easy -p 7860:7860 doc-classifier:latest

# Run interactively
docker run -it -p 7860:7860 doc-classifier:latest /bin/bash
```

### Docker Output
```
* Running Gradio server
* Listening on http://0.0.0.0:7860
* Health check: PASS
```

---

## Cloud Deployment (Hugging Face Spaces)

### Step 1: Create Space
1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Name: `document-classifier-env`
4. License: MIT
5. Space SDK: Docker
6. Click "Create Space"

### Step 2: Upload Files
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/document-classifier-env
cd document-classifier-env

# Copy all files from project
cp c:\Users\91748\Desktop\metax\* .

# Commit and push
git add .
git commit -m "Initial OpenEnv environment"
git push
```

### Step 3: Monitor Deployment
- Space automatically builds Docker image
- Watch build logs in Spaces UI
- Takes 5-10 minutes first time
- Then available at: https://huggingface.co/spaces/YOUR_USERNAME/document-classifier-env

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'gymnasium'"
**Solution**:
```bash
pip install -r requirements.txt
# or
pip install gymnasium numpy pandas scikit-learn pyyaml requests flask gradio
```

### Issue: "Port 7860 already in use"
**Solution**:
```bash
# Option 1: Use different port
python -c "from app import create_interface; create_interface().launch(server_port=8080)"

# Option 2: Kill process using port 7860
# Windows: taskkill /IM python.exe /F
# Linux: lsof -ti:7860 | xargs kill -9
```

### Issue: "Docker build fails"
**Solution**:
```bash
# Clean build
docker build --no-cache -t doc-classifier:latest .

# Check Docker is running
docker ps

# Check Dockerfile syntax
docker build --progress=plain -t doc-classifier .
```

### Issue: Tests fail with "Feature extraction error"
**Solution**:
```bash
# Reinstall scikit-learn
pip install --upgrade scikit-learn
python test_environment.py
```

---

## Performance Tuning

### For Speed
```python
# Use Easy task
env = DocumentClassificationEnv("easy")  # 100 docs, no time limit

# Process in batches
batch_size = 10
for _ in range(batch_size):
    action = agent.decide(obs)
    obs, _, _, _, _ = env.step(action)
```

### For Accuracy
```python
# Use Hard task
env = DocumentClassificationEnv("hard")  # 1000 docs, tight deadline

# Give more time per decision
import time
start = time.time()
action = agent.decide(obs)  # Can take up to 1 second
elapsed = time.time() - start
```

### For Reproducibility
```python
# Use fixed seed
env = DocumentClassificationEnv("easy", seed=42)
obs, _ = env.reset(seed=42)

# Results will be identical across runs
```

---

## Monitoring & Logging

### Enable Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

env = DocumentClassificationEnv("easy")
obs, _ = env.reset()
```

### Save Results
```bash
# Baseline evaluation saves JSON
python baseline_inference.py --task all --output results.json

# View results
cat results.json
```

---

## File Monitoring

### Watch for Changes
```bash
# Windows: Use `watchdog` package
pip install watchdog
watchmedo shell-command \
    --patterns="*.py" \
    --recursive \
    --command='python test_environment.py' \
    .
```

---

## Getting Help

### Check Logs
```bash
# Python logs
python -u baseline_inference.py --task easy 2>&1 | tee run.log

# Docker logs
docker logs CONTAINER_ID
docker logs -f CONTAINER_ID  # Follow logs
```

### Verify Setup
```bash
# Run diagnostic
python -c """
import gymnasium
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from environment import DocumentClassificationEnv

print('✓ All imports successful')

env = DocumentClassificationEnv('easy')
obs, _ = env.reset()
print(f'✓ Environment initialized')
print(f'✓ Observation keys: {list(obs.keys())}')
print(f'✓ Features shape: {obs[\"features\"].shape}')
print('✓ Setup verified - ready to go!')
"""
```

---

## Quick Command Reference

```bash
# Setup
pip install -r requirements.txt

# Test
python test_environment.py

# Evaluate
python baseline_inference.py --task all

# Run examples
python example_usage.py

# Interactive demo
python app.py

# Docker
docker build -t doc-classifier .
docker run -p 7860:7860 doc-classifier

# Cleanup
rm -rf __pycache__
rm -rf *.egg-info
rm -rf .pytest_cache
```

---

## Summary

✅ **Installation**: `pip install -r requirements.txt`
✅ **Test**: `python test_environment.py`
✅ **Try it**: `python app.py`
✅ **Evaluate**: `python baseline_inference.py --task all`
✅ **Deploy**: `docker build . && docker run -p 7860:7860 doc-classifier`

**Your environment is ready to use!**
