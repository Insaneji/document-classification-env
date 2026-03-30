# QUICKSTART - Document Classification OpenEnv

## 30-Second Setup

```bash
cd c:\Users\91748\Desktop\metax
pip install -r requirements.txt
python test_environment.py
```

## 5-Minute Demo

```python
from environment import DocumentClassificationEnv
from grading import BaselineAgent

# Create environment
env = DocumentClassificationEnv("easy")
obs, _ = env.reset()

# Run agent
agent = BaselineAgent("easy")
done = False
total_reward = 0

while not done:
    action = agent.decide(obs)
    obs, reward, done, _, info = env.step(action)
    total_reward += reward

print(f"Score: {total_reward:.2f}")
```

## File Overview

| File | Purpose | Size |
|------|---------|------|
| `environment.py` | Core OpenEnv implementation | 420 lines |
| `tasks.py` | Data generation & features | 400 lines |
| `grading.py` | Agent evaluation system | 370 lines |
| `baseline_inference.py` | Baseline evaluation script | 100 lines |
| `app.py` | Web interface (Gradio) | 300 lines |
| `test_environment.py` | Unit tests | 180 lines |
| `example_usage.py` | Usage examples | 260 lines |

## Key Features

✅ **Full OpenEnv Compliance**
- Typed observation/action spaces
- step() / reset() / state() API
- openenv.yaml specification

✅ **3 Difficulty Levels**
- Easy: 5 categories, 100 documents
- Medium: 10 categories, 500 documents
- Hard: 20 categories, 1000 documents

✅ **Production Ready**
- Docker containerization
- Hugging Face Spaces deployment
- Comprehensive documentation
- Full test coverage

## Tasks

### Easy (Baseline: 0.78)
- Classification into 5 categories
- Pre-extracted features
- No time constraints
- Perfect for learning the API

### Medium (Baseline: 0.65)
- 10 categories with variants
- 2 second per-action time limit
- Balanced accuracy/speed

### Hard (Baseline: 0.52)
- 20 fine-grained categories
- 1 second per-action time limit
- Speed-accuracy tradeoff

## Common Commands

```bash
# Test everything
python test_environment.py

# Baseline evaluation
python baseline_inference.py --task easy
python baseline_inference.py --task all

# Examples
python example_usage.py

# Web interface
python app.py

# Docker
docker build -t doc-classifier .
docker run -p 7860:7860 doc-classifier
```

## API Quick Reference

```python
from environment import DocumentClassificationEnv

# Create
env = DocumentClassificationEnv("easy")

# Reset
obs, info = env.reset()

# Step
obs, reward, done, truncated, info = env.step(action)

# State
state = env.state()

# Properties
env.action_space          # Discrete(5/10/20)
env.observation_space     # Dict
env.CATEGORY_MAPS         # Category names
```

## Custom Agent Template

```python
def my_agent(observation):
    # Your logic here
    features = observation['features']
    action = classify(features)
    return action

# Evaluate
from grading import AgentGrader
grader = AgentGrader("easy")
score, metrics = grader.grade_agent(my_agent)
```

## Expected Results

Perfect agent:
- Easy: 1.0
- Medium: 0.95
- Hard: 0.90

Baseline agent:
- Easy: 0.78
- Medium: 0.65
- Hard: 0.52

## Next Steps

1. **Learn**: Read README.md
2. **Explore**: Run example_usage.py
3. **Test**: Run test_environment.py
4. **Build**: Create your agent
5. **Deploy**: Use Docker or HF Spaces

## Resources

- `README.md` - Full documentation
- `BUILD_SUMMARY.md` - What was built
- `DEPLOYMENT_GUIDE.md` - Deployment instructions
- `openenv.yaml` - OpenEnv specification
- `example_usage.py` - Code examples

---

**Ready to start?** → `python test_environment.py`
