# DEPLOYMENT & INTEGRATION GUIDE

## Quick Deployment

### Local Development

```bash
# 1. Clone/Navigate to project
cd c:\Users\91748\Desktop\metax

# 2. Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run tests
python test_environment.py

# 5. Run baseline evaluation
python baseline_inference.py --task all

# 6. Start web interface
python app.py
# Visit http://localhost:7860
```

### Docker Deployment

```bash
# Build image
docker build -t doc-classifier:latest .

# Run container
docker run -p 7860:7860 doc-classifier:latest

# Run with specific task
docker run -p 7860:7860 doc-classifier:latest \
  python baseline_inference.py --task hard --output results.json
```

### Hugging Face Spaces Deployment

1. Create a new Space on Hugging Face with Docker SDK
2. Clone the Space repository
3. Copy all files from this project
4. Push to the repository:
   ```bash
   git add .
   git commit -m "Initial OpenEnv environment"
   git push
   ```
5. Spaces will automatically build and deploy
6. Access at: https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME

## Project Structure Reference

```
Project Root/
├── Core Environment
│   ├── environment.py           - Main OpenEnv implementation
│   ├── tasks.py                 - Data generation
│   └── grading.py               - Evaluation system
│
├── Evaluation & Testing
│   ├── baseline_inference.py    - Baseline agent evaluation
│   ├── test_environment.py      - Unit tests
│   └── example_usage.py         - Usage examples
│
├── Web Interface
│   └── app.py                   - Gradio interface
│
├── Configuration
│   ├── requirements.txt         - Python dependencies
│   ├── setup.py                 - Package setup
│   ├── openenv.yaml             - OpenEnv specification
│   ├── Dockerfile               - Container definition
│   ├── app_config.yaml          - HF Spaces config
│   └── .gitignore              - Git exclusions
│
└── Documentation
    ├── README.md                - Main documentation
    ├── BUILD_SUMMARY.md         - Build summary
    └── DEPLOYMENT_GUIDE.md      - This file
```

## API Reference

### Environment Initialization

```python
from environment import DocumentClassificationEnv

# Create easy environment
env = DocumentClassificationEnv(task_difficulty="easy")

# Create with seed for reproducibility
env = DocumentClassificationEnv(task_difficulty="hard", seed=42)
```

### Core Methods

```python
# Reset environment
observation, info = env.reset()
observation, info = env.reset(seed=42)

# Take action
obs, reward, done, truncated, info = env.step(action)

# Get state
state = env.state()

# Close environment
env.close()
```

### Accessing Environment Properties

```python
# Action space
env.action_space  # Discrete(5/10/20)

# Observation space
env.observation_space  # Dict with typed fields

# Category mapping
env.CATEGORY_MAPS[difficulty]  # Dict of category names

# Task configuration
env.task_config  # Dict with task parameters
```

## Using the Baseline Agent

```python
from grading import BaselineAgent

agent = BaselineAgent("easy")

# Get action for observation
action = agent.decide(observation)
```

## Evaluating an Agent

```python
from grading import AgentGrader

grader = AgentGrader("easy")

# Define your agent policy
def my_agent(observation):
    # Your agent logic here
    return action

# Grade the agent
score, metrics = grader.grade_agent(my_agent, verbose=True)
```

## Running Baseline Evaluation

```bash
# Evaluate specific task
python baseline_inference.py --task easy

# Evaluate all tasks
python baseline_inference.py --task all

# Save results to file
python baseline_inference.py --task all --output my_results.json

# Verbose output
python baseline_inference.py --task hard --verbose
```

## Understanding the Evaluation Metrics

### Easy Task
- **Accuracy**: Percentage of correct classifications (0.0-1.0)
- **Reward**: Accuracy + speed bonus - error penalties
- **Score**: Direct accuracy value

### Medium Task
- **Accuracy**: Percentage of correct classifications
- **Speed Bonus**: Reward for fast responses (2s limit)
- **Score**: 0.8 × Accuracy + 0.2 × SpeedBonus

### Hard Task
- **Accuracy**: Percentage of correct classifications
- **Speed Bonus**: Reward for fast responses (1s limit)
- **Score**: 0.75 × Accuracy + 0.25 × SpeedBonus

## Expected Baseline Performance

```
Easy:    0.78  (Target: 0.95+) - Gap: 17%
Medium:  0.65  (Target: 0.85+) - Gap: 20%
Hard:    0.52  (Target: 0.75+) - Gap: 23%
```

## Troubleshooting

### Import Errors

```
ModuleNotFoundError: No module named 'gymnasium'
```
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Environment Creation Fails

```python
# Ensure proper initialization
env = DocumentClassificationEnv(task_difficulty="easy")
assert env is not None
```

### Baseline Agent Not Working

```python
# Check agent output
action = agent.decide(obs)
assert isinstance(action, int)
assert 0 <= action < env.num_categories
```

### Docker Build Issues

```bash
# Clean build (remove cache)
docker build --no-cache -t doc-classifier:latest .

# Check build logs
docker build -t doc-classifier:latest . 2>&1 | tail -50
```

## Performance Optimization

### For Fast Inference
```python
# Use easy task for quick testing
env = DocumentClassificationEnv(task_difficulty="easy")  # Fastest

# Batch operations
for _ in range(100):
    action = agent.decide(obs)
    obs, reward, done, _, _ = env.step(action)
```

### For Accurate Evaluation
```python
# Use hard task for realistic evaluation
env = DocumentClassificationEnv(task_difficulty="hard")

# Run full episodes
while not done:
    action = agent.decide(obs)
    obs, reward, done, _, _ = env.step(action)
```

## Integration with Custom Agents

### Simple Agent Template

```python
def my_agent(observation):
    """
    Custom agent implementation
    
    Args:
        observation: Dict with keys:
            - 'document_id': str
            - 'content': str
            - 'word_count': np.array
            - 'has_urgency_markers': np.array
            - 'features': np.array (100-dim)
    
    Returns:
        action: int in range [0, num_categories)
    """
    # Extract features from observation
    features = observation['features']
    content = observation['content']
    
    # Your classification logic
    action = classify(features, content)
    
    return action

# Evaluate your agent
grader = AgentGrader("easy")
score, metrics = grader.grade_agent(my_agent)
```

### Deep Learning Agent Example

```python
import torch
from environment import DocumentClassificationEnv

class NeuralAgent:
    def __init__(self, model):
        self.model = model
    
    def decide(self, observation):
        features = torch.FloatTensor(observation['features']).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(features)
            action = torch.argmax(logits, dim=1).item()
        return action

# Use with grader
agent = NeuralAgent(my_model)
grader = AgentGrader("hard")
score, metrics = grader.grade_agent(agent.decide)
```

## Testing Your Setup

```bash
# Quick verification (5 min)
python test_environment.py

# Full baseline evaluation (15 min)
python baseline_inference.py --task all

# Run examples (10 min)
python example_usage.py

# Total time: ~30 minutes for full test
```

## Advanced Features

### Reproducible Results

```python
# Same seed = identical data and evaluation
env1 = DocumentClassificationEnv("easy", seed=42)
env2 = DocumentClassificationEnv("easy", seed=42)

obs1, _ = env1.reset(seed=42)
obs2, _ = env2.reset(seed=42)

assert np.array_equal(obs1['features'], obs2['features'])
```

### Accessing Episode Statistics

```python
while not done:
    obs, reward, done, _, info = env.step(action)
    
    # Current episode stats
    print(info['episode_accuracy'])
    print(info['is_correct'])
    print(info['processing_time_ms'])
    
    # Final summary (available when done=True)
    if done:
        summary = info['episode_summary']
        print(summary['accuracy'])
        print(summary['total_reward'])
```

### Custom Reward Analysis

```python
# Track rewards over episode
rewards = []
while not done:
    obs, reward, done, _, _ = env.step(action)
    rewards.append(reward)

print(f"Mean reward: {np.mean(rewards):.3f}")
print(f"Std dev: {np.std(rewards):.3f}")
print(f"Max: {np.max(rewards):.3f}")
print(f"Min: {np.min(rewards):.3f}")
```

## Support & Resources

- **Documentation**: See README.md
- **Examples**: See example_usage.py
- **Tests**: Run test_environment.py
- **OpenEnv Spec**: See openenv.yaml

## Deployment Checklist

- [x] Environment passes all tests
- [x] Baseline evaluation runs successfully
- [x] Docker image builds without errors
- [x] Web interface loads in browser
- [x] Documentation is complete
- [x] Examples run successfully
- [x] Reproducibility verified
- [x] Ready for production

---

**Status**: ✅ Ready for Deployment

For issues or questions, refer to README.md or BUILD_SUMMARY.md
