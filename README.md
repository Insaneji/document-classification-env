# Document Classification OpenEnv Environment

A complete, real-world OpenEnv environment for training AI agents on document classification and routing tasks. This environment implements the full OpenEnv specification with typed models, reproducible grading, and progressive difficulty levels.

## Overview

This environment simulates a document routing system where an AI agent must classify incoming documents and route them to appropriate departments. The task involves understanding document content, making fast and accurate classification decisions, and balancing speed vs. accuracy tradeoffs.

## Environment Specification

### Observation Space
The agent receives a dictionary observation containing:
- `document_id`: Unique identifier for the document
- `content`: Raw text content of the document
- `word_count`: Number of words in the document
- `has_urgency_markers`: Whether document contains urgency indicators
- `features`: Pre-extracted TF-IDF features (numpy array)

### Action Space
The agent must select one of the available document categories:
- **Easy (5 categories)**: General, Billing, Support, Technical, HR
- **Medium (10 categories)**: General, Billing, Billing-Dispute, Support, Technical, Technical-Bug, HR-Payroll, HR-Benefits, Legal, Executive
- **Hard (20 categories)**: Detailed classification including priority levels and sub-categories

### Reward Function
```
reward = accuracy_bonus + speed_bonus - error_penalty - processing_cost
```
- **Accuracy Bonus**: +1.0 for correct classification, 0 otherwise
- **Speed Bonus**: Up to +0.2 based on processing speed
- **Error Penalty**: -0.5 for incorrect classification
- **Processing Cost**: Small deduction for resource usage in hard mode

### Tasks

#### Task 1 - Easy
- **Description**: Classify pre-processed documents into 5 categories
- **Document Count**: 100 documents
- **Features**: Pre-extracted and normalized
- **Time Limit**: None
- **Target Score**: 0.95+

#### Task 2 - Medium  
- **Description**: Classify raw text documents into 10 categories with moderate complexity
- **Document Count**: 500 documents
- **Features**: Raw text requiring processing
- **Time Limit**: 2 seconds per decision
- **Target Score**: 0.85+

#### Task 3 - Hard
- **Description**: Classify complex documents into 20 fine-grained categories with mixed formats
- **Document Count**: 1000 documents
- **Features**: Complex, mixed format, requires real-time processing
- **Time Limit**: 1 second per decision
- **Time-Cost Tradeoff**: Faster processing reduces accuracy potential
- **Target Score**: 0.75+

## Quick Start

### Installation

```bash
# Clone the repository
cd openenv-doc-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from environment import DocumentClassificationEnv

# Initialize environment
env = DocumentClassificationEnv(task_difficulty="easy")

# Reset for a new episode
observation = env.reset()

# Run interaction loop
done = False
total_reward = 0

while not done:
    # Your agent decision logic here
    action = agent.decide(observation)
    
    # Step environment
    observation, reward, done, info = env.step(action)
    total_reward += reward

print(f"Episode reward: {total_reward}")
```

### Running the Baseline

```bash
# Evaluate baseline performance
python baseline_inference.py --task easy
python baseline_inference.py --task medium  
python baseline_inference.py --task hard

# Expected baseline scores:
# Easy: 0.78
# Medium: 0.65
# Hard: 0.52
```

## Environment Specification (openenv.yaml)

The environment is fully specified in `openenv.yaml` including:
- Environment metadata (name, version, description)
- Action/observation spaces with types
- Difficulty levels and grading criteria
- Reward function specification

## Agent Grading

Each task includes built-in agent graders:
- **Easy Grader**: Evaluates 100 test documents, requires 0.0-1.0 score
- **Medium Grader**: Evaluates 500 test documents with time constraints
- **Hard Grader**: Evaluates 1000 documents with complex routing rules

Grading produces reproducible scores from 0.0 to 1.0.

## Deployment

### Docker

```bash
# Build Docker image
docker build -t doc-classifier-env:latest .

# Run container
docker run -p 8000:8000 doc-classifier-env:latest
```

### Hugging Face Spaces

The environment is deployed to Hugging Face Spaces:
- Access: [Link to Space]
- Direct API integration for evaluation

## Project Structure

```
├── README.md
├── requirements.txt
├── openenv.yaml
├── environment.py              # Core environment implementation
├── tasks.py                    # Task definitions and data generation
├── grading.py                  # Agent grading system
├── reward_function.py          # Reward calculation logic
├── baseline_inference.py       # Baseline agent implementation
├── Dockerfile
└── data/
    ├── documents_easy.json
    ├── documents_medium.json
    └── documents_hard.json
```

## Technical Details

### Implementation Stack
- **Framework**: Gymnasium (OpenAI Gym compatible)
- **Language**: Python 3.8+
- **Data Processing**: NumPy, Pandas, scikit-learn
- **Deployment**: Docker, Hugging Face Spaces

### OpenEnv Compliance
- ✓ Typed observation and action spaces
- ✓ step() / reset() / state() API
- ✓ openenv.yaml specification
- ✓ Deterministic grading
- ✓ Reproducible seeds

## Evaluation Criteria (Weights)

- **Real-world Utility (30%)**: Effectively simulates real document routing
- **Task-Grader Quality (25%)**: Well-defined graders with proper difficulty scaling
- **Environment Design (20%)**: Clear action/observation spaces, good documentation
- **Reproducibility & Compatibility (15%)**: Deterministic results, proper containerization
- **Creativity & Novelty (10%)**: Interesting design choices and features

## License

MIT

## Contributing

Contributions welcome. Please ensure all changes maintain OpenEnv spec compliance.

---

Built with ❤️ for AI agent training and evaluation.
