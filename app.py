"""
Hugging Face Spaces Integration for Document Classification Environment
Provides a web interface for interacting with the environment
"""

import gradio as gr
import json
import numpy as np
from environment import DocumentClassificationEnv
from grading import BaselineAgent, AgentGrader


# Global environment instance
current_env = None
current_agent = None
current_difficulty = "easy"


def create_environment(difficulty):
    """Create a new environment instance"""
    global current_env
    current_env = DocumentClassificationEnv(task_difficulty=difficulty)
    current_env.reset(seed=42)
    return f"Environment created: {difficulty}"


def reset_environment():
    """Reset the current environment"""
    global current_env
    if current_env is None:
        return "Environment not initialized", "", "", ""
    
    obs, _ = current_env.reset()
    
    doc_id = obs["document_id"]
    content = obs["content"]
    word_count = int(obs["word_count"][0])
    
    return f"Episode reset. Document: {doc_id}", content, f"Words: {word_count}", ""


def take_action(category_name):
    """Take an action in the environment"""
    global current_env
    
    if current_env is None:
        return "Error: Environment not initialized", "", "", "", ""
    
    # Map category name to index
    categories = list(current_env.CATEGORY_MAPS[current_env.task_difficulty].values())
    if category_name not in categories:
        return "Error: Invalid category", "", "", "", ""
    
    action = categories.index(category_name)
    
    # Step environment
    obs, reward, done, truncated, info = current_env.step(action)
    
    # Prepare response
    doc_id = obs["document_id"]
    content = obs["content"]
    word_count = int(obs["word_count"][0])
    
    status = f"Action: {category_name} | Reward: {reward:.3f}"
    if not done:
        status += f"\nNext Document: {doc_id}"
    else:
        summary = info.get("episode_summary", {})
        status += f"\n\nEpisode Complete!\nAccuracy: {summary.get('accuracy', 0):.4f}\nTotal Reward: {summary.get('total_reward', 0):.3f}"
    
    is_correct = info.get("is_correct", False)
    result_status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
    
    return status, content if not done else "Episode complete", f"Words: {word_count}" if not done else "", result_status, ""


def run_baseline_evaluation(difficulty):
    """Run baseline evaluation"""
    grader = AgentGrader(difficulty)
    agent = BaselineAgent(difficulty)
    
    score, metrics = grader.grade_agent(agent.decide, verbose=False)
    
    result_text = f"""
**Baseline Evaluation Results - {difficulty.upper()}**

**Overall Score:** {score:.4f}

**Metrics:**
- Accuracy: {metrics['accuracy']:.4f}
- Total Documents: {metrics['total_documents']}
- Correct Classifications: {metrics['correct_classifications']}
- Average Reward: {metrics['average_reward']:.4f}
- Average Processing Time: {metrics['average_processing_time_ms']:.2f}ms
- Total Time: {metrics['total_time_seconds']:.2f}s

**Difficulty Level Analysis:**
- Easy (Expected): 0.78
- Medium (Expected): 0.65
- Hard (Expected): 0.52
    """
    
    return result_text


# Create Gradio interface
def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="Document Classification OpenEnv") as demo:
        gr.Markdown("# Document Classification Environment")
        gr.Markdown("""
This is a real-world OpenEnv environment where AI agents learn to classify and route documents.
The environment implements the full OpenEnv specification with typed models and progressive difficulty levels.
        """)
        
        with gr.Tabs():
            # Tab 1: Interactive Demo
            with gr.Tab("Interactive Demo"):
                difficulty_selector = gr.Radio(
                    choices=["easy", "medium", "hard"],
                    value="easy",
                    label="Select Difficulty"
                )
                init_btn = gr.Button("Create Environment")
                
                with gr.Row():
                    reset_btn = gr.Button("Reset Episode")
                    status_output = gr.Textbox(label="Status", interactive=False)
                
                with gr.Row():
                    document_display = gr.Textbox(
                        label="Document Content",
                        interactive=False,
                        max_lines=10
                    )
                    metrics_display = gr.Textbox(label="Metrics", interactive=False)
                
                # Category selector depends on difficulty
                def update_categories(difficulty):
                    categories = list(DocumentClassificationEnv(difficulty).CATEGORY_MAPS[difficulty].values())
                    return gr.Dropdown(choices=categories, label="Select Category")
                
                category_selector = gr.Dropdown(
                    choices=["General", "Billing", "Support", "Technical", "HR"],
                    label="Select Category"
                )
                
                action_btn = gr.Button("Classify Document")
                result_display = gr.Textbox(label="Result", interactive=False)
                feedback_display = gr.Textbox(label="Feedback", interactive=False)
                
                # Event handlers
                init_btn.click(
                    create_environment,
                    inputs=[difficulty_selector],
                    outputs=[status_output]
                )
                
                reset_btn.click(
                    reset_environment,
                    outputs=[status_output, document_display, metrics_display, feedback_display]
                )
                
                action_btn.click(
                    take_action,
                    inputs=[category_selector],
                    outputs=[status_output, document_display, metrics_display, result_display, feedback_display]
                )
            
            # Tab 2: Environment Info
            with gr.Tab("Environment Info"):
                gr.Markdown("""
## Overview
This environment simulates a document routing system where an AI agent must classify 
incoming documents and route them to appropriate departments.

## Tasks
- **Easy**: 5 categories, 100 documents, no time constraint
- **Medium**: 10 categories, 500 documents, 2 seconds per decision
- **Hard**: 20 categories, 1000 documents, 1 second per decision

## Action Space
Select one of the available categories for each document.

## Observation Space
- `document_id`: Unique identifier
- `content`: Raw text content
- `word_count`: Number of words
- `has_urgency_markers`: Boolean indicator
- `features`: Pre-extracted TF-IDF features (100-dim)

## Reward Function
- Accuracy bonus: +1.0 for correct, -0.5 for incorrect
- Speed bonus: Up to +0.2 based on processing time
- Total reward range: [-0.5, 1.2]

## OpenEnv Specification
This environment fully implements the OpenEnv specification including:
- Typed observation and action spaces
- step() / reset() / state() API
- Deterministic grading
- Reproducible evaluation
                """)
            
            # Tab 3: Baseline Evaluation
            with gr.Tab("Baseline Evaluation"):
                gr.Markdown("## Run Baseline Agent Evaluation")
                
                difficulty_select = gr.Radio(
                    choices=["easy", "medium", "hard"],
                    value="easy",
                    label="Select Difficulty"
                )
                
                eval_btn = gr.Button("Run Evaluation")
                eval_output = gr.Markdown(label="Results")
                
                eval_btn.click(
                    run_baseline_evaluation,
                    inputs=[difficulty_select],
                    outputs=[eval_output]
                )
            
            # Tab 4: Specification
            with gr.Tab("OpenEnv Spec"):
                gr.Markdown("""
## OpenEnv Specification

### Environment Details
- **Name**: document-classification-env
- **Version**: 1.0.0
- **Framework**: Gymnasium (OpenAI Gym compatible)

### Difficulty Levels
1. **Easy** (difficulty=0.3)
   - 5 categories
   - 100 documents
   - Pre-extracted features
   - Target score: 0.95

2. **Medium** (difficulty=0.6)
   - 10 categories
   - 500 documents
   - 2 second time limit per action
   - Target score: 0.85

3. **Hard** (difficulty=1.0)
   - 20 categories
   - 1000 documents
   - 1 second time limit per action
   - Target score: 0.75

### Reproducibility
- Deterministic data generation with seed control
- Fixed test sets for grading
- Reproducible baseline scores

### Deployment
- Docker containerization
- Hugging Face Spaces integration
- Full API compatibility

### Success Criteria
- ✓ Real-world task simulation
- ✓ 3 progressive difficulty levels
- ✓ Agent graders with 0.0-1.0 scoring
- ✓ Reproducible baseline results
                """)
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
