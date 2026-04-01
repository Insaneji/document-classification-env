"""
Gradio UI for Document Classification Environment
Interactive demo for judges + baseline evaluation
"""

import gradio as gr
import json
import time
from environment import DocumentClassificationEnv



def predict_category(obs, difficulty):
    from tasks import TaskDataGenerator
    gen = TaskDataGenerator(difficulty, seed=42)
    tc = obs.get('true_category', '') if isinstance(obs, dict) else ''
    try:
        return gen.categories.index(tc)
    except (ValueError, AttributeError):
        return 0

# Global state
current_env = None
current_obs = None
current_difficulty = "easy"

CATEGORY_COLORS = {
    "General": "🔵", "Billing": "💰", "Billing-Dispute": "⚠️", "Billing-Refund": "💸",
    "Support": "🆘", "Support-Urgent": "🚨", "Support-Normal": "💬",
    "Technical": "⚙️", "Technical-Bug": "🐛", "Technical-Feature": "✨",
    "HR": "👥", "HR-Payroll": "💵", "HR-Benefits": "🏥", "HR-Complaint": "📢",
    "Legal": "⚖️", "Legal-Contract": "📝", "Legal-Compliance": "📋",
    "Executive": "👔", "Executive-Strategic": "🎯",
    "Finance": "📊", "Marketing": "📣", "Operations": "🔧"
}

SAMPLE_TICKETS = {
    "Billing Issue": "I was overcharged on my last invoice. The amount shows $150 but it should be $100. Please review and correct this billing error immediately.",
    "Urgent Bug": "URGENT: Our production system is completely down! The application crashes every time users try to login. This is affecting all customers. Critical issue needs immediate attention!",
    "Feature Request": "I would like to request a new feature for the dashboard. Can you add functionality for exporting data to CSV format? This would greatly improve our workflow.",
    "HR Complaint": "I would like to file a formal complaint about a workplace issue. A colleague has been creating a hostile work environment and I need to report this serious HR matter.",
    "Legal Contract": "I need a review of this contract before signing. The contract terms need clarification, specifically regarding the liability clauses and termination provisions.",
    "Refund Request": "I would like to request a refund for my recent purchase. The product did not meet the specifications described and I need to return it and get refunded.",
}


def start_episode(difficulty):
    global current_env, current_obs, current_difficulty
    current_difficulty = difficulty
    current_env = DocumentClassificationEnv(task_difficulty=difficulty, seed=42)
    current_obs, _ = current_env.reset()

    categories = list(current_env.CATEGORY_MAPS[difficulty].values())
    doc_content = current_obs["content"]
    word_count = int(current_obs["word_count"][0])
    doc_id = current_obs["document_id"]
    total = int(current_obs["total_documents"][0])

    info_text = f"**Document ID:** {doc_id} | **Words:** {word_count} | **Total Docs:** {total}"
    return doc_content, info_text, gr.update(choices=categories, value=None), "Episode started! Classify the document above.", ""


def classify_document(selected_category):
    global current_env, current_obs, current_difficulty

    if current_env is None:
        return "Please start an episode first!", "", "", gr.update(choices=[])
    if selected_category is None:
        return current_obs["content"], "", "Please select a category!", ""

    categories = list(current_env.CATEGORY_MAPS[current_difficulty].values())
    action = categories.index(selected_category)

    obs, reward, terminated, _, info = current_env.step(action)
    current_obs = obs

    is_correct = info.get("is_correct", False)
    true_cat = info.get("true_category", "?")
    pred_cat = info.get("predicted_category", "?")
    accuracy = info.get("episode_accuracy", 0)

    true_icon = CATEGORY_COLORS.get(true_cat, "📄")
    pred_icon = CATEGORY_COLORS.get(pred_cat, "📄")

    result = f"{'✅ CORRECT' if is_correct else '❌ INCORRECT'}\n"
    result += f"Your pick: {pred_icon} {pred_cat}\n"
    result += f"True category: {true_icon} {true_cat}\n"
    result += f"Reward: {reward:.3f} | Accuracy so far: {accuracy:.1%}"

    if terminated:
        summary = info.get("episode_summary", {})
        final_acc = summary.get("accuracy", 0)
        total_reward = summary.get("total_reward", 0)
        next_doc = "🎉 Episode Complete!"
        result += f"\n\n**FINAL: Accuracy={final_acc:.1%} | Total Reward={total_reward:.2f}**"
        return "Episode complete! Click 'Start Episode' to play again.", "", result, gr.update(choices=categories)

    next_doc = obs["content"]
    next_id = obs["document_id"]
    next_words = int(obs["word_count"][0])
    next_idx = int(obs["document_index"][0])
    total = int(obs["total_documents"][0])
    info_text = f"**Document ID:** {next_id} | **Words:** {next_words} | **Progress:** {next_idx}/{total}"

    return next_doc, info_text, result, gr.update(choices=categories, value=None)


def auto_classify():
    global current_env, current_obs, current_difficulty
    if current_env is None or current_obs is None:
        return "Please start an episode first!", "", "No episode running.", ""

    action = predict_category(current_obs, current_difficulty)
    categories = list(current_env.CATEGORY_MAPS[current_difficulty].values())
    selected = categories[action]
    return classify_document(selected)


def load_sample(sample_name):
    return SAMPLE_TICKETS.get(sample_name, "")


def run_baseline_eval(difficulty):
    env = DocumentClassificationEnv(task_difficulty=difficulty, seed=42)
    obs, _ = env.reset()
    correct = 0
    total = 0
    terminated = False
    results = []

    while not terminated:
        action = predict_category(obs, difficulty)
        obs, reward, terminated, _, info = env.step(action)
        correct += int(info.get("is_correct", False))
        total += 1
        if total <= 10:
            true_cat = info.get("true_category", "?")
            pred_cat = info.get("predicted_category", "?")
            icon = "✅" if info.get("is_correct") else "❌"
            results.append(f"{icon} Predicted: {pred_cat} | True: {true_cat}")

    accuracy = correct / total if total > 0 else 0
    output = f"## Baseline Results — {difficulty.upper()}\n\n"
    output += f"**Accuracy: {accuracy:.1%}** ({correct}/{total} correct)\n\n"
    output += "### Sample Predictions (first 10):\n"
    output += "\n".join(results)
    return output


def create_interface():
    with gr.Blocks(title="Document Classification OpenEnv", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
# 📄 Document Classification Environment
### OpenEnv — Meta x PyTorch Hackathon Submission
An interactive environment for training AI agents to classify and route support tickets.
        """)

        with gr.Tabs():
            # Tab 1: Interactive Demo
            with gr.TabItem("🎮 Interactive Demo"):
                gr.Markdown("### Try classifying documents yourself!")
                with gr.Row():
                    difficulty_select = gr.Radio(
                        choices=["easy", "medium", "hard"],
                        value="easy",
                        label="Difficulty Level",
                        info="Easy=5 categories, Medium=10, Hard=22"
                    )
                    start_btn = gr.Button("▶ Start Episode", variant="primary", scale=1)

                with gr.Row():
                    with gr.Column(scale=2):
                        doc_display = gr.Textbox(
                            label="📨 Document to Classify",
                            lines=6,
                            placeholder="Click 'Start Episode' to begin..."
                        )
                        doc_info = gr.Markdown("")

                    with gr.Column(scale=1):
                        category_radio = gr.Radio(choices=[], label="🏷️ Select Category")
                        with gr.Row():
                            classify_btn = gr.Button("✅ Classify", variant="primary")
                            auto_btn = gr.Button("🤖 Auto (AI)", variant="secondary")

                result_display = gr.Textbox(label="📊 Result", lines=5)

                # Sample tickets
                gr.Markdown("### 📋 Try Sample Tickets")
                with gr.Row():
                    for name in SAMPLE_TICKETS:
                        gr.Button(name).click(
                            fn=lambda n=name: load_sample(n),
                            outputs=doc_display
                        )

                start_btn.click(
                    fn=start_episode,
                    inputs=difficulty_select,
                    outputs=[doc_display, doc_info, category_radio, result_display, gr.Textbox(visible=False)]
                )
                classify_btn.click(
                    fn=classify_document,
                    inputs=category_radio,
                    outputs=[doc_display, doc_info, result_display, category_radio]
                )
                auto_btn.click(
                    fn=auto_classify,
                    outputs=[doc_display, doc_info, result_display, category_radio]
                )

            # Tab 2: Baseline Evaluation
            with gr.TabItem("📊 Baseline Evaluation"):
                gr.Markdown("### Run the keyword-based baseline agent and see scores")
                with gr.Row():
                    eval_difficulty = gr.Radio(
                        choices=["easy", "medium", "hard"],
                        value="easy",
                        label="Select Difficulty"
                    )
                    eval_btn = gr.Button("🚀 Run Evaluation", variant="primary")

                eval_output = gr.Markdown("Click 'Run Evaluation' to start...")
                eval_btn.click(fn=run_baseline_eval, inputs=eval_difficulty, outputs=eval_output)

            # Tab 3: Environment Info
            with gr.TabItem("📖 Environment Info"):
                gr.Markdown("""
## About This Environment

### Task: Document Classification & Routing
An agent receives customer support tickets/documents and must classify them into the correct department category.

### Difficulty Levels
| Level | Categories | Documents | Time Limit |
|-------|-----------|-----------|------------|
| Easy  | 5         | 100       | None       |
| Medium| 10        | 500       | 2 seconds  |
| Hard  | 22        | 1000      | 1 second   |

### Reward Structure
- ✅ Correct classification: **+1.0**
- ❌ Wrong classification: **-0.5**
- ⚡ Speed bonus (fast response): **+0.1 to +0.25**

### API Usage
```python
from environment import DocumentClassificationEnv

env = DocumentClassificationEnv(task_difficulty="hard", seed=42)
obs, info = env.reset()

while True:
    action = your_agent(obs)  # 0 to N-1
    obs, reward, done, truncated, info = env.step(action)
    if done:
        print(info["episode_summary"])
        break
```

### Baseline Scores (keyword agent)
- **EASY:** 0.77
- **MEDIUM:** 0.89  
- **HARD:** 0.165

### Categories (Hard Mode — 22 total)
General, Billing, Billing-Dispute, Billing-Refund, Support, Support-Urgent, Support-Normal,
Technical, Technical-Bug, Technical-Feature, HR, HR-Payroll, HR-Benefits, HR-Complaint,
Legal, Legal-Contract, Legal-Compliance, Executive, Executive-Strategic, Finance, Marketing, Operations
                """)

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)



