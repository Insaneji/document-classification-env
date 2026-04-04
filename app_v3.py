import gradio as gr
import threading
import os
import pickle
import time
import json
from flask import Flask, request, jsonify
from environment import DocumentClassificationEnv
from baseline_inference import run_task, load_or_train
from agent import TicketAgent
from rl_trainer import RLTrainer
import numpy as np

print("=" * 60)
print("    MetaX AI Agent - HACKATHON EDITION")
print("=" * 60)
print("Loading AI models...")

models = {}
for d in ["easy", "medium", "hard"]:
    models[d] = load_or_train(d)
    print(f"  [OK] {d.upper()} model loaded")

agent = TicketAgent(model_dir=".")
print("[OK] Agent ready")

CATEGORIES = {
    "easy": ["General", "Billing", "Support", "Technical", "HR"],
    "medium": [
        "General",
        "Billing",
        "Support",
        "Technical",
        "HR",
        "Legal",
        "Sales",
        "Marketing",
        "Operations",
        "Complaints",
    ],
    "hard": [f"Cat_{i}" for i in range(22)],
}

api = Flask(__name__)
api_envs = {}

rl_state = {"trainer": None, "is_training": False}


@api.route("/api/reset", methods=["POST"])
def api_reset():
    data = request.json or {}
    difficulty = data.get("difficulty", "easy")
    seed = data.get("seed", 42)
    env = DocumentClassificationEnv(task_difficulty=difficulty, seed=seed)
    obs, _ = env.reset(seed=seed)
    api_envs["current"] = env
    return jsonify(
        {
            "observation": {
                k: v.tolist() if hasattr(v, "tolist") else v for k, v in obs.items()
            }
        }
    )


@api.route("/api/step", methods=["POST"])
def api_step():
    data = request.json or {}
    action = data.get("action", 0)
    env = api_envs.get("current")
    if env is None:
        return jsonify({"error": "Call /api/reset first"}), 400
    obs, reward, done, _, info = env.step(int(action))
    return jsonify(
        {
            "observation": {
                k: v.tolist() if hasattr(v, "tolist") else v for k, v in obs.items()
            },
            "reward": reward,
            "done": done,
            "info": info,
        }
    )


@api.route("/api/run", methods=["POST"])
def api_run():
    data = request.json or {}
    difficulty = data.get("difficulty", "easy")
    score, t = run_task(difficulty)
    return jsonify({"difficulty": difficulty, "score": score, "time": t})


@api.route("/api/agent/process", methods=["POST"])
def api_agent_process():
    data = request.json or {}
    text = data.get("text", "")
    difficulty = data.get("difficulty", "easy")
    ticket_id = data.get("ticket_id", None)
    if not text:
        return jsonify({"error": "text field required"}), 400
    result = agent.process_ticket(text, difficulty=difficulty, ticket_id=ticket_id)
    return jsonify(result)


@api.route("/api/agent/classify", methods=["POST"])
def api_agent_classify():
    data = request.json or {}
    text = data.get("text", "")
    difficulty = data.get("difficulty", "easy")
    category, scores = agent.classify(text, difficulty)
    priority = agent.get_priority(text)
    return jsonify({"category": category, "priority": priority, "scores": scores})


@api.route("/api/rl/train", methods=["POST"])
def api_rl_train():
    global rl_state
    data = request.json or {}
    difficulty = data.get("difficulty", "easy")
    episodes = data.get("episodes", 20)

    if rl_state.get("is_training"):
        return jsonify({"status": "Training already in progress"}), 400

    env = DocumentClassificationEnv(task_difficulty=difficulty, seed=42)
    trainer = RLTrainer(env, num_episodes=episodes)
    rl_state["trainer"] = trainer
    rl_state["is_training"] = True

    result = trainer.train_step()
    eval_result = trainer.evaluate(num_episodes=5)
    curve = trainer.generate_learning_curve()
    summary = trainer.get_metrics_summary()

    rl_state["is_training"] = False

    return jsonify(
        {
            "training_result": result,
            "evaluation": eval_result,
            "learning_curve": curve,
            "summary": summary,
        }
    )


@api.route("/api/rl/status", methods=["GET"])
def api_rl_status():
    if rl_state.get("trainer") is None:
        return jsonify({"status": "No training data"})
    return jsonify(rl_state["trainer"].get_metrics_summary())


def run_flask():
    api.run(host="0.0.0.0", port=7861, debug=False)


SAMPLE_TICKETS = {
    "Billing Complaint": "My invoice shows an incorrect amount! I was charged $500 but should have been charged $75. This is unacceptable!",
    "Critical Bug": "The application crashes whenever I try to upload a file larger than 10MB. This is critical!",
    "HR Policy": "I have a question about the company's maternity leave policy. How many weeks of paid leave am I entitled to?",
    "Login Issue": "I can't log into my account. The password reset link isn't working. Please help!",
    "Sales Inquiry": "I would like to inquire about upgrading our current subscription to include additional enterprise features.",
    "Legal Request": "We need legal review of the new vendor contract before signing.",
    "Frustrated Customer": "I'm extremely disappointed with the service. This is the third time this month I've had issues.",
}

env_state = {"env": None, "difficulty": "easy", "trainer": None}


def create_env(difficulty):
    e = DocumentClassificationEnv(task_difficulty=difficulty, seed=42)
    obs, _ = e.reset()
    env_state["env"] = e
    env_state["difficulty"] = difficulty
    model = models[difficulty]
    cats = CATEGORIES[difficulty]
    proba = model.predict_proba([obs["content"]])[0]
    top3 = sorted(enumerate(proba), key=lambda x: -x[1])[:3]
    explain = "\n".join([f"  {cats[i]}: {p * 100:.1f}%" for i, p in top3])
    pred_idx = int(model.predict([obs["content"]])[0])
    pred_cat = cats[pred_idx]
    return (
        f"Environment Ready | {difficulty.upper()}",
        obs["content"],
        f"ML Prediction: {pred_cat}\n\nTop 3:\n{explain}",
        "",
    )


def classify_doc(category_idx):
    e = env_state["env"]
    if e is None:
        return "Create environment first!", "", "", ""
    obs, reward, done, _, info = e.step(int(category_idx))
    result = (
        f"{'CORRECT' if info.get('is_correct') else 'WRONG'} | Reward: {reward:.2f}"
    )
    if done:
        acc = info.get("episode_accuracy", 0)
        result += f"\nEpisode Complete! Accuracy: {acc:.1%}"
        return result, "Episode complete", "", str(info)
    difficulty = env_state["difficulty"]
    model = models[difficulty]
    cats = CATEGORIES[difficulty]
    proba = model.predict_proba([obs["content"]])[0]
    top3 = sorted(enumerate(proba), key=lambda x: -x[1])[:3]
    explain = "\n".join([f"  {cats[i]}: {p * 100:.1f}%" for i, p in top3])
    pred_idx = int(model.predict([obs["content"]])[0])
    pred_cat = cats[pred_idx]
    return (
        result,
        obs["content"],
        f"ML Prediction: {pred_cat}\n\nTop 3:\n{explain}",
        str(info),
    )


def process_ticket_ui(ticket_text, difficulty):
    if not ticket_text.strip():
        return "Please enter ticket text!", "", "", "", "", ""
    result = agent.process_ticket(ticket_text, difficulty=difficulty)
    cat_out = f"{result['category']}\n   Confidence: {result['confidence'] * 100:.1f}%"
    pri_out = f"{result['priority'].upper()}"
    dept_out = f"{result['department']}\n   {result['email']}\n   SLA: {result['sla']}"
    top3_out = "\n".join([f"   {c}: {p * 100:.1f}%" for c, p in result["top3"]])
    ref_out = f"{result['ref_id']}\n   {result['timestamp'][:19]}"
    reply_out = result["reply"]
    return cat_out, pri_out, dept_out, top3_out, ref_out, reply_out


def run_rl_training(difficulty, num_episodes):
    difficulty = difficulty or "easy"
    num_episodes = num_episodes or 20
    try:
        env = DocumentClassificationEnv(task_difficulty=difficulty, seed=42)
        trainer = RLTrainer(env, num_episodes=num_episodes)
        env_state["trainer"] = trainer
        result = trainer.train_step()
        curve_img = trainer.generate_learning_curve()
        summary = trainer.get_metrics_summary()
        status = f"Training Complete!\n\nResults:\n   Episodes: {num_episodes}\n   Avg Reward: {result.get('avg_reward', 0):.2f}\n   Avg Accuracy: {result.get('avg_accuracy', 0):.1%}"
        eval_result = trainer.evaluate(num_episodes=5)
        status += f"\n\nEvaluation Accuracy: {eval_result.get('accuracy', 0):.1%}"
        return status, curve_img if curve_img else "", str(summary)
    except Exception as e:
        return f"Error: {str(e)}", "", ""


def load_sample(sample_name):
    return SAMPLE_TICKETS.get(sample_name, "")


def run_evaluation():
    rows = []
    for d in ["easy", "medium", "hard"]:
        score, t = run_task(d)
        rows.append([d.upper(), f"{score:.4f}", f"{score * 100:.1f}%", "Ready"])
    return rows


def process_image(file_obj):
    if file_obj is None:
        return "No image uploaded. Click 'Upload' button, select an image, then click 'Analyze Image'."
    return f"Image received: {type(file_obj)}. Image analysis requires PyTorch installation."


with gr.Blocks(title="MetaX AI - Hackathon Demo") as demo:
    gr.Markdown("""
    <div style="text-align: center; padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin-bottom: 25px;">
        <h1 style="color: white; margin: 0; font-size: 2.5em;">MetaX AI Agent</h1>
        <p style="color: white; opacity: 0.95;">Document Classification | RL Training | AI-Powered</p>
    </div>
    """)

    with gr.Tab("AI Agent Demo"):
        gr.Markdown("### AI-Powered Ticket Classification")
        with gr.Row():
            with gr.Column(scale=1):
                sample_dd = gr.Dropdown(
                    choices=list(SAMPLE_TICKETS.keys()), label="Load Sample"
                )
                ticket_input = gr.Textbox(
                    label="Ticket Text",
                    lines=5,
                    placeholder="Enter customer message...",
                )
                diff_agent = gr.Radio(
                    ["easy", "medium", "hard"], value="medium", label="Difficulty"
                )
                btn_process = gr.Button("Process Ticket", variant="primary", size="lg")
            with gr.Column(scale=1):
                cat_out_box = gr.Textbox(label="Category", lines=2)
                pri_out_box = gr.Textbox(label="Priority", lines=2)
                dept_out_box = gr.Textbox(label="Routing", lines=2)
                top3_out_box = gr.Textbox(label="Top Predictions", lines=2)
        with gr.Row():
            ref_out_box = gr.Textbox(label="Reference", lines=1)
        reply_out_box = gr.Textbox(label="AI Response", lines=8)
        sample_dd.change(load_sample, inputs=sample_dd, outputs=ticket_input)
        btn_process.click(
            process_ticket_ui,
            inputs=[ticket_input, diff_agent],
            outputs=[
                cat_out_box,
                pri_out_box,
                dept_out_box,
                top3_out_box,
                ref_out_box,
                reply_out_box,
            ],
        )

    with gr.Tab("RL Training"):
        gr.Markdown("### Train RL Agent")
        with gr.Row():
            with gr.Column(scale=1):
                diff_rl = gr.Radio(
                    ["easy", "medium", "hard"], value="easy", label="Difficulty"
                )
                episodes_slider = gr.Slider(10, 100, value=30, step=5, label="Episodes")
                btn_train = gr.Button("Start Training", variant="primary", size="lg")
            with gr.Column(scale=2):
                train_status = gr.Textbox(label="Status", lines=8)
        learning_curve_img = gr.Image(label="Training Progress")
        btn_train.click(
            run_rl_training,
            inputs=[diff_rl, episodes_slider],
            outputs=[train_status, learning_curve_img, train_status],
        )

    with gr.Tab("Multi-Modal"):
        gr.Markdown("### Image Upload (Experimental)")
        gr.Markdown(
            "Note: Full image analysis requires PyTorch. Install: `pip install torch transformers`"
        )
        with gr.Row():
            file_input = gr.File(
                label="Upload Image", file_count="single", file_types=["image"]
            )
            image_btn = gr.Button("Analyze Image")
        image_output = gr.Textbox(label="Result", lines=4)
        image_btn.click(process_image, inputs=file_input, outputs=image_output)

    with gr.Tab("Interactive Env"):
        gr.Markdown("### Test Classification Environment")
        with gr.Row():
            diff = gr.Radio(
                ["easy", "medium", "hard"], value="easy", label="Difficulty"
            )
            btn_create = gr.Button("Create Environment", variant="primary")
            status = gr.Textbox(label="Status")
        with gr.Row():
            doc_content = gr.Textbox(label="Document", lines=4)
            explain_box = gr.Textbox(label="Model Reasoning", lines=4)
        with gr.Row():
            category = gr.Number(label="Category Index", value=0, precision=0)
            btn_classify = gr.Button("Classify")
            result = gr.Textbox(label="Result", lines=2)
        btn_create.click(
            create_env, inputs=diff, outputs=[status, doc_content, explain_box, result]
        )
        btn_classify.click(
            classify_doc,
            inputs=category,
            outputs=[result, doc_content, explain_box, result],
        )

    with gr.Tab("Performance"):
        gr.Markdown("### Model Performance")
        btn_eval = gr.Button("Run Evaluation", variant="primary")
        score_table = gr.Dataframe(
            headers=["Task", "Score", "Accuracy", "Status"], label="Results"
        )
        btn_eval.click(run_evaluation, outputs=score_table)

    with gr.Tab("Info"):
        gr.Markdown("""
        ## API Endpoints (port 7861)
        - POST /api/reset
        - POST /api/step  
        - POST /api/agent/process
        - POST /api/rl/train
        - GET /api/rl/status
        
        ## Run
        python app_v3.py
        Then open http://localhost:7860
        """)

if __name__ == "__main__":
    t = threading.Thread(target=run_flask, daemon=True)
    t.start()
    demo.launch(server_name="0.0.0.0", server_port=7860)
