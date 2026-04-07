import gradio as gr
import threading
from flask import Flask, request, jsonify
from fastapi import Request
from fastapi.responses import JSONResponse
from environment import DocumentClassificationEnv
from baseline_inference import run_task, load_or_train
from agent import TicketAgent

print("Pre-training models...")
models = {}
for d in ["easy", "medium", "hard"]:
    models[d] = load_or_train(d)
    print(f"  {d} model ready!")
print("All models ready!")

agent = TicketAgent(model_dir=".")
print("Agent ready!")

CATEGORIES = {
    "easy":   ["General","Billing","Support","Technical","HR"],
    "medium": ["General","Billing","Support","Technical","HR","Legal","Sales","Marketing","Operations","Complaints"],
    "hard":   ["Cat_"+str(i) for i in range(22)]
}

api = Flask(__name__)
api_envs = {}

@api.route("/api/reset", methods=["POST"])
def api_reset():
    data = request.json or {}
    difficulty = data.get("difficulty", "easy")
    seed = data.get("seed", 42)
    env = DocumentClassificationEnv(task_difficulty=difficulty, seed=seed)
    obs, _ = env.reset(seed=seed)
    api_envs["current"] = env
    return jsonify({"observation": {k: v.tolist() if hasattr(v, "tolist") else v for k, v in obs.items()}})

@api.route("/api/step", methods=["POST"])
def api_step():
    data = request.json or {}
    action = data.get("action", 0)
    env = api_envs.get("current")
    if env is None:
        return jsonify({"error": "Call /api/reset first"}), 400
    obs, reward, done, _, info = env.step(int(action))
    return jsonify({"observation": {k: v.tolist() if hasattr(v, "tolist") else v for k, v in obs.items()}, "reward": reward, "done": done, "info": info})

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

def run_flask():
    api.run(host="0.0.0.0", port=7861, debug=False, use_reloader=False, threaded=True)

env_state = {"env": None, "difficulty": "easy"}

def create_env(difficulty):
    e = DocumentClassificationEnv(task_difficulty=difficulty, seed=42)
    obs, _ = e.reset()
    env_state["env"] = e
    env_state["difficulty"] = difficulty
    model = models[difficulty]
    cats = CATEGORIES[difficulty]
    proba = model.predict_proba([obs["content"]])[0]
    top3 = sorted(enumerate(proba), key=lambda x: -x[1])[:3]
    explain = "\n".join(["  "+str(cats[i] if i < len(cats) else "Cat_"+str(i))+": "+f"{p*100:.1f}%" for i, p in top3])
    pred_idx = int(model.predict([obs["content"]])[0])
    pred_cat = cats[pred_idx] if pred_idx < len(cats) else "Cat_"+str(pred_idx)
    return ("Environment created! Difficulty: "+difficulty, obs["content"], "Model predicts: "+pred_cat+"\n\nTop 3:\n"+explain, "")

def classify_doc(category_idx):
    e = env_state["env"]
    if e is None:
        return "Create environment first!", "", "", ""
    obs, reward, done, _, info = e.step(int(category_idx))
    res = "Reward: "+f"{reward:.2f}"+" | Correct: "+str(info.get("is_correct", False))
    if done:
        acc = info.get("episode_accuracy", 0)
        res += " | Final Accuracy: "+f"{acc:.2%}"
        return res, "Episode complete", "", str(info)
    difficulty = env_state["difficulty"]
    model = models[difficulty]
    cats = CATEGORIES[difficulty]
    proba = model.predict_proba([obs["content"]])[0]
    top3 = sorted(enumerate(proba), key=lambda x: -x[1])[:3]
    explain = "\n".join(["  "+str(cats[i] if i < len(cats) else "Cat_"+str(i))+": "+f"{p*100:.1f}%" for i, p in top3])
    pred_idx = int(model.predict([obs["content"]])[0])
    pred_cat = cats[pred_idx] if pred_idx < len(cats) else "Cat_"+str(pred_idx)
    return (res, obs["content"], "Model predicts: "+pred_cat+"\n\nTop 3:\n"+explain, str(info))

def run_baseline_all():
    rows = []
    for d in ["easy", "medium", "hard"]:
        score, t = run_task(d)
        rows.append([d.upper(), f"{score:.4f}", f"{score*100:.1f}%", f"{t:.1f}s"])
    return rows

def process_ticket_ui(ticket_text, difficulty):
    if not ticket_text.strip():
        return "Please enter ticket text!", "", "", "", "", ""
    result = agent.process_ticket(ticket_text, difficulty=difficulty)
    cat = "Category: "+result["category"]+" ("+f"{result['confidence']*100:.1f}"+"% confidence)"
    pri = "Priority: "+result["priority"].upper()
    dept = "Department: "+result["department"]+"\nEmail: "+result["email"]+"\nSLA: "+result["sla"]
    top3 = "\n".join(["  "+c+": "+f"{p*100:.1f}"+"%" for c, p in result["top3"]])
    ref = "Ref: "+result["ref_id"]+" | "+result["timestamp"][:19]
    return cat, pri, dept, top3, ref, result["reply"]

SAMPLE_TICKETS = {
    "Billing complaint": "My invoice shows an incorrect amount. I was charged $150 but should have been charged $75.",
    "Bug report": "The application crashes whenever I try to upload a file larger than 10MB.",
    "HR query": "I have a question about the company maternity leave policy.",
    "Chat message": "Hey, I cannot log into my account. The password reset link is not working.",
    "Email ticket": "Dear Support, I would like to inquire about upgrading my subscription plan.",
}

def load_sample(name):
    return SAMPLE_TICKETS.get(name, "")

with gr.Blocks(title="Document Classification OpenEnv") as demo:
    gr.Markdown("# Document Classification OpenEnv")
    gr.Markdown("Real-world customer support ticket routing environment for RL agent training.")
    with gr.Tab("Agent Demo"):
        with gr.Row():
            with gr.Column(scale=2):
                sample_dd = gr.Dropdown(choices=list(SAMPLE_TICKETS.keys()), label="Load Sample Ticket", value=None)
                ticket_input = gr.Textbox(label="Ticket Text", lines=6, placeholder="Paste email, chat message, bug report...")
                diff_agent = gr.Radio(["easy", "medium", "hard"], value="medium", label="Model Difficulty")
                btn_process = gr.Button("Process Ticket", variant="primary")
            with gr.Column(scale=2):
                ref_out_box = gr.Textbox(label="Reference ID")
                cat_out_box = gr.Textbox(label="Category")
                pri_out_box = gr.Textbox(label="Priority")
                dept_out_box = gr.Textbox(label="Routed To", lines=3)
                top3_out_box = gr.Textbox(label="Top 3 Predictions", lines=3)
        reply_out_box = gr.Textbox(label="Auto-Generated Reply", lines=12)
        sample_dd.change(load_sample, inputs=sample_dd, outputs=ticket_input)
        btn_process.click(process_ticket_ui, inputs=[ticket_input, diff_agent], outputs=[cat_out_box, pri_out_box, dept_out_box, top3_out_box, ref_out_box, reply_out_box])
    with gr.Tab("Interactive Demo"):
        diff = gr.Radio(["easy", "medium", "hard"], value="easy", label="Difficulty")
        btn_create = gr.Button("Create Environment", variant="primary")
        status = gr.Textbox(label="Status")
        with gr.Row():
            doc_content = gr.Textbox(label="Document Content", lines=6)
            explain_box = gr.Textbox(label="Model Reasoning", lines=6)
        category = gr.Number(label="Category Index", value=0)
        btn_classify = gr.Button("Classify Document")
        result_box = gr.Textbox(label="Result")
        info_box = gr.Textbox(label="Info")
        btn_create.click(create_env, inputs=diff, outputs=[status, doc_content, explain_box, result_box])
        btn_classify.click(classify_doc, inputs=category, outputs=[result_box, doc_content, explain_box, info_box])
    with gr.Tab("Baseline Evaluation"):
        btn_eval = gr.Button("Run All Tasks", variant="primary")
        score_table = gr.Dataframe(headers=["Task", "Score", "Accuracy", "Time"], label="Results")
        btn_eval.click(run_baseline_all, outputs=score_table)
    with gr.Tab("Environment Info"):
        gr.Markdown("## API Endpoints (port 7861)\n- POST /api/reset\n- POST /api/step\n- POST /api/run\n- POST /api/agent/process\n- POST /api/agent/classify")

# Mount /reset and /step on Gradio's FastAPI app for the hackathon checker
@demo.app.post("/reset")
async def reset_endpoint(request: Request):
    data = await request.json()
    difficulty = data.get("difficulty", "easy")
    seed = data.get("seed", 42)
    env = DocumentClassificationEnv(task_difficulty=difficulty, seed=seed)
    obs, _ = env.reset(seed=seed)
    api_envs["current"] = env
    return JSONResponse({"observation": {k: v.tolist() if hasattr(v, "tolist") else v for k, v in obs.items()}})

@demo.app.post("/step")
async def step_endpoint(request: Request):
    data = await request.json()
    action = data.get("action", 0)
    env = api_envs.get("current")
    if env is None:
        return JSONResponse({"error": "Call /reset first"}, status_code=400)
    obs, reward, done, _, info = env.step(int(action))
    return JSONResponse({
        "observation": {k: v.tolist() if hasattr(v, "tolist") else v for k, v in obs.items()},
        "reward": reward, "done": done, "info": info
    })

if __name__ == "__main__":
    t = threading.Thread(target=run_flask, daemon=True)
    t.start()
    demo.launch(server_name="0.0.0.0", server_port=7860)