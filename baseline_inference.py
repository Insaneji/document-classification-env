import argparse
import json
import time
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from tasks import TaskDataGenerator
from environment import DocumentClassificationEnv

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

def get_model_path(difficulty):
    return os.path.join(MODEL_DIR, f"model_{difficulty}.pkl")

def train_model(difficulty):
    print(f"Training {difficulty} model...")
    all_texts, all_labels = [], []
    for seed in range(10):
        gen = TaskDataGenerator(difficulty, seed=seed)
        docs, labels = gen.generate_task_data()
        for doc, label in zip(docs, labels):
            all_texts.append(doc["content"])
            all_labels.append(label)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=5000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=5.0, solver="lbfgs"))
    ])
    pipeline.fit(all_texts, all_labels)
    with open(get_model_path(difficulty), "wb") as f:
        pickle.dump(pipeline, f)
    print(f"  Saved!")
    return pipeline

def load_or_train(difficulty):
    path = get_model_path(difficulty)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return train_model(difficulty)

def run_task(difficulty):
    model = load_or_train(difficulty)
    env = DocumentClassificationEnv(task_difficulty=difficulty, seed=42)
    obs, _ = env.reset()
    correct, total, t0, terminated = 0, 0, time.time(), False
    while not terminated:
        action = int(model.predict([obs["content"]])[0])
        obs, reward, terminated, _, info = env.step(action)
        correct += int(info.get("is_correct", False))
        total += 1
    return correct/total if total > 0 else 0, time.time()-t0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all")
    parser.add_argument("--output", default="baseline_results.json")
    args = parser.parse_args()
    tasks = ["easy","medium","hard"] if args.task=="all" else [args.task]
    results = {}
    for task in tasks:
        print(f"\nRunning {task.upper()} task...")
        score, elapsed = run_task(task)
        results[task] = {"score": round(score,4), "time": round(elapsed,2)}
        print(f"  {task.upper()} Score: {score:.4f} ({elapsed:.1f}s)")
    print("\n=== BASELINE SCORES ===")
    for t,r in results.items():
        print(f"  {t.upper()} Score: {r['score']:.4f}")
    with open(args.output,"w") as f:
        json.dump(results,f,indent=2)

if __name__ == "__main__":
    main()