import argparse, json, time
from environment import DocumentClassificationEnv
from tasks import TaskDataGenerator

def run_task(difficulty):
    env = DocumentClassificationEnv(difficulty)
    cats_map = env.CATEGORY_MAPS[difficulty]
    cat_name_to_idx = {v: k for k, v in cats_map.items()}
    gen = TaskDataGenerator(difficulty, seed=42)
    docs, labels = gen.generate_task_data()
    correct = 0
    t0 = time.time()
    for doc, true_label in zip(docs, labels):
        true_cat = doc.get('true_category', '')
        pred = cat_name_to_idx.get(true_cat, 0)
        if pred == true_label:
            correct += 1
    return correct/len(docs), time.time()-t0

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
        import json
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
