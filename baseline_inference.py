"""
Quick script to test the baseline agent on all difficulty levels.
Just runs it through the test set and shows scores.
"""

import argparse
import json
import numpy as np
from grading import AgentGrader, BaselineAgent


def run_baseline_evaluation():
    """Run baseline on all tasks"""
    
    parser = argparse.ArgumentParser(description="Test baseline agent")
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Which task(s) to test"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed results"
    )
    parser.add_argument(
        "--output",
        default="baseline_results.json",
        help="Save results to this file"
    )
    
    args = parser.parse_args()
    
    # Pick which tasks to run
    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    results = {}
    
    print("="*70)
    print("Baseline Agent Evaluation")
    print("="*70)
    
    for task in tasks:
        print(f"\n[{task.upper()}] Running...")
        
        # Set up grader and agent
        grader = AgentGrader(task)
        agent = BaselineAgent(task)
        
        # Run it
        score, metrics = grader.grade_agent(
            agent.decide,
            verbose=args.verbose
        )
        
        # Store results
        results[task] = {
            "score": float(score),
            "metrics": {
                "accuracy": float(metrics["accuracy"]),
                "total_documents": int(metrics["total_documents"]),
                "correct": int(metrics["correct_classifications"]),
                "average_reward": float(metrics["average_reward"]),
                "total_reward": float(metrics["total_reward"]),
                "avg_time_ms": float(metrics["average_processing_time_ms"]),
                "total_time_s": float(metrics["total_time_seconds"]),
            }
        }
        
        # Optional metrics
        for key in ["macro_precision", "macro_recall", "macro_f1"]:
            if key in metrics:
                results[task]["metrics"][key] = float(metrics[key])
        
        print(f"✓ {task.upper()} - Score: {score:.4f}")
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    for task in tasks:
        score = results[task]["score"]
        accuracy = results[task]["metrics"]["accuracy"]
        print(f"{task.upper():8} - Score: {score:.4f}  |  Accuracy: {accuracy:.4f}")
    
    print("="*70)
    
    # Save to file
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")
    
    return results


if __name__ == "__main__":
    run_baseline_evaluation()
