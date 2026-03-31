"""
Baseline inference script - keyword matching agent
"""

import json
import argparse
import time
from environment import DocumentClassificationEnv


def keyword_agent(observation: dict, task_difficulty: str) -> int:
    """
    Improved keyword-based agent with priority ordering.
    More specific patterns checked first.
    """
    content = observation["content"].lower()
    has_urgency = observation["has_urgency_markers"][0]

    if task_difficulty == "easy":
        if any(w in content for w in ["invoice", "billing", "payment", "charge", "bill", "price", "cost"]):
            return 1  # Billing
        elif any(w in content for w in ["technical", "bug", "error", "software", "api", "integration", "crash"]):
            return 3  # Technical
        elif any(w in content for w in ["support", "help", "assist", "issue", "problem", "trouble"]):
            return 2  # Support
        elif any(w in content for w in ["hr", "payroll", "salary", "benefits", "employee", "complaint"]):
            return 4  # HR
        else:
            return 0  # General

    elif task_difficulty == "medium":
        if any(w in content for w in ["overcharged", "dispute", "incorrect charge", "should not appear", "billing error"]):
            return 2  # Billing-Dispute
        elif any(w in content for w in ["invoice", "billing", "payment", "charge", "bill"]):
            return 1  # Billing
        elif any(w in content for w in ["bug", "crash", "error", "broken", "not working", "throws"]):
            return 5  # Technical-Bug
        elif any(w in content for w in ["technical", "software", "api", "integration", "documentation"]):
            return 4  # Technical
        elif any(w in content for w in ["paycheck", "salary", "payroll", "deductions", "payroll cycle"]):
            return 6  # HR-Payroll
        elif any(w in content for w in ["benefits", "health", "enroll", "eligible", "insurance"]):
            return 7  # HR-Benefits
        elif any(w in content for w in ["legal", "contract", "terms", "compliance", "regulation"]):
            return 8  # Legal
        elif any(w in content for w in ["executive", "senior manager", "management", "strategic"]):
            return 9  # Executive
        elif any(w in content for w in ["support", "help", "assist", "issue"]):
            return 3  # Support
        else:
            return 0  # General

    else:  # hard: 20 categories
        # Billing
        if any(w in content for w in ["refund", "return", "money back", "get refunded"]):
            return 3  # Billing-Refund
        elif any(w in content for w in ["overcharged", "dispute", "incorrect", "should not appear", "billing error", "pricing is incorrect"]):
            return 2  # Billing-Dispute
        elif any(w in content for w in ["invoice", "billing", "payment", "charge", "bill", "billing address"]):
            return 1  # Billing

        # Support
        elif has_urgency or any(w in content for w in ["urgent", "critical", "emergency", "immediate", "asap", "production", "system is down"]):
            return 4  # Support-Urgent
        elif any(w in content for w in ["walk me through", "how do i", "how to use", "basic feature", "functionality", "understand this"]):
            return 5  # Support-Normal
        elif any(w in content for w in ["support", "help", "assist", "customer support"]):
            return 5  # Support-Normal

        # Technical
        elif any(w in content for w in ["feature request", "new feature", "add functionality", "implement", "capability", "add support for"]):
            return 8  # Technical-Feature
        elif any(w in content for w in ["bug", "crash", "error", "broken", "not working", "throws", "crashes", "bug report"]):
            return 7  # Technical-Bug
        elif any(w in content for w in ["technical", "software", "api", "integration", "documentation", "system"]):
            return 6  # Technical

        # HR
        elif any(w in content for w in ["formal complaint", "file a complaint", "report a workplace", "filing an official"]):
            return 11  # HR-Complaint
        elif any(w in content for w in ["paycheck", "salary", "payroll", "deductions", "payroll cycle"]):
            return 9  # HR-Payroll
        elif any(w in content for w in ["benefits", "health", "enroll", "eligible", "insurance", "benefits plan"]):
            return 10  # HR-Benefits

        # Legal
        elif any(w in content for w in ["compliance", "regulation", "regulatory", "requirement", "regulatory compliance"]):
            return 14  # Legal-Compliance
        elif any(w in content for w in ["contract", "signing", "provisions", "obligations", "agreement", "contract terms"]):
            return 13  # Legal-Contract
        elif any(w in content for w in ["legal", "advice", "documentation", "legal matters"]):
            return 12  # Legal

        # Executive
        elif any(w in content for w in ["strategic", "partnership", "corporate strategy", "initiative", "strategic-level"]):
            return 16  # Executive-Strategic
        elif any(w in content for w in ["executive", "senior manager", "management", "schedule a meeting"]):
            return 15  # Executive

        # Others
        elif any(w in content for w in ["financial", "finance", "statements", "financial records", "financial matters"]):
            return 17  # Finance
        elif any(w in content for w in ["marketing", "marketing partnership", "marketing proposal", "collaboration"]):
            return 18  # Marketing
        elif any(w in content for w in ["operational", "operations", "scheduling", "procedures", "efficiency"]):
            return 19  # Operations

        else:
            return 0  # General


def run_task(task_difficulty: str, num_episodes: int = 1) -> dict:
    """Run the agent on a specific task"""
    env = DocumentClassificationEnv(task_difficulty=task_difficulty, seed=42)

    all_results = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode * 100)
        episode_rewards = []
        episode_correct = 0
        episode_total = 0

        terminated = False
        while not terminated:
            action = keyword_agent(obs, task_difficulty)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_rewards.append(reward)
            if info.get("is_correct", False):
                episode_correct += 1
            episode_total += 1

        episode_result = {
            "episode": episode,
            "total_reward": sum(episode_rewards),
            "accuracy": episode_correct / episode_total if episode_total > 0 else 0.0,
            "num_documents": episode_total,
        }
        all_results.append(episode_result)

    avg_accuracy = sum(r["accuracy"] for r in all_results) / len(all_results)
    avg_reward = sum(r["total_reward"] for r in all_results) / len(all_results)

    return {
        "task": task_difficulty,
        "num_episodes": num_episodes,
        "average_accuracy": avg_accuracy,
        "average_total_reward": avg_reward,
        "episodes": all_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Baseline inference for document classification")
    parser.add_argument("--task", type=str, default="all", choices=["easy", "medium", "hard", "all"])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output", type=str, default="baseline_results.json")
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    results = {}

    for task in tasks:
        print(f"\nRunning {task.upper()} task...")
        start = time.time()
        task_results = run_task(task, args.episodes)
        elapsed = time.time() - start

        score = task_results["average_accuracy"]
        results[task] = task_results
        results[task]["score"] = score

        print(f"  {task.upper()} Score: {score:.4f} ({elapsed:.1f}s)")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")

    print("\n=== BASELINE SCORES ===")
    for task in tasks:
        print(f"  {task.upper()} Score: {results[task]['score']:.4f}")


if __name__ == "__main__":
    main()
