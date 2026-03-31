"""
Grade agents on document classification.
Uses hierarchical partial credit - related categories get partial score.
"""

import numpy as np
from typing import Dict, List
from environment import DocumentClassificationEnv


# Hierarchy map - categories that are "close" get partial credit
CATEGORY_HIERARCHY = {
    "Billing":           ["Billing-Dispute", "Billing-Refund"],
    "Billing-Dispute":   ["Billing", "Billing-Refund"],
    "Billing-Refund":    ["Billing", "Billing-Dispute"],
    "Support":           ["Support-Urgent", "Support-Normal"],
    "Support-Urgent":    ["Support", "Support-Normal"],
    "Support-Normal":    ["Support", "Support-Urgent"],
    "Technical":         ["Technical-Bug", "Technical-Feature"],
    "Technical-Bug":     ["Technical", "Technical-Feature"],
    "Technical-Feature": ["Technical", "Technical-Bug"],
    "HR":                ["HR-Payroll", "HR-Benefits", "HR-Complaint"],
    "HR-Payroll":        ["HR", "HR-Benefits"],
    "HR-Benefits":       ["HR", "HR-Payroll"],
    "HR-Complaint":      ["HR"],
    "Legal":             ["Legal-Contract", "Legal-Compliance"],
    "Legal-Contract":    ["Legal", "Legal-Compliance"],
    "Legal-Compliance":  ["Legal", "Legal-Contract"],
    "Executive":         ["Executive-Strategic"],
    "Executive-Strategic": ["Executive"],
}

PARTIAL_CREDIT = 0.4  # Score for picking a related category


def hierarchical_accuracy(true_label: str, pred_label: str) -> float:
    """
    Returns:
      1.0 if exact match
      0.4 if same parent category (e.g. Billing vs Billing-Dispute)
      0.0 otherwise
    """
    if true_label == pred_label:
        return 1.0
    related = CATEGORY_HIERARCHY.get(true_label, [])
    if pred_label in related:
        return PARTIAL_CREDIT
    return 0.0


class AgentGrader:
    """Grade an agent on a specific task difficulty"""

    def __init__(self, task_difficulty: str, num_episodes: int = 3):
        self.task_difficulty = task_difficulty
        self.num_episodes = num_episodes
        self.env = DocumentClassificationEnv(task_difficulty=task_difficulty, seed=42)

    def grade(self, agent_fn) -> Dict:
        """
        Run agent through multiple episodes and return score 0.0-1.0.

        Args:
            agent_fn: callable(observation) -> action (int)

        Returns:
            dict with score and detailed metrics
        """
        all_scores = []
        all_accuracies = []
        all_partial_credits = []

        for ep in range(self.num_episodes):
            obs, _ = self.env.reset(seed=ep * 777)
            ep_true = []
            ep_pred = []
            ep_rewards = []
            terminated = False

            while not terminated:
                action = agent_fn(obs)
                obs, reward, terminated, _, info = self.env.step(action)
                ep_true.append(info.get("true_category", ""))
                pred_cat = info.get("predicted_category", "")
                ep_pred.append(pred_cat)
                ep_rewards.append(reward)

            # Exact accuracy
            exact = sum(t == p for t, p in zip(ep_true, ep_pred)) / len(ep_true)

            # Hierarchical accuracy (with partial credit)
            hier_scores = [hierarchical_accuracy(t, p) for t, p in zip(ep_true, ep_pred)]
            hier_acc = np.mean(hier_scores)

            # Partial credit ratio
            partial = sum(1 for s in hier_scores if 0 < s < 1) / len(hier_scores)

            all_accuracies.append(exact)
            all_scores.append(hier_acc)
            all_partial_credits.append(partial)

        avg_accuracy = np.mean(all_accuracies)
        avg_hier = np.mean(all_scores)
        avg_partial = np.mean(all_partial_credits)

        # Compute final score based on difficulty
        metrics = {
            "accuracy": avg_accuracy,
            "hierarchical_accuracy": avg_hier,
            "partial_credit_rate": avg_partial,
            "average_processing_time_ms": 50,  # placeholder
        }

        score = self._compute_score(metrics)

        return {
            "score": score,
            "accuracy": avg_accuracy,
            "hierarchical_accuracy": avg_hier,
            "partial_credit_rate": avg_partial,
            "task_difficulty": self.task_difficulty,
            "num_episodes": self.num_episodes,
        }

    def _compute_score(self, metrics: Dict) -> float:
        accuracy = metrics["accuracy"]
        hier_accuracy = metrics["hierarchical_accuracy"]
        avg_time = metrics.get("average_processing_time_ms", 100)

        if self.task_difficulty == "easy":
            time_bonus = 0.1 if avg_time < 100 else 0.0
            # Easy: mostly exact accuracy
            score = 0.85 * accuracy + 0.05 * hier_accuracy + 0.1 * time_bonus

        elif self.task_difficulty == "medium":
            time_bonus = 0.0
            if avg_time < 200:
                time_bonus = 0.15
            elif avg_time < 500:
                time_bonus = 0.10
            # Medium: mix of exact + hierarchical
            score = 0.70 * accuracy + 0.15 * hier_accuracy + 0.15 * time_bonus

        else:  # hard / extreme
            time_bonus = 0.0
            if avg_time < 100:
                time_bonus = 0.25
            elif avg_time < 300:
                time_bonus = 0.15
            elif avg_time < 500:
                time_bonus = 0.05
            # Hard: hierarchical accuracy matters more (22+ categories)
            score = 0.50 * accuracy + 0.30 * hier_accuracy + 0.20 * time_bonus

        return min(1.0, max(0.0, score))


def grade_baseline(task_difficulty: str) -> float:
    """Quick helper to grade and return score"""
    from baseline_inference import TFIDFAgent

    agent = TFIDFAgent(task_difficulty)
    grader = AgentGrader(task_difficulty)

    def agent_fn(obs):
        return agent.predict(obs["content"])

    result = grader.grade(agent_fn)
    return result["score"]


if __name__ == "__main__":
    print("Grading baseline agent with hierarchical partial credit...\n")
    for difficulty in ["easy", "medium", "hard"]:
        score = grade_baseline(difficulty)
        print(f"  {difficulty.upper()} Score: {score:.4f}")
