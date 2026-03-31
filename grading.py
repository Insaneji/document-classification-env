"""
Grade agents on how well they classify documents.
Basically runs them through a test set and scores them.
"""

import numpy as np
from typing import Dict, List, Tuple
from environment import DocumentClassificationEnv
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import time


class AgentGrader:
    """Run an agent and grade how well it does"""
    
    def __init__(self, task_difficulty: str):
        """Set up the grader for a specific difficulty"""
        self.task_difficulty = task_difficulty
        self.env = DocumentClassificationEnv(task_difficulty=task_difficulty, seed=42)
    
    def grade_agent(self, agent_policy, verbose: bool = False) -> float:
        """
        Run agent through test set and grade it.
        
        Args:
            agent_policy: Function that takes observation, returns action
            verbose: Print detailed results?
            
        Returns:
            score (0-1) and metrics dict
        """
        obs, _ = self.env.reset(seed=42)
        
        predictions = []
        ground_truths = []
        rewards = []
        processing_times = []
        episode_done = False
        step_count = 0
        start_time = time.time()
        
        while not episode_done and step_count < 5000:
            try:
                # Get agent's decision
                action = agent_policy(obs)
                
                # Clean up the action (sometimes agents return floats or bad indices)
                if not isinstance(action, (int, np.integer)):
                    action = int(action)
                action = np.clip(int(action), 0, self.env.num_categories - 1)
                
                step_start = time.time()
                obs, reward, episode_done, truncated, info = self.env.step(action)
                processing_times.append(time.time() - step_start)
                
                rewards.append(reward)
                predictions.append(action)
                ground_truths.append(info.get("true_category"))
                step_count += 1
                
            except Exception as e:
                if verbose:
                    print(f"Error: {e}")
                break
        
        total_time = time.time() - start_time
        
        # Calculate everything
        metrics = self._calculate_metrics(
            predictions, 
            ground_truths, 
            rewards,
            processing_times,
            total_time
        )
        
        # Get final score
        score = self._calculate_score(metrics)
        
        if verbose:
            self._print_results(metrics, score)
        
        return score, metrics
    
    def _calculate_metrics(self, 
                          predictions: List[int],
                          ground_truths: List[str],
                          rewards: List[float],
                          processing_times: List[float],
                          total_time: float) -> Dict:
        """Calculate performance metrics"""
        
        # Convert ground truth category names to indices
        env_categories = list(self.env.CATEGORY_MAPS[self.task_difficulty].values())
        ground_truth_indices = []
        for gt in ground_truths:
            try:
                idx = env_categories.index(gt)
                ground_truth_indices.append(idx)
            except ValueError:
                continue
        
        predictions = predictions[:len(ground_truth_indices)]
        
        accuracy = np.mean(np.array(predictions) == np.array(ground_truth_indices)) if len(predictions) > 0 else 0.0
        
        metrics = {
            "accuracy": accuracy,
            "total_documents": len(predictions),
            "correct_classifications": np.sum(np.array(predictions) == np.array(ground_truth_indices)),
            "average_reward": np.mean(rewards) if rewards else 0.0,
            "total_reward": np.sum(rewards) if rewards else 0.0,
            "average_processing_time_ms": np.mean(processing_times) * 1000 if processing_times else 0.0,
            "total_time_seconds": total_time,
        }
        
        # Add precision/recall if applicable
        if len(set(ground_truth_indices)) > 1 and len(set(predictions)) > 1:
            try:
                metrics["macro_precision"] = precision_score(
                    ground_truth_indices, predictions, 
                    average='macro', zero_division=0
                )
                metrics["macro_recall"] = recall_score(
                    ground_truth_indices, predictions,
                    average='macro', zero_division=0
                )
                metrics["macro_f1"] = f1_score(
                    ground_truth_indices, predictions,
                    average='macro', zero_division=0
                )
            except:
                pass
        
        return metrics
    
    def _calculate_score(self, metrics: Dict) -> float:
        """
        Calculate final score.
        Easy = just accuracy
        Medium = 80% accuracy + 20% speed bonus
        Hard = 75% accuracy + 25% speed bonus
        """
        accuracy = metrics["accuracy"]
        avg_reward = metrics["average_reward"]
        
        if self.task_difficulty == "easy":
            # Easy: just care about being right
            score = accuracy
        
        elif self.task_difficulty == "medium":
            # Medium: care about both
            time_bonus = 0.0
            avg_time = metrics["average_processing_time_ms"]
            if avg_time < 200:
                time_bonus = 0.15
            elif avg_time < 500:
                time_bonus = 0.05
            
            score = 0.8 * accuracy + 0.2 * time_bonus
        
        else:  # hard
            # Hard: need speed to compensate for lower accuracy
            time_bonus = 0.0
            avg_time = metrics["average_processing_time_ms"]
            if avg_time < 100:
                time_bonus = 0.25
            elif avg_time < 300:
                time_bonus = 0.1
            
            score = 0.75 * accuracy + 0.25 * time_bonus
        
        # Keep it in [0, 1]
        score = np.clip(score, 0.0, 1.0)
        
        return score
    
    def _print_results(self, metrics: Dict, score: float):
        """Pretty print the results"""
        print(f"\n{'='*60}")
        print(f"Task: {self.task_difficulty.upper()}")
        print(f"{'='*60}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Got it right: {metrics['correct_classifications']}/{metrics['total_documents']}")
        print(f"Avg Reward: {metrics['average_reward']:.4f}")
        print(f"Total Reward: {metrics['total_reward']:.4f}")
        print(f"Avg Processing Time: {metrics['average_processing_time_ms']:.2f}ms")
        print(f"Total Time: {metrics['total_time_seconds']:.2f}s")
        
        if "macro_precision" in metrics:
            print(f"Precision: {metrics['macro_precision']:.4f}")
            print(f"Recall: {metrics['macro_recall']:.4f}")
            print(f"F1: {metrics['macro_f1']:.4f}")
        
        print(f"\nFinal Score: {score:.4f}")
        print(f"{'='*60}\n")


class BaselineAgent:
    """Simple baseline - just uses keyword matching"""
    
    def __init__(self, difficulty: str):
        self.difficulty = difficulty
    
    def decide(self, observation):
        """Look for keywords in the document, make a guess"""
        content = observation.get("content", "").lower()
        features = observation.get("features", [])
        
        if self.difficulty == "easy":
            # Easy mode - just look for obvious keywords
            if "billing" in content or "bill" in content or "invoice" in content:
                return 1  # Billing
            elif "support" in content or "help" in content or "issue" in content:
                return 2  # Support
            elif "technical" in content or "bug" in content or "error" in content:
                return 3  # Technical
            elif "hr" in content or "payroll" in content or "benefits" in content:
                return 4  # HR
            return 0  # Default: General
        
        elif self.difficulty == "medium":
            # Medium - more keywords to look for
            if "dispute" in content or "overcharg" in content:
                return 2  # Billing-Dispute
            elif "urgent" in content or "critical" in content or "emergency" in content:
                return 3  # Support (using urgent as proxy)
            elif "bug" in content or "error" in content or "crash" in content:
                return 5  # Technical-Bug
            elif "payroll" in content or "salary" in content or "payment" in content:
                return 6  # HR-Payroll
            elif "benefits" in content or "health" in content or "enroll" in content:
                return 7  # HR-Benefits
            elif "billing" in content or "invoice" in content:
                return 1  # Billing
            elif "technical" in content:
                return 4  # Technical
            elif "legal" in content or "contract" in content:
                return 8  # Legal
            elif "executive" in content or "manager" in content or "strategy" in content:
                return 9  # Executive
            return 0  # General
        
        else:  # hard
            # Hard mode - need more keywords and combinations
            if "urgent" in content or "critical" in content:
                return 4  # Support-Urgent
            elif "dispute" in content or "overcharge" in content:
                return 2  # Billing-Dispute
            elif "refund" in content:
                return 3  # Billing-Refund
            elif "bug" in content or "crash" in content:
                return 7  # Technical-Bug
            elif "feature" in content or "request" in content:
                return 8  # Technical-Feature
            elif "payroll" in content:
                return 9  # HR-Payroll
            elif "benefits" in content:
                return 10  # HR-Benefits
            elif "complaint" in content or "complain" in content:
                return 11  # HR-Complaint
            elif "contract" in content:
                return 13  # Legal-Contract
            elif "compliance" in content:
                return 14  # Legal-Compliance
            elif "strategic" in content or "strategy" in content:
                return 16  # Executive-Strategic
            elif "billing" in content or "invoice" in content:
                return 1  # Billing
            elif "support" in content:
                return 5  # Support-Normal
            elif "technical" in content:
                return 6  # Technical
            elif "legal" in content:
                return 12  # Legal
            elif "finance" in content:
                return 17  # Finance
            elif "marketing" in content:
                return 18  # Marketing
            elif "operations" in content:
                return 19  # Operations
            elif "executive" in content:
                return 15  # Executive
            return 0  # General
