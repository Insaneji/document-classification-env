"""
Document routing environment - teaches agents to classify and route docs
This is basically a mail sorter that learns what category each email belongs to
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Any
import time
import random
from tasks import TaskDataGenerator, get_task_config


class DocumentClassificationEnv(gym.Env):
    """
    Real environment for document classification.
    Agent sees documents, has to pick the right category, gets rewarded for correct picks.
    That's it. Simple but surprisingly hard.
    """
    
    metadata = {
        "render_modes": ["human"],
        "name": "DocumentClassification-v1"
    }
    
    # Different difficulty levels = different number of categories
    # Easy = 5, Medium = 10, Hard = 20 (gets gnarly fast)
    CATEGORY_MAPS = {
        "easy": {
            0: "General",
            1: "Billing", 
            2: "Support",
            3: "Technical",
            4: "HR"
        },
        "medium": {
            0: "General",
            1: "Billing",
            2: "Billing-Dispute",  # People complaining about charges
            3: "Support",
            4: "Technical",
            5: "Technical-Bug",    # Specific problems
            6: "HR-Payroll",
            7: "HR-Benefits",
            8: "Legal",
            9: "Executive"
        },
        "hard": {
            0: "General",
            1: "Billing",
            2: "Billing-Dispute", 
            3: "Billing-Refund",   # Money back requests
            4: "Support-Urgent",   # URGENT URGENT URGENT
            5: "Support-Normal",   # Can wait a bit
            6: "Technical",
            7: "Technical-Bug",    # Broken stuff
            8: "Technical-Feature",  # Want something new
            9: "HR-Payroll",
            10: "HR-Benefits",
            11: "HR-Complaint",    # Someone mad at someone else
            12: "Legal",
            13: "Legal-Contract",  # Signed agreements
            14: "Legal-Compliance",  # Regulatory stuff, boring
            15: "Executive",
            16: "Executive-Strategic",  # Big picture stuff
            17: "Finance",
            18: "Marketing",
            19: "Operations",
            20: "HR",
            21: "Support"
        }
    }
    
    def __init__(self, task_difficulty: str = "easy", seed: int = None):
        """
        Set up the environment.
        
        Args:
            task_difficulty: "easy", "medium", or "hard" - how mean we're gonna be
            seed: For reproducibility (always helpful)
        """
        super().__init__()
        
        self.task_difficulty = task_difficulty
        if task_difficulty not in ["easy", "medium", "hard"]:
            raise ValueError(f"Pick easy, medium, or hard. Got: {task_difficulty}")
        
        # Reproducibility is good, so set seeds if given
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Load config for this difficulty level
        self.task_config = get_task_config(task_difficulty)
        self.num_categories = self.task_config["num_categories"]
        self.feature_dim = self.task_config["feature_dim"]
        
        # Define what the agent can do and what it sees
        self.action_space = spaces.Discrete(self.num_categories)
        
        self.observation_space = spaces.Dict({
            "document_id": spaces.Text(max_length=20),
            "content": spaces.Text(max_length=10000),
            "word_count": spaces.Box(low=0, high=10000, shape=(1,), dtype=np.int32),
            "has_urgency_markers": spaces.MultiBinary(1),
            "features": spaces.Box(low=-1.0, high=1.0, shape=(self.feature_dim,), dtype=np.float32),
            "document_index": spaces.Box(low=0, high=10000, shape=(1,), dtype=np.int32),
            "total_documents": spaces.Box(low=0, high=10000, shape=(1,), dtype=np.int32),
        })
        
        # Initialize the data generator (creates fake documents)
        self.data_generator = TaskDataGenerator(task_difficulty, seed)
        self.documents = None
        self.labels = None
        self.current_document_index = 0
        self.episode_rewards = []
        self.episode_accuracy_count = 0
        self.episode_total_count = 0
        self.episode_times = []
        self.episode_start_time = None
        
    def reset(self, seed: int = None, options: Dict = None) -> Tuple[Dict, Dict]:
        """
        Reset for a new episode.
        This generates a new batch of documents to classify.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Generate documents for this episode
        self.documents, self.labels = self.data_generator.generate_task_data()
        self.current_document_index = 0
        self.episode_rewards = []
        self.episode_accuracy_count = 0
        self.episode_total_count = 0
        self.episode_times = []
        self.episode_start_time = time.time()
        
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step of the environment
        
        Args:
            action: The category index selected by the agent
            
        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        step_start_time = time.time()
        
        # Check if episode is done
        if self.current_document_index >= len(self.documents):
            terminated = True
            observation = self._get_observation()
            return observation, 0.0, terminated, False, {"episode_complete": True}
        
        # Get current document label
        true_label = self.labels[self.current_document_index]
        
        # Calculate reward
        is_correct = action == true_label
        reward = self._calculate_reward(
            is_correct, 
            action, 
            true_label,
            step_start_time
        )
        
        # Update episode statistics
        self.episode_rewards.append(reward)
        if is_correct:
            self.episode_accuracy_count += 1
        self.episode_total_count += 1
        self.episode_times.append(time.time() - step_start_time)
        
        # Move to next document
        self.current_document_index += 1
        
        # Check if episode is complete
        terminated = self.current_document_index >= len(self.documents)
        
        # Get next observation
        observation = self._get_observation()
        
        # Prepare info dict
        info = {
            "is_correct": is_correct,
            "true_category": self.CATEGORY_MAPS[self.task_difficulty][true_label],
            "predicted_category": self.CATEGORY_MAPS[self.task_difficulty][action],
            "processing_time_ms": (time.time() - step_start_time) * 1000,
            "episode_accuracy": self.episode_accuracy_count / self.episode_total_count if self.episode_total_count > 0 else 0.0,
        }
        
        if terminated:
            info["episode_summary"] = self._get_episode_summary()
        
        return observation, reward, terminated, False, info
    
    def state(self) -> Dict:
        """
        Get the current state of the environment
        
        Returns:
            dict: Environment state including observations, episode progress, and statistics
        """
        return {
            "current_observation": self._get_observation(),
            "current_document_index": self.current_document_index,
            "total_documents": len(self.documents) if self.documents is not None else 0,
            "episode_reward_total": sum(self.episode_rewards),
            "episode_accuracy": self.episode_accuracy_count / self.episode_total_count if self.episode_total_count > 0 else 0.0,
            "average_processing_time_ms": np.mean(self.episode_times) * 1000 if self.episode_times else 0.0,
            "task_difficulty": self.task_difficulty,
        }
    
    def _get_observation(self) -> Dict:
        """
        Get the current observation
        
        Returns:
            dict: Current observation containing document features and metadata
        """
        if self.current_document_index >= len(self.documents):
            # Return terminal observation
            return {
                "document_id": "",
                "content": "",
                "word_count": np.array([0], dtype=np.int32),
                "has_urgency_markers": np.array([0], dtype=np.int8),
                "features": np.zeros(self.feature_dim, dtype=np.float32),
                "document_index": np.array([self.current_document_index], dtype=np.int32),
                "total_documents": np.array([len(self.documents)], dtype=np.int32),
            }
        
        doc = self.documents[self.current_document_index]
        
        return {
            "document_id": doc["id"],
            "content": doc["content"],
            "word_count": np.array([doc["word_count"]], dtype=np.int32),
            "has_urgency_markers": np.array([int(doc["has_urgency_markers"])], dtype=np.int8),
            "features": np.array(doc["features"], dtype=np.float32),
            "document_index": np.array([self.current_document_index], dtype=np.int32),
            "total_documents": np.array([len(self.documents)], dtype=np.int32),
        }
    
    def _calculate_reward(self, is_correct: bool, action: int, true_label: int, step_start_time: float) -> float:
        """
        Calculate how much reward the agent gets.
        
        Simple formula:
        - Correct = +1.0 (nice!)
        - Wrong = -0.5 (oops)
        - Fast = bonus (0.1 to 0.2 depending on difficulty)
        """
        processing_time = time.time() - step_start_time
        
        # Give reward for being right
        accuracy_bonus = 1.0 if is_correct else -0.5
        
        # Give bonus for being fast (different per difficulty)
        speed_bonus = 0.0
        if self.task_difficulty == "easy":
            # Easy - reward if under 100ms
            speed_bonus = 0.1 if processing_time < 0.1 else 0.0
        elif self.task_difficulty == "medium":
            # Medium - 2 second limit, graduated bonus
            if processing_time < 0.2:
                speed_bonus = 0.15
            elif processing_time < 0.5:
                speed_bonus = 0.1
        else:  # hard
            # Hard - 1 second limit, really reward speed
            if processing_time < 0.1:
                speed_bonus = 0.2
            elif processing_time < 0.3:
                speed_bonus = 0.1
        
        # Total it up
        total_reward = accuracy_bonus + speed_bonus
        
        return total_reward
    
    def _get_episode_summary(self) -> Dict:
        """
        Get a summary of the completed episode
        
        Returns:
            dict: Episode statistics and performance metrics
        """
        total_reward = sum(self.episode_rewards)
        avg_reward = total_reward / len(self.episode_rewards) if self.episode_rewards else 0.0
        accuracy = self.episode_accuracy_count / self.episode_total_count if self.episode_total_count > 0 else 0.0
        avg_time_ms = np.mean(self.episode_times) * 1000 if self.episode_times else 0.0
        
        return {
            "total_reward": total_reward,
            "average_reward": avg_reward,
            "accuracy": accuracy,
            "total_documents_classified": self.episode_total_count,
            "average_processing_time_ms": avg_time_ms,
            "total_episode_time_seconds": time.time() - self.episode_start_time,
        }
    
    def render(self):
        """Render the environment (optional)"""
        pass
    
    def close(self):
        """Close the environment"""
        pass


# For testing/debugging
if __name__ == "__main__":
    env = DocumentClassificationEnv(task_difficulty="easy")
    obs, info = env.reset()
    
    print(f"Initial observation keys: {obs.keys()}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Take a few steps
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Reward: {reward:.3f}, Accuracy so far: {info['episode_accuracy']:.3f}")
        if done:
            print(f"Episode complete: {info['episode_summary']}")
            break


