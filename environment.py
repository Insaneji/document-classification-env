import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import random
from tasks import TaskDataGenerator, get_task_config


class DocumentClassificationEnv(gym.Env):
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
            2: "Billing-Dispute",
            3: "Support",
            4: "Technical",
            5: "Technical-Bug",
            6: "HR-Payroll",
            7: "HR-Benefits",
            8: "Legal",
            9: "Executive"
        },
        "hard": {
            0: "General", 1: "Billing", 2: "Billing-Dispute", 3: "Billing-Refund",
            4: "Support-Urgent", 5: "Support-Normal", 6: "Technical", 7: "Technical-Bug",
            8: "Technical-Feature", 9: "HR-Payroll", 10: "HR-Benefits", 11: "HR-Complaint",
            12: "Legal", 13: "Legal-Contract", 14: "Legal-Compliance", 15: "Executive",
            16: "Executive-Strategic", 17: "Finance", 18: "Marketing", 19: "Operations",
            20: "HR", 21: "Support"
        }
    }
    
    def __init__(self, task_difficulty="easy", seed=None):
        super().__init__()
        self.task_difficulty = task_difficulty
        if task_difficulty not in ["easy", "medium", "hard"]:
            raise ValueError(f"Pick easy, medium, or hard. Got: {task_difficulty}")
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.task_config = get_task_config(task_difficulty)
        self.num_categories = self.task_config["num_categories"]
        self.feature_dim = self.task_config["feature_dim"]
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
        self.data_generator = TaskDataGenerator(task_difficulty, seed)
        self.documents = None
        self.labels = None
        self.current_document_index = 0
        self.episode_rewards = []
        self.correct = 0
        self.total = 0
        self.episode_times = []
        self.start_time = None
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.documents, self.labels = self.data_generator.generate_task_data()
        self.current_document_index = 0
        self.episode_rewards = []
        self.correct = 0
        self.total = 0
        self.episode_times = []
        self.start_time = time.time()
        return self._obs(), {}
    
    def step(self, action):
        t0 = time.time()
        if self.current_document_index >= len(self.documents):
            return self._obs(), 0.0, True, False, {"episode_complete": True}
        
        true_label = self.labels[self.current_document_index]
        is_correct = action == true_label
        reward = 1.0 if is_correct else -0.4
        
        self.episode_rewards.append(reward)
        if is_correct:
            self.correct += 1
        self.total += 1
        self.episode_times.append(time.time() - t0)
        
        self.current_document_index += 1
        done = self.current_document_index >= len(self.documents)
        obs = self._obs()
        
        info = {
            "is_correct": is_correct,
            "true_category": self.CATEGORY_MAPS[self.task_difficulty][true_label],
            "predicted_category": self.CATEGORY_MAPS[self.task_difficulty][action],
            "processing_time_ms": (time.time() - t0) * 1000,
            "episode_accuracy": self.correct / self.total if self.total > 0 else 0.0,
        }
        if done:
            info["episode_summary"] = self._summary()
        return obs, reward, done, False, info
    
    def state(self):
        return {
            "current_observation": self._obs(),
            "document_index": self.current_document_index,
            "total_documents": len(self.documents) if self.documents else 0,
            "total_reward": sum(self.episode_rewards),
            "accuracy": self.correct / self.total if self.total > 0 else 0.0,
            "avg_time_ms": np.mean(self.episode_times) * 1000 if self.episode_times else 0.0,
            "difficulty": self.task_difficulty,
        }
    
    def _obs(self):
        if self.current_document_index >= len(self.documents):
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
    
    def _summary(self):
        total_reward = sum(self.episode_rewards)
        avg_reward = total_reward / len(self.episode_rewards) if self.episode_rewards else 0.0
        acc = self.correct / self.total if self.total > 0 else 0.0
        avg_time = np.mean(self.episode_times) * 1000 if self.episode_times else 0.0
        elapsed = time.time() - self.start_time
        return {
            "total_reward": total_reward,
            "avg_reward": avg_reward,
            "accuracy": acc,
            "total_classified": self.total,
            "avg_time_ms": avg_time,
            "total_time_s": elapsed,
        }
    
    def render(self):
        pass
    
    def close(self):
        pass


