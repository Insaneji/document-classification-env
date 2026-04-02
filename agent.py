import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

CATEGORY_TEMPLATES = {
    "General": ["general inquiry about services", "question about company", "general information request", "policies question"],
    "Billing": ["billing invoice payment", "account charges subscription", "payment status bill", "billing address update"],
    "Billing-Dispute": ["overcharged incorrect invoice", "dispute charge billing error", "wrong amount charged", "charge should not appear"],
    "Billing-Refund": ["request refund money back", "process refund return", "refund my payment", "need refund overpayment"],
    "Support": ["technical assistance help", "customer support question", "help with issue product", "contact support"],
    "Support-Urgent": ["urgent critical system down", "emergency production outage", "immediate help critical", "everything broken urgent"],
    "Support-Normal": ["basic feature help", "how to use product", "walk me through steps", "understanding functionality"],
    "Technical": ["technical issue software", "API integration documentation", "system technical problem", "software technical help"],
    "Technical-Bug": ["bug crash error exception", "software not working defect", "application crashing broken", "bug report error found"],
    "Technical-Feature": ["feature request enhancement", "new feature suggestion", "add functionality improvement", "product feature idea"],
    "HR": ["HR department human resources", "employee workforce staff", "HR policy question", "human resources inquiry"],
    "HR-Payroll": ["paycheck salary payroll", "wage compensation pay stub", "payroll department salary", "incorrect pay payroll"],
    "HR-Benefits": ["benefits insurance enrollment", "health dental coverage", "employee benefits package", "401k vacation benefits"],
    "HR-Complaint": ["complaint harassment workplace", "misconduct grievance hostile", "discrimination complaint report", "hostile environment complaint"],
    "Legal": ["legal assistance attorney", "legal department advice", "law contract legal question", "legal liability counsel"],
    "Legal-Contract": ["contract review agreement", "signing contract terms clause", "contract dispute legal", "vendor agreement review"],
    "Legal-Compliance": ["compliance regulation GDPR", "regulatory audit requirement", "HIPAA compliance legal", "regulatory requirement compliance"],
    "Executive": ["executive management leadership", "senior team escalation", "executive inquiry question", "management leadership team"],
    "Executive-Strategic": ["strategic planning business", "executive strategy roadmap", "business vision direction", "strategic initiative planning"],
    "Finance": ["finance budget accounting", "revenue fiscal financial", "financial report budget", "accounting finance question"],
    "Marketing": ["marketing campaign promotion", "advertising brand marketing", "campaign strategy promotion", "marketing question brand"],
    "Operations": ["operations logistics workflow", "supply chain operational", "process management operations", "operational question logistics"],
}

class SmartClassificationAgent:
    def __init__(self, difficulty):
        self.difficulty = difficulty
        self.vectorizer = None
        self.template_matrix = None
        self.template_labels = []
        self.categories = []
        self._trained = False

    def train(self, categories):
        self.categories = categories
        texts, labels = [], []
        for cat in categories:
            if cat in CATEGORY_TEMPLATES:
                for t in CATEGORY_TEMPLATES[cat]:
                    texts.append(t)
                    labels.append(cat)
            else:
                texts.append(cat.lower().replace("-", " "))
                labels.append(cat)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, max_features=10000)
        self.template_matrix = self.vectorizer.fit_transform(texts)
        self.template_labels = labels
        self._trained = True

    def predict(self, obs):
        if not self._trained:
            return 0, 0.0

        content = obs.get("content", "").lower()
        features = np.array(obs.get("features", []))
        word_count = obs.get("word_count", 0)
        has_urgency = obs.get("has_urgency_markers", False)

        # TF-IDF content similarity
        content_vec = self.vectorizer.transform([content])
        sims = cosine_similarity(content_vec, self.template_matrix)[0]
        best_idx = int(np.argmax(sims))
        best_cat = self.template_labels[best_idx]
        content_conf = float(sims[best_idx])

        # Feature vector signal
        feat_cat = None
        feat_conf = 0.0
        if len(features) > 0 and np.max(features) > -0.5:
            valid = [(i, v) for i, v in enumerate(features) if v > -0.5]
            if valid:
                best_feat_idx = max(valid, key=lambda x: x[1])[0]
                if best_feat_idx < len(self.categories):
                    feat_cat = self.categories[best_feat_idx]
                    feat_conf = float(features[best_feat_idx])

        # Urgency boost
        if has_urgency and "Support" in best_cat and "Urgent" not in best_cat:
            if "Support-Urgent" in self.categories:
                best_cat = "Support-Urgent"
                content_conf = min(content_conf + 0.2, 1.0)

        # Combine signals
        final_cat = best_cat
        if feat_cat and feat_cat in self.categories:
            if feat_conf > content_conf * 0.8:
                final_cat = feat_cat

        try:
            action = self.categories.index(final_cat)
            confidence = max(content_conf, feat_conf)
        except ValueError:
            action = 0
            confidence = 0.1

        return action, confidence


def run_episode(difficulty, seed=42):
    from environment import DocumentClassificationEnv
    env = DocumentClassificationEnv(difficulty)
    cats_map = env.CATEGORY_MAPS[difficulty]
    cats_list = [cats_map[i] for i in sorted(cats_map.keys())]

    agent = SmartClassificationAgent(difficulty)
    agent.train(cats_list)

    obs, info = env.reset(seed=seed)
    results = []
    total_reward = 0
    step = 0

    while True:
        t0 = time.time()
        action, confidence = agent.predict(obs)
        elapsed_ms = (time.time() - t0) * 1000

        obs_next, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        results.append({
            "step": step,
            "predicted": cats_map.get(action, "Unknown"),
            "reward": reward,
            "confidence": confidence,
            "time_ms": elapsed_ms,
        })

        if done or truncated:
            break
        obs = obs_next

    accuracy = total_reward / step if step > 0 else 0
    return results, accuracy, info


if __name__ == "__main__":
    for diff in ["easy", "medium", "hard"]:
        results, acc, info = run_episode(diff)
        print(f"{diff.upper()}: accuracy={acc:.4f}, steps={len(results)}")
