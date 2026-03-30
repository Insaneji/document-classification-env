"""
Generate fake documents for training.
Templates for each category, then we mess with them a bit to make variations.
"""

import numpy as np
import random
from typing import Tuple, List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer


def get_task_config(difficulty: str) -> Dict:
    """Just return the config for a difficulty level"""
    configs = {
        "easy": {
            "num_documents": 100,
            "num_categories": 5,
            "feature_dim": 100,
            "max_word_count": 200,
            "time_limit": None,  # Can take as long as you want
        },
        "medium": {
            "num_documents": 500,
            "num_categories": 10,
            "feature_dim": 100,
            "max_word_count": 500,
            "time_limit": 2.0,  # 2 seconds per doc
        },
        "hard": {
            "num_documents": 1000,
            "num_categories": 20,
            "feature_dim": 100,
            "max_word_count": 1000,
            "time_limit": 1.0,  # 1 second, get moving
        }
    }
    return configs[difficulty]


class TaskDataGenerator:
    """Generate synthetic documents for the task"""
    
    DOCUMENT_TEMPLATES = {
        # Basic category - just regular stuff
        "General": [
            "This is a general inquiry about our services.",
            "I would like to know more about your company.",
            "Could you provide general information?",
            "General question regarding policies.",
        ],
        # Billing stuff
        "Billing": [
            "My invoice shows an incorrect amount. Please review.",
            "I need to update my billing address.",
            "What is the status of my payment?",
            "I have questions about my bill.",
        ],
        # People complaining about being overcharged
        "Billing-Dispute": [
            "I was overcharged on my last invoice.",
            "This charge should not appear on my account.",
            "I dispute this billing error.",
            "The pricing on this invoice is incorrect.",
        ],
        # Refund requests
        "Billing-Refund": [
            "I would like to request a refund.",
            "Please process a refund for my order.",
            "When will my refund be processed?",
            "I need to return and get refunded.",
        ],
        # Basic support
        "Support": [
            "I need technical assistance with your product.",
            "How can I contact customer support?",
            "I have a question about your services.",
            "Can you help me with this issue?",
        ],
        # EVERYTHING'S ON FIRE type support
        "Support-Urgent": [
            "URGENT: My system is down and I need immediate help!",
            "CRITICAL: This is affecting production!",
            "Emergency support needed immediately!",
            "This is urgent and needs immediate attention!",
        ],
        # Normal support that can wait
        "Support-Normal": [
            "I am having trouble with a basic feature.",
            "Can you walk me through the process?",
            "I need help understanding this feature.",
            "How do I use this functionality?",
        ],
        # Technical stuff
        "Technical": [
            "I am experiencing a technical issue with your software.",
            "There seems to be a bug in the system.",
            "Technical support needed for integration.",
            "I need help with API documentation.",
        ],
        # Code broken
        "Technical-Bug": [
            "I found a bug in the system that crashes the app.",
            "The system throws an error when I try to login.",
            "Error: Function X is not working correctly.",
            "Bug report: System crashes on this operation.",
        ],
        # New features people want
        "Technical-Feature": [
            "I would like to request a new feature.",
            "Can you add functionality for X?",
            "Feature request: Please add support for this.",
            "Would it be possible to implement this capability?",
        ],
        # HR stuff
        "HR": [
            "I have an HR-related question.",
            "I need assistance with HR matters.",
            "Can you help with HR policies?",
            "I have a question about HR.",
        ],
        # Money questions
        "HR-Payroll": [
            "I have a question about my paycheck.",
            "My salary appears to be incorrect.",
            "When is the next payroll cycle?",
            "Can you explain my payroll deductions?",
        ],
        # Health insurance etc
        "HR-Benefits": [
            "I need to enroll in health benefits.",
            "Can I update my benefits information?",
            "What benefits am I eligible for?",
            "I have questions about the benefits plan.",
        ],
        # Someone's upset
        "HR-Complaint": [
            "I would like to file a formal complaint.",
            "I need to report a workplace issue.",
            "This is a serious HR matter that needs attention.",
            "I am filing an official complaint.",
        ],
        # Legal stuff
        "Legal": [
            "I need legal advice regarding a contract.",
            "I have legal questions about terms.",
            "Can you help with legal documentation?",
            "I need assistance with legal matters.",
        ],
        # Contracts
        "Legal-Contract": [
            "I need review of this contract before signing.",
            "The contract terms need clarification.",
            "I have questions about contract obligations.",
            "Can you explain these contract provisions?",
        ],
        # Compliance - boring but important
        "Legal-Compliance": [
            "We need to ensure compliance with regulations.",
            "I have questions about legal compliance.",
            "Are we meeting compliance requirements?",
            "I need guidance on regulatory compliance.",
        ],
        # Executive stuff
        "Executive": [
            "I would like to schedule a meeting with management.",
            "This is an executive-level inquiry.",
            "Can I speak with a senior manager?",
            "I need executive attention for this matter.",
        ],
        # Big picture strategy
        "Executive-Strategic": [
            "Strategic partnership opportunity for discussion.",
            "I would like to discuss corporate strategy.",
            "This requires strategic-level decision making.",
            "Executive strategic initiative to discuss.",
        ],
        # Finance
        "Finance": [
            "I have questions about financial matters.",
            "Can you provide financial statements?",
            "I need to review financial records.",
            "Financial assistance or information needed.",
        ],
        # Marketing
        "Marketing": [
            "I am interested in marketing partnerships.",
            "Can we discuss marketing opportunities?",
            "I have a marketing proposal for you.",
            "Marketing collaboration inquiry.",
        ],
        # Operations
        "Operations": [
            "I have operational questions.",
            "Operations scheduling needs adjustment.",
            "Operational efficiency improvement proposal.",
            "I need to discuss operational procedures.",
        ],
    }
    
    def __init__(self, difficulty: str, seed: int = None):
        self.difficulty = difficulty
        self.config = get_task_config(difficulty)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Generate category list based on difficulty
        if difficulty == "easy":
            self.categories = ["General", "Billing", "Support", "Technical", "HR"]
        elif difficulty == "medium":
            self.categories = [
                "General", "Billing", "Billing-Dispute", "Support",
                "Technical", "Technical-Bug", "HR-Payroll", "HR-Benefits",
                "Legal", "Executive"
            ]
        else:  # hard
            self.categories = list(self.DOCUMENT_TEMPLATES.keys())
        
        # Initialize TF-IDF vectorizer
        all_docs = []
        for templates in self.DOCUMENT_TEMPLATES.values():
            all_docs.extend(templates)
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.config["feature_dim"],
            lowercase=True,
            stop_words='english'
        )
        self.vectorizer.fit(all_docs)
    
    def generate_task_data(self) -> Tuple[List[Dict], np.ndarray]:
        """
        Create a batch of documents.
        Pick random categories, grab templates, add some noise, extract features.
        """
        documents = []
        labels = []
        
        num_docs = self.config["num_documents"]
        
        for i in range(num_docs):
            # Pick a random category for this doc
            category = random.choice(self.categories)
            category_idx = self.categories.index(category)
            
            # Get a template from this category
            template = random.choice(self.DOCUMENT_TEMPLATES[category])
            
            # Add some variation so it's not too repetitive
            variations = self._add_variation(template)
            
            # Extract features (TF-IDF)
            features = self.vectorizer.transform([variations]).toarray()[0]
            
            # Normalize to [-1, 1] range
            features = 2 * features / (np.max(features) + 1e-8) - 1
            
            # Create document dict
            doc = {
                "id": f"doc_{i:06d}",
                "content": variations,
                "word_count": len(variations.split()),
                "has_urgency_markers": self._has_urgency(variations),
                "features": features.tolist(),
                "true_category": category,
            }
            
            documents.append(doc)
            labels.append(category_idx)
        
        return documents, np.array(labels)
    
    def _add_variation(self, template: str) -> str:
        """Add some noise/variation to make it feel less templated"""
        words = template.split()
        
        # Sometimes add filler words
        if random.random() < 0.3:
            filler = random.choice([
                "I think", "Regarding", "In addition", "Furthermore",
                "Also", "Additionally", "Moreover", "Please note"
            ])
            words.insert(random.randint(0, len(words)), filler)
        
        # Maybe add some detail
        if random.random() < 0.2:
            details = random.choice([
                " regarding order #12345.",
                " It's urgent.",
                " This needs immediate attention.",
                " Thank you for your help.",
                " Please let me know ASAP.",
            ])
            words.append(details)
        
        return " ".join(words)
    
    def _has_urgency(self, text: str) -> bool:
        """Check if text sounds urgent/important"""
        urgency_words = ["urgent", "emergency", "critical", "immediate", "asap", "urgent!"]
        text_lower = text.lower()
        return any(word in text_lower for word in urgency_words)


# For testing
if __name__ == "__main__":
    # Test data generation
    for difficulty in ["easy", "medium", "hard"]:
        gen = TaskDataGenerator(difficulty, seed=42)
        docs, labels = gen.generate_task_data()
        print(f"\n{difficulty.upper()} Task:")
        print(f"  Documents: {len(docs)}")
        print(f"  Categories: {len(set(labels))}")
        print(f"  Sample doc: {docs[0]['id']}")
        print(f"  Sample content: {docs[0]['content'][:50]}...")
        print(f"  Features shape: {len(docs[0]['features'])}")
