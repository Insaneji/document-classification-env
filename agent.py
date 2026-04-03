import pickle
import os
from datetime import datetime

CATEGORY_NAMES = {
    "easy":   {0:"General", 1:"Billing", 2:"Support", 3:"Technical", 4:"HR"},
    "medium": {0:"General", 1:"Billing", 2:"Support", 3:"Technical", 4:"HR",
               5:"Legal", 6:"Sales", 7:"Marketing", 8:"Operations", 9:"Complaints"},
}

CATEGORY_OVERRIDE = {
    "Technical":  ["crash", "crashes", "bug", "error", "not working", "broken", "upload fail",
                   "application", "software", "login fail", "404", "500", "exception",
                   "loading", "slow", "freeze", "hang", "install", "update failed", "glitch"],
    "Billing":    ["invoice", "charge", "charged", "payment", "refund", "subscription", "billing",
                   "amount", "price", "cost", "fee", "receipt", "transaction", "overcharged"],
    "HR":         ["leave", "salary", "policy", "maternity", "paternity", "holiday", "resignation",
                   "offer letter", "payslip", "appraisal", "human resources", "benefits"],
    "Complaints": ["complaint", "unacceptable", "disgusted", "terrible", "worst", "rude", "awful",
                   "disappointed", "frustrated", "escalate"],
    "Legal":      ["legal", "lawsuit", "compliance", "gdpr", "contract", "terms", "court",
                   "attorney", "lawyer", "liability", "copyright", "trademark"],
}

PRIORITY_RULES = {
    "urgent": ["urgent", "asap", "immediately", "critical", "emergency", "not working",
               "broken", "down", "error", "failed", "crashes", "crash"],
    "high":   ["incorrect", "wrong", "missing", "cannot", "can't", "unable",
               "problem", "issue", "complaint"],
    "medium": ["review", "check", "update", "change", "request", "need", "want", "please"],
    "low":    ["inquiry", "question", "info", "information", "general", "curious", "wondering"]
}

DEPARTMENT_MAP = {
    "Billing":    {"team": "Billing & Finance",   "email": "billing@company.com",    "sla": "24h"},
    "Technical":  {"team": "Tech Support",         "email": "techsupport@company.com","sla": "4h"},
    "HR":         {"team": "Human Resources",      "email": "hr@company.com",         "sla": "48h"},
    "Support":    {"team": "Customer Support",     "email": "support@company.com",    "sla": "12h"},
    "General":    {"team": "General Enquiries",    "email": "hello@company.com",      "sla": "48h"},
    "Legal":      {"team": "Legal & Compliance",   "email": "legal@company.com",      "sla": "72h"},
    "Sales":      {"team": "Sales",                "email": "sales@company.com",      "sla": "6h"},
    "Marketing":  {"team": "Marketing",            "email": "marketing@company.com",  "sla": "48h"},
    "Operations": {"team": "Operations",           "email": "ops@company.com",        "sla": "24h"},
    "Complaints": {"team": "Customer Relations",   "email": "complaints@company.com", "sla": "8h"},
}

REPLY_TEMPLATES = {
    "Billing": """Dear Customer,

Thank you for contacting us regarding your billing inquiry.

We have received your request and our Billing & Finance team will review your account details within 24 hours.

What we'll do:
  - Review your invoice/payment details
  - Verify the transaction in question
  - Send you a detailed resolution

Reference ID: {ref_id}
Priority: {priority}
Expected resolution: {sla}

If urgent, call: +1-800-BILLING

Best regards,
Billing & Finance Team""",

    "Technical": """Dear Customer,

Thank you for reporting this technical issue.

Our Tech Support team has been notified and will respond within 4 hours.

Immediate steps you can try:
  - Clear browser cache and retry
  - Check our status page: status.company.com
  - Restart the application

Reference ID: {ref_id}
Priority: {priority}
Expected resolution: {sla}

For critical issues: techsupport@company.com

Best regards,
Tech Support Team""",

    "HR": """Dear Team Member,

Thank you for reaching out to Human Resources.

Your query has been received and will be handled with full confidentiality.

Our HR team will respond within 48 hours regarding:
  - Policy clarifications
  - Benefits & compensation
  - Workplace concerns

Reference ID: {ref_id}
Priority: {priority}
Expected resolution: {sla}

Best regards,
Human Resources Team""",

    "Support": """Dear Customer,

Thank you for contacting Customer Support.

We have logged your request and a support agent will be in touch within 12 hours.

Reference ID: {ref_id}
Priority: {priority}
Expected resolution: {sla}

Track your ticket: support.company.com/track/{ref_id}

Best regards,
Customer Support Team""",

    "General": """Dear Customer,

Thank you for your inquiry.

We have received your message and will respond within 48 hours.

Reference ID: {ref_id}
Priority: {priority}

Best regards,
General Enquiries Team""",

    "Complaints": """Dear Customer,

We sincerely apologize for your experience.

Your complaint has been escalated to our Customer Relations team who will personally handle this within 8 hours.

Reference ID: {ref_id}
Priority: {priority}
Expected resolution: {sla}

We take all complaints seriously and are committed to resolving this promptly.

Best regards,
Customer Relations Team""",

    "Sales": """Dear Customer,

Thank you for your interest!

Our Sales team will reach out within 6 hours with a personalized proposal.

Reference ID: {ref_id}
Priority: {priority}

Best regards,
Sales Team""",

    "Legal": """Dear Customer,

Thank you for contacting our Legal & Compliance team.

Your matter will be reviewed by our legal team within 72 hours.

Reference ID: {ref_id}
Priority: {priority}
Expected resolution: {sla}

Best regards,
Legal & Compliance Team""",

    "Marketing": """Dear Customer,

Thank you for reaching out to Marketing.

We will respond within 48 hours.

Reference ID: {ref_id}
Priority: {priority}

Best regards,
Marketing Team""",

    "Operations": """Dear Customer,

Thank you for contacting Operations.

Your request has been logged and will be addressed within 24 hours.

Reference ID: {ref_id}
Priority: {priority}
Expected resolution: {sla}

Best regards,
Operations Team""",
}


class TicketAgent:
    def __init__(self, model_dir="."):
        self.models = {}
        self.model_dir = model_dir
        self._load_models()

    def _load_models(self):
        for d in ["easy", "medium", "hard"]:
            path = os.path.join(self.model_dir, f"model_{d}.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self.models[d] = pickle.load(f)

    def _to_name(self, c, difficulty):
        name_map = CATEGORY_NAMES.get(difficulty, {})
        try:
            return name_map.get(int(c), str(c))
        except:
            return str(c)

    def classify(self, text, difficulty="easy"):
        text_lower = text.lower()

        # Step 1 - Keyword override (hybrid rule check)
        for category, keywords in CATEGORY_OVERRIDE.items():
            if any(kw in text_lower for kw in keywords):
                if difficulty in self.models:
                    model = self.models[difficulty]
                    raw_classes = model.classes_
                    proba = model.predict_proba([text])[0]
                    scores = {
                        self._to_name(c, difficulty): round(float(p), 4)
                        for c, p in zip(raw_classes, proba)
                    }
                else:
                    scores = {category: 1.0}
                scores[category] = max(scores.get(category, 0), 0.85)
                return category, scores

        # Step 2 - Pure ML classification
        if difficulty not in self.models:
            return "General", {"General": 1.0}
        model = self.models[difficulty]
        raw_classes = model.classes_
        proba = model.predict_proba([text])[0]
        pred_raw = raw_classes[proba.argmax()]
        top_cat = self._to_name(pred_raw, difficulty)
        scores = {
            self._to_name(c, difficulty): round(float(p), 4)
            for c, p in zip(raw_classes, proba)
        }
        return top_cat, scores

    def get_priority(self, text):
        text_lower = text.lower()
        for priority, keywords in PRIORITY_RULES.items():
            if any(kw in text_lower for kw in keywords):
                return priority
        return "low"

    def route(self, category):
        return DEPARTMENT_MAP.get(category, DEPARTMENT_MAP["General"])

    def generate_reply(self, category, priority, ref_id):
        dept = self.route(category)
        template = REPLY_TEMPLATES.get(category, REPLY_TEMPLATES["General"])
        return template.format(
            ref_id=ref_id,
            priority=priority.upper(),
            sla=dept["sla"]
        )

    def process_ticket(self, text, difficulty="easy", ticket_id=None):
        ref_id = ticket_id or f"TKT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        category, scores = self.classify(text, difficulty)
        priority = self.get_priority(text)
        dept = self.route(category)
        reply = self.generate_reply(category, priority, ref_id)
        top3 = sorted(scores.items(), key=lambda x: -x[1])[:3]

        return {
            "ref_id":     ref_id,
            "category":   category,
            "priority":   priority,
            "department": dept["team"],
            "email":      dept["email"],
            "sla":        dept["sla"],
            "confidence": scores.get(category, 0.0),
            "top3":       top3,
            "reply":      reply,
            "timestamp":  datetime.now().isoformat()
        }
