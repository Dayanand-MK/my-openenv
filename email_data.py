"""
Email dataset and task configuration.
All 20 emails are labelled with ground-truth metadata used by the graders.
"""

# ─────────────────────────────────────────────
#  Ground-truth email dataset
# ─────────────────────────────────────────────
# Each email carries:
#   category  – true label for Task 1
#   urgency   – 1 (low) … 5 (critical) for Task 2
#   target_id – which email is the complaint target for Task 3

EMAILS: list[dict] = [
    {
        "id": "e01",
        "subject": "URGENT: Payment failed – account suspended",
        "sender": "billing@acmecorp.com",
        "body": (
            "Your payment of $299 has failed for the third consecutive time. "
            "Your account has been suspended. Please update your payment method "
            "immediately to restore access. If you believe this is an error, "
            "contact our billing team within 24 hours."
        ),
        "timestamp": "2024-01-15T09:00:00Z",
        "category": "billing",
        "urgency": 5,
    },
    {
        "id": "e02",
        "subject": "Re: My order still hasn't arrived after 3 weeks",
        "sender": "angry.customer@gmail.com",
        "body": (
            "I placed order #78432 three weeks ago and it has not arrived. "
            "I have emailed twice with no response. This is completely unacceptable. "
            "I want a full refund immediately and an explanation for the delay. "
            "I will be filing a chargeback if I don't hear back within 24 hours."
        ),
        "timestamp": "2024-01-15T09:15:00Z",
        "category": "support",
        "urgency": 5,
    },
    {
        "id": "e03",
        "subject": "Congratulations! You've won a free iPhone 15",
        "sender": "noreply@prize-winner-2024.biz",
        "body": (
            "Dear valued customer, you have been selected as the lucky winner "
            "of a brand new iPhone 15! Click the link below to claim your prize. "
            "Offer expires in 24 hours. Act now!"
        ),
        "timestamp": "2024-01-15T09:20:00Z",
        "category": "spam",
        "urgency": 1,
    },
    {
        "id": "e04",
        "subject": "Server down – production outage",
        "sender": "devops@internalteam.com",
        "body": (
            "Production server is unreachable as of 09:18 UTC. "
            "All customer-facing services are affected. "
            "On-call engineer has been paged. ETA for resolution: 30 minutes. "
            "Please stand by for updates."
        ),
        "timestamp": "2024-01-15T09:18:00Z",
        "category": "urgent",
        "urgency": 5,
    },
    {
        "id": "e05",
        "subject": "Question about your refund policy",
        "sender": "customer123@yahoo.com",
        "body": (
            "Hi, I recently purchased your Pro plan but I'm not sure if it's the right "
            "fit for me. Can you explain your refund policy? "
            "I'm within the 30-day window. Thanks!"
        ),
        "timestamp": "2024-01-15T10:00:00Z",
        "category": "support",
        "urgency": 3,
    },
    {
        "id": "e06",
        "subject": "Invoice #INV-2024-0089 attached",
        "sender": "accounts@vendor.io",
        "body": (
            "Please find attached Invoice #INV-2024-0089 for $1,250 due on 2024-01-30. "
            "Payment via bank transfer or credit card. "
            "Contact us at billing@vendor.io with any queries."
        ),
        "timestamp": "2024-01-15T10:30:00Z",
        "category": "billing",
        "urgency": 3,
    },
    {
        "id": "e07",
        "subject": "Newsletter: Top 10 productivity tips for 2024",
        "sender": "newsletter@productivityhub.com",
        "body": (
            "Hi subscriber, here are our top 10 productivity tips for the new year! "
            "1. Use the Pomodoro technique. 2. Batch your emails. 3. Automate repetitive tasks..."
        ),
        "timestamp": "2024-01-15T11:00:00Z",
        "category": "general",
        "urgency": 1,
    },
    {
        "id": "e08",
        "subject": "Security alert: New login from unknown device",
        "sender": "security@ourplatform.com",
        "body": (
            "We detected a login to your account from an unrecognised device in Lagos, Nigeria. "
            "If this was you, no action is needed. "
            "If not, please reset your password immediately and contact support."
        ),
        "timestamp": "2024-01-15T11:10:00Z",
        "category": "urgent",
        "urgency": 4,
    },
    {
        "id": "e09",
        "subject": "Can I get a bulk discount?",
        "sender": "procurement@bigclient.co",
        "body": (
            "Hello, we are interested in purchasing 50 enterprise licences. "
            "Do you offer volume discounts? We would like to discuss pricing "
            "before end of month. Please have your sales team reach out."
        ),
        "timestamp": "2024-01-15T11:30:00Z",
        "category": "general",
        "urgency": 3,
    },
    {
        "id": "e10",
        "subject": "Your subscription renews tomorrow",
        "sender": "billing@ourplatform.com",
        "body": (
            "This is a reminder that your annual subscription ($149) will auto-renew "
            "tomorrow on your card ending in 4242. "
            "To cancel or change your plan, visit your account settings."
        ),
        "timestamp": "2024-01-15T12:00:00Z",
        "category": "billing",
        "urgency": 2,
    },
    {
        "id": "e11",
        "subject": "Feature request: Dark mode",
        "sender": "user456@hotmail.com",
        "body": (
            "Hi team, love the product! One request — could you add a dark mode? "
            "Many of us use the app at night and the bright screen is tiring. "
            "Would be a great addition. Thanks!"
        ),
        "timestamp": "2024-01-15T12:30:00Z",
        "category": "general",
        "urgency": 1,
    },
    {
        "id": "e12",
        "subject": "FREE Viagra — limited time offer!!!",
        "sender": "deals@pharma-discount-online.net",
        "body": "Buy now get 80% off all medications. No prescription needed. Click here...",
        "timestamp": "2024-01-15T13:00:00Z",
        "category": "spam",
        "urgency": 1,
    },
    {
        "id": "e13",
        "subject": "App crash on iOS 17 – losing customer data",
        "sender": "cto@partnercompany.com",
        "body": (
            "Our integration with your API is crashing on iOS 17 devices. "
            "This is affecting ~2,000 of our users and causing data loss. "
            "We need an urgent fix or at least a workaround. "
            "Please escalate to your engineering team immediately."
        ),
        "timestamp": "2024-01-15T13:15:00Z",
        "category": "urgent",
        "urgency": 5,
    },
    {
        "id": "e14",
        "subject": "Meeting notes from yesterday's call",
        "sender": "colleague@company.com",
        "body": (
            "Hi, attaching the notes from our product sync yesterday. "
            "Key decisions: launch pushed to Q2, design review next Friday. "
            "Let me know if I missed anything!"
        ),
        "timestamp": "2024-01-15T14:00:00Z",
        "category": "general",
        "urgency": 2,
    },
    {
        "id": "e15",
        "subject": "Double-charged on my last invoice",
        "sender": "smallbiz@outlook.com",
        "body": (
            "I was charged $49 twice on January 12th. "
            "Please refund the duplicate charge as soon as possible. "
            "Transaction IDs: TXN-8821 and TXN-8822."
        ),
        "timestamp": "2024-01-15T14:30:00Z",
        "category": "billing",
        "urgency": 4,
    },
    {
        "id": "e16",
        "subject": "Password reset request",
        "sender": "noreply@ourplatform.com",
        "body": (
            "Someone requested a password reset for this email address. "
            "If this was you, click the link below (expires in 1 hour). "
            "If not, ignore this email."
        ),
        "timestamp": "2024-01-15T14:45:00Z",
        "category": "support",
        "urgency": 3,
    },
    {
        "id": "e17",
        "subject": "Make $5000/week from home – no experience needed!",
        "sender": "rich@money-fast-easy.club",
        "body": (
            "AMAZING work-from-home opportunity! Join thousands who are already "
            "earning huge income. Zero risk. Sign up today!"
        ),
        "timestamp": "2024-01-15T15:00:00Z",
        "category": "spam",
        "urgency": 1,
    },
    {
        "id": "e18",
        "subject": "How do I export my data?",
        "sender": "user789@gmail.com",
        "body": (
            "Hi support, I'm trying to export all my project data as a CSV "
            "but I can't find the option in settings. Can you help? Thanks."
        ),
        "timestamp": "2024-01-15T15:30:00Z",
        "category": "support",
        "urgency": 2,
    },
    {
        "id": "e19",
        "subject": "Legal notice: copyright infringement claim",
        "sender": "legal@lawfirm.com",
        "body": (
            "Our client alleges that content hosted on your platform infringes "
            "their copyright. We require immediate takedown of the listed content "
            "within 48 hours or we will proceed with formal legal action. "
            "Reference: CASE-2024-LG-0045."
        ),
        "timestamp": "2024-01-15T15:45:00Z",
        "category": "urgent",
        "urgency": 5,
    },
    {
        "id": "e20",
        "subject": "Thanks for the great service!",
        "sender": "happycustomer@email.com",
        "body": (
            "Just wanted to say your support team was incredibly helpful today. "
            "Maria resolved my issue in under 10 minutes. Keep up the great work!"
        ),
        "timestamp": "2024-01-15T16:00:00Z",
        "category": "general",
        "urgency": 1,
    },
]

# ─────────────────────────────────────────────
#  Task configuration
# ─────────────────────────────────────────────

TASK_CONFIGS: dict[str, dict] = {
    "task1": {
        "name": "Email Classification",
        "difficulty": "easy",
        "num_emails": 8,
        "max_steps": 3,
        "instruction": (
            "You are an email triage assistant. "
            "Classify each email into exactly one of these categories: "
            "spam | urgent | billing | support | general. "
            "Return a JSON object with email_id as key and category as value. "
            "Example: {\"e01\": \"billing\", \"e03\": \"spam\"}"
        ),
        "valid_labels": {"spam", "urgent", "billing", "support", "general"},
    },
    "task2": {
        "name": "Inbox Prioritisation",
        "difficulty": "medium",
        "num_emails": 10,
        "max_steps": 4,
        "instruction": (
            "You are an inbox manager. "
            "Rank the emails from most urgent to least urgent. "
            "Return a JSON array of email_ids in descending urgency order. "
            "Example: [\"e04\", \"e01\", \"e13\", ...]"
        ),
    },
    "task3": {
        "name": "Professional Reply Drafting",
        "difficulty": "hard",
        "num_emails": 5,
        "max_steps": 5,
        "instruction": (
            "You are a customer support specialist. "
            "Find the email that is a customer complaint requiring an immediate response. "
            "Draft a professional, empathetic reply that: "
            "(1) acknowledges the problem, (2) apologises sincerely, "
            "(3) explains the next steps, (4) provides a resolution timeline. "
            "Return JSON: {\"email_id\": \"...\", \"subject\": \"Re: ...\", \"body\": \"...\"}"
        ),
        "complaint_ids": {"e02", "e13", "e15"},  # emails that qualify as complaints
    },
}
