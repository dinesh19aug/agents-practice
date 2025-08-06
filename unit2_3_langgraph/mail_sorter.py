"""
Project: 
Read incoming emails
Classify them as spam or legitimate
Draft a preliminary response for legitimate emails
Send information to Mr. Wayne when legitimate (printing only)
"""


from graph import Graph

# Example emails for testing
legitimate_email = {
    "sender": "order-status@flexispot.com",
    "subject": "Order 234-001-EA has been shipped",
    "body": "Mr. Wayne, Your order has been shipped and will arrive on 2025-11-01. Thank you for your purchase!"
}

spam_email = {
    "sender": "Crypto bro",
    "subject": "The best investment of 2025",
    "body": "Mr Wayne, I just launched an ALT coin and want you to buy some !"
}

# Process legitimate email
print("\nProcessing legitimate email...")
compiled_graph = Graph().create_graph()
legitimate_result = compiled_graph.invoke({
    "email": legitimate_email,
    "is_spam": None,
    "spam_reason": None,
    "email_category": None,
    "email_draft": None,
    "messages": []
})

# Process spam email
print("\nProcessing spam email...")
spam_result = compiled_graph.invoke({
    "email": spam_email,
    "is_spam": None,
    "spam_reason": None,
    "email_category": None,
    "email_draft": None,
    "messages": []
})