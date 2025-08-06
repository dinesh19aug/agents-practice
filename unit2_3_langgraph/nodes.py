from email_state import EmailState
from llm_models import LLM_model

from langchain_core.messages import HumanMessage


def read_email(state: EmailState) -> None:
    """
    Reads an email from the state and processes it.
    
    Args:
        state (EmailState): The current state containing email information.
    """
    email = state['email']
    print(f"Reading email from: {email['sender']} with subject: {email['subject']}")
    return {}

def classify_email(state: EmailState) -> None:
    """
    Classifies the email as spam or legitimate.
    
    Args:
        state (EmailState): The current state containing email information.
    """
    email = state['email']
    #create prompt
    
    prompt = f"""As Alfred the butler, analyze this email and determine if it is spam or legitimate. 
                      Sender: {email['sender']}
                      Subject: {email['subject']}
                      Content: {email['body']} 
      First, determine if this email is spam. If it is spam, explain why.
      If it is legitimate, categorize it (inquiry, complaint, thank you, etc.). Finally return the response as simple key value
        pairs: is_spam (true/false), spam_reason (if spam), email_category (if legitimate).
      """
    
    #Call the language model to classify the email
    llm = LLM_model("qwen2.5").get_llm_model()
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)

    # Simple logic to parse the response (in a real app, you'd want more robust parsing)
    response_text = response.content.lower()
    print(f"LLM response: {response_text}")
    is_spam = "is_spam: true" in response_text and "is_spam: false" not in response_text
    
    # Extract a reason if it's spam
    spam_reason = None
    if is_spam and "spam_reason:" in response_text:
        spam_reason = response_text.split("reason:")[1].strip()
    
    # Determine category if legitimate
    email_category = None
    if not is_spam:
        categories = ["inquiry", "complaint", "thank you", "request", "information"]
        for category in categories:
            if category in response_text:
                email_category = category
                break
    
    # Update messages for tracking
    new_messages = state.get("messages", []) + [
        
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content}
    ]
    

    # Return state updates
    return {
        "is_spam": is_spam,
        "spam_reason": spam_reason,
        "email_category": email_category,
        "messages": new_messages
    }

def handle_spam(state: EmailState) -> None:
    """Alfred discards spam email with a note"""
    print(f"Alfred has marked the email as spam. Reason: {state['spam_reason']}")
    print("The email has been moved to the spam folder.")
    
    # We're done processing this email
    return {}
    

def draft_response(state: EmailState):
    """Alfred drafts a preliminary response for legitimate emails"""
    email = state["email"]
    category = state["email_category"] or "general"
    
    # Prepare our prompt for the LLM
    prompt = f"""
    As Alfred the butler, draft a polite preliminary response to this email.
    
    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}
    
    This email has been categorized as: {category}
    
    Draft a brief, professional response that Mr. Hugg can review and personalize before sending.
    """
    
    # Call the LLM
    llm = LLM_model("qwen2.5").get_llm_model()
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    
    # Update messages for tracking
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content}
    ]
    
    # Return state updates
    return {
        "email_draft": response.content,
        "messages": new_messages
    }

def notify_mr_hugg(state: EmailState):
    """Alfred notifies Mr. Hugg about the email and presents the draft response"""
    email = state["email"]
    
    print("\n" + "="*50)
    print(f"Sir, you've received an email from {email['sender']}.")
    print(f"Subject: {email['subject']}")
    print(f"Category: {state['email_category']}")
    print("\nI've prepared a draft response for your review:")
    print("-"*50)
    print(state["email_draft"])
    print("="*50 + "\n")
    
    # We're done processing this email
    return {}