from email_state import EmailState


class Edge():
    """Base class for edges in a graph"""
    def __init__(self, name: str):
        self.name = name
        
def route_email(state: EmailState) -> str:
    """Determine the next step based on spam classification"""
    if state["is_spam"]:
        return "spam"
    else:
        return "legitimate"