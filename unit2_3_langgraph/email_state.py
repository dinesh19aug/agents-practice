from typing import Any, Dict, List, Optional, TypedDict


class EmailState(TypedDict):
    email: Dict[str, Any ]
    email_category: Optional[str]

    # Reason why the email was marked as spam
    spam_reason: Optional[str]

    # Analysis and decisions
    is_spam: Optional[bool]
    
    # Response generation
    email_draft: Optional[str]
    
    # Processing metadata
    messages: List[Dict[str, Any]] 