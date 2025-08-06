"""
Project: 
Read incoming emails
Classify them as spam or legitimate
Draft a preliminary response for legitimate emails
Send information to Mr. Wayne when legitimate (printing only)
"""

import os
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from IPython.display import Image, display
from edges import route_email
from email_state import EmailState
from nodes import classify_email, draft_response, handle_spam, notify_mr_hugg, read_email

class Graph():
  def __init__(self):
        self.graph = None

  def create_graph(self) -> StateGraph:
    
    # Create the graph
    email_graph = StateGraph(EmailState)

    # Add nodes
    email_graph.add_node("read_email", read_email)  # the read_email node executes the read_mail function
    email_graph.add_node("classify_email", classify_email)  # the classify_email node will execute the classify_email function
    email_graph.add_node("handle_spam", handle_spam)  #same logic
    email_graph.add_node("draft_response", draft_response)  #same logic
    email_graph.add_node("notify_mr_hugg", notify_mr_hugg)  # same logic

    # Add edges
    email_graph.add_edge(START, "read_email")  # After starting we go to the "read_email" node

    email_graph.add_edge("read_email", "classify_email")  # after_reading we classify


    # Add conditional edges
    email_graph.add_conditional_edges(
        "classify_email",  # after classify, we run the "route_email" function"
        route_email,
        {
            "spam": "handle_spam",  # if it return "Spam", we go the "handle_span" node
            "legitimate": "draft_response"  # and if it's legitimate, we go to the "drafting response" node
        }
    )

    # Add final edges
    email_graph.add_edge("handle_spam", END)  # after handling spam we always end
    email_graph.add_edge("draft_response", "notify_mr_hugg")
    email_graph.add_edge("notify_mr_hugg", END)  # after notifyinf Me wayne, we can end  too

    # Compile the graph
    self.graph = email_graph.compile()
    return self.graph 
  
  def display_graph(self) -> None:
    # Display the graph using Mermaid
    display(Image(self.graph.draw_mermaid_png()))