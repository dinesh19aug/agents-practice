from typing import Any

from langchain_ollama import ChatOllama


class LLM_model:
    """
    Base class for LLM models.
    This class should be extended by specific LLM model implementations.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name

    def get_llm_model(self) -> ChatOllama:
        """
        Returns the LLM model instance.
        This method should be implemented by subclasses.
        """
        return ChatOllama(model = self.model_name,reasoning=False,
                          num_ctx=1000,
                          num_gpu=1,
                          temperature=0.5,
                          
                          )

        