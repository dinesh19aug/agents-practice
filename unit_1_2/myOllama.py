# Custom Ollama Model Class
from typing import Dict, List
from smolagents.models import ChatMessage 
import requests



class OllamaModel:
    def __init__(self, model_name="qwen:2.5", host="http://localhost:11434", 
                 max_tokens=4096, 
                 temperature=0.5):
        self.model_name = model_name
        self.host = host
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    @property
    def model_id(self):
        return self.model_name
    
    def generate(self, messages, **kwargs):
        # Convert list of ChatMessages to a single prompt string
        if isinstance(messages, list) and all(isinstance(m, ChatMessage) for m in messages):
            prompt = self._convert_messages_to_prompt(messages)
        elif isinstance(messages, str):
            prompt = messages
        else:
            raise TypeError("Unsupported message format")

        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        content = response.json()["response"]


        # Wrap response in a ChatMessage to match smolagents expectations
        #return ChatMessage(role="assistant", content=content)
        # Create a custom response object that mimics what smolagents expects
        class MockResponse:
            def __init__(self, content, input_tokens, output_tokens):
                self.content = content
                self.input_tokens = input_tokens
                self.output_tokens = output_tokens
                # Create a token_usage object to satisfy smolagents requirements
                self.token_usage = type('TokenUsage', (), {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': input_tokens + output_tokens
                })()
        
        # Estimate token counts since Ollama doesn't provide them
        prompt_tokens = len(prompt.split()) * 1.3  # Rough estimation
        completion_tokens = len(content.split()) * 1.3  # Rough estimation
        
        # Return the mock response with token usage
        return MockResponse(
            content=content,
            input_tokens=int(prompt_tokens),
            output_tokens=int(completion_tokens)
        )

    def __call__(self, messages, **kwargs):
        return self.generate(messages, **kwargs)

    def _convert_messages_to_prompt(self, messages):
        prompt = ""
        for msg in messages:
            if msg.role == "system":
                prompt += f"[System]: {msg.content}\n"
            elif msg.role == "user":
                prompt += f"[User]: {msg.content}\n"
            elif msg.role == "assistant":
                prompt += f"[Assistant]: {msg.content}\n"
            #elif msg.role == "TOOL_RESPONSE":
              # Make this unambiguous so Qwen knows to trust it
              #prompt += f"Observation: {msg.content}\n"
              #print(f"Observation: {msg.content}\n")
        prompt += "[Assistant]:"
        
        return prompt
    
    
# Test Ollama connection
def test_ollama_connection(model_name: str = "qwen2.5:latest", base_url: str = "http://localhost:11434"):
    """Test if Ollama is running and the model is available"""
    try:
        # Test if Ollama is running
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=10)
        response.raise_for_status()
        
        models = response.json().get('models', [])
        available_models = [model['name'] for model in models]
        
        print(f"Ollama is running. Available models: {available_models}")
        
        # Check if our model is available
        model_found = any(model_name in model for model in available_models)
        if model_found:
            print(f"✓ Model '{model_name}' is available")
            return True
        else:
            print(f"✗ Model '{model_name}' not found. Available models: {available_models}")
            print(f"Make sure you've pulled the model with: ollama pull {model_name}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot connect to Ollama at {base_url}: {e}")
        print("Make sure Ollama is running with: ollama serve")
        return False


