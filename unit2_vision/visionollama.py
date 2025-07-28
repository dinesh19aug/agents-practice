"""
Custom Ollama Vision Model for smolagents compatibility
"""
from typing import Dict, List, Optional, Any
from smolagents.models import ChatMessage 
import requests
import base64
from io import BytesIO
from PIL import Image


def pil_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class MockResponse:
    """Mock response object to match smolagents expectations"""
    def __init__(self, content, input_tokens, output_tokens):
        self.content = content
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.token_usage = type('TokenUsage', (), {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens
        })()


class OllamaVisionModel:
    """Custom Ollama Vision Model for smolagents compatibility"""
    
    def __init__(self, model_name="qwen:2.5", host="http://localhost:11434", 
                 max_tokens=4096, temperature=0.5, verbosity_level=1):
        """Initialize the Ollama Vision Model"""
        self.model_name = model_name
        self.host = host
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbosity_level = verbosity_level
        
        if self.verbosity_level >= 1:
            print(f"🤖 Initialized OllamaVisionModel with:")
            print(f"   Model: {self.model_name}")
            print(f"   Host: {self.host}")
            print(f"   Max tokens: {self.max_tokens}")
            print(f"   Temperature: {self.temperature}")
    
    @property
    def model_id(self):
        """Return model identifier"""
        self._add_images_to_messages
        return self.model_name
    
    def __call__(self, messages, **kwargs):
        """Make the model callable - this is how smolagents calls the model"""
        images = kwargs.get('images', None)
        
        if self.verbosity_level >= 2:
            print(f"🔄 Model called with {len(messages) if isinstance(messages, list) else 1} messages")
            if images:
                print(f"📸 Images provided: {len(images)}")
        
        return self.generate(messages, images=images, **kwargs)
    """
    def generate(self, messages, images=None, **kwargs):
        
        # Check if we have images (either passed directly or in kwargs)
        print(f"🔍 Checking for images in kwargs: {kwargs.get('images', None)}")
        if images is None:
            images = kwargs.get('images', None)
        
        # Use different endpoints based on whether we have images
        if images and len(images) > 0:
            # Use chat API for vision tasks
            return self._generate_with_chat_api(messages, images)
        else:
            # Use generate API for text-only tasks (better smolagents compatibility)
            return self._generate_with_generate_api(messages)
    """
    def _generate_with_chat_api(self, messages, images):
        """Use /api/chat endpoint for vision tasks"""
        # Handle different message formats for chat API
        ollama_messages = self._process_messages_for_chat(messages)
        
        # Add images to the last user message
        if images:
            self._add_images_to_messages(ollama_messages, images)
        
        try:
            if self.verbosity_level >= 1:
                print(f"🌐 Using /api/chat endpoint (vision)")
                print(f"📋 Model: {self.model_name}")
                print(f"💬 Messages: {len(ollama_messages)} messages")
                print(f"🖼️ Images: {len(images)} images attached")
            
            response = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": ollama_messages,
                    "stream": False,
                    "temperature": self.temperature,
                    "options": {"num_ctx": self.max_tokens}
                },
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            if "message" not in result or "content" not in result["message"]:
                print(f"❌ Unexpected response structure: {result}")
                content = "Error: Unexpected response structure from Ollama"
            else:
                content = result["message"]["content"]
                if self.verbosity_level >= 1:
                    print(f"✅ Successfully received response from Ollama")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error communicating with Ollama: {e}")
            content = f"Error: Could not connect to Ollama model. {str(e)}"
        except KeyError as e:
            print(f"❌ Unexpected response format from Ollama: {e}")
            content = "Error: Unexpected response format from Ollama"
        
        # Calculate approximate token counts
        prompt_text = " ".join(m.get("content", "") for m in ollama_messages)
        prompt_tokens = int(len(prompt_text.split()) * 1.3)
        completion_tokens = int(len(content.split()) * 1.3)
        
        return MockResponse(
            content=content,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens
        )
    
    def _generate_with_generate_api(self, messages):
        """Use /api/generate endpoint for text-only tasks (smolagents compatible)"""
        # Convert messages to prompt string
        if isinstance(messages, list) and all(hasattr(m, 'role') and hasattr(m, 'content') for m in messages):
            prompt = self._convert_messages_to_prompt(messages)
        elif isinstance(messages, str):
            prompt = messages
        else:
            raise TypeError("Unsupported message format")
        
        try:
            if self.verbosity_level >= 1:
                print(f"🌐 Using /api/generate endpoint (text-only)")
                print(f"📋 Model: {self.model_name}")
                print(f"📝 Prompt length: {len(prompt)} characters")
            
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": self.temperature,
                    "options": {"num_ctx": self.max_tokens}
                },
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            if "response" not in result:
                print(f"❌ Unexpected response structure: {result}")
                content = "Error: Unexpected response structure from Ollama"
            else:
                content = result["response"]
                if self.verbosity_level >= 1:
                    print(f"✅ Successfully received response from Ollama")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error communicating with Ollama: {e}")
            content = f"Error: Could not connect to Ollama model. {str(e)}"
        except KeyError as e:
            print(f"❌ Unexpected response format from Ollama: {e}")
            content = "Error: Unexpected response format from Ollama"
        
        # Calculate approximate token counts
        prompt_tokens = int(len(prompt.split()) * 1.3)
        completion_tokens = int(len(content.split()) * 1.3)
        
        return MockResponse(
            content=content,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens
        )
    
    def _convert_messages_to_prompt(self, messages):
        """Convert list of ChatMessages to a single prompt string for /api/generate"""
        prompt = ""
        for msg in messages:
            if msg.role == "system":
                prompt += f"[System]: {msg.content}\n"
            elif msg.role == "user":
                prompt += f"[User]: {msg.content}\n"
            elif msg.role == "assistant":
                prompt += f"[Assistant]: {msg.content}\n"
            elif msg.role == "TOOL_RESPONSE":
                # Make this unambiguous so the model knows to trust it
                prompt += f"Observation: {msg.content}\n"
        prompt += "[Assistant]:"
        
        return prompt
    
    def _process_messages_for_chat(self, messages):
        """Convert various message formats to Ollama chat format"""
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        
        if isinstance(messages, list):
            ollama_messages = []
            for m in messages:
                if hasattr(m, "role") and hasattr(m, "content"):
                    # Handle ChatMessage objects
                    content = self._extract_content(m.content)
                    if content:
                        ollama_messages.append({"role": m.role, "content": content})
                elif isinstance(m, dict) and "role" in m and "content" in m:
                    # Handle dict messages
                    content = self._extract_content(m["content"])
                    if content:
                        ollama_messages.append({"role": m["role"], "content": content})
            return ollama_messages
        
        raise TypeError(f"Unsupported message format: {type(messages)}")
        """Convert various message formats to Ollama format"""
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        
        if isinstance(messages, list):
            ollama_messages = []
            for m in messages:
                if hasattr(m, "role") and hasattr(m, "content"):
                    # Handle ChatMessage objects
                    content = self._extract_content(m.content)
                    if content:
                        ollama_messages.append({"role": m.role, "content": content})
                elif isinstance(m, dict) and "role" in m and "content" in m:
                    # Handle dict messages
                    content = self._extract_content(m["content"])
                    if content:
                        ollama_messages.append({"role": m["role"], "content": content})
            return ollama_messages
        
        raise TypeError(f"Unsupported message format: {type(messages)}")
    
    def _extract_content(self, content):
        """Extract text content from various content formats"""
        if isinstance(content, str):
            return content.strip()
        elif isinstance(content, list):
            return " ".join(str(item) for item in content if str(item).strip())
        else:
            return str(content).strip()
    
    def _add_images_to_messages(self, ollama_messages, images):
        """Add base64 encoded images to the last user message"""
        try:
            b64_images = []
            for img in images:
                if isinstance(img, Image.Image):
                    b64_images.append(pil_to_base64(img))
                elif isinstance(img, str):
                    # Assume it's a file path
                    with Image.open(img) as pil_img:
                        b64_images.append(pil_to_base64(pil_img.convert("RGB")))
                else:
                    print(f"⚠️ Warning: Unsupported image type: {type(img)}")
            
            # Add images to the last user message
            for msg in reversed(ollama_messages):
                if msg["role"] == "user":
                    msg["images"] = b64_images
                    break
                    
        except Exception as e:
            print(f"❌ Error processing images: {e}")


def test_ollama_connection(model_name: str = "qwen2.5:latest", base_url: str = "http://localhost:11434"):
    """Test if Ollama is running and the model is available"""
    try:
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=10)
        response.raise_for_status()
        
        models = response.json().get('models', [])
        available_models = [model['name'] for model in models]
        
        print(f"🔗 Ollama is running. Available models: {available_models}")
        
        model_found = any(model_name in model for model in available_models)
        if model_found:
            print(f"✅ Model '{model_name}' is available")
            return True
        else:
            print(f"❌ Model '{model_name}' not found. Available models: {available_models}")
            print(f"Run: ollama pull {model_name}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to Ollama at {base_url}: {e}")
        print("Make sure Ollama is running with: ollama serve")
        return False


# Explicit exports
__all__ = ['OllamaVisionModel', 'MockResponse', 'test_ollama_connection']