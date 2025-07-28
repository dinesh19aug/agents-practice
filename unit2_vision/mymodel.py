import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
## Add Model
from unit2_vision.visionollama import OllamaVisionModel


class MyModel():
    def __init__(self):
        self.model = None

    def get_qwen2_5_model(self):
        
        self.model = OllamaVisionModel(
        model_name="qwen2.5:latest",  # Use the exact model name from ollama list
        host="http://localhost:11434",
        max_tokens=8096,
        temperature=0.5)

        return self.model
    
    def get_gemma3_model(self):
        
        self.model = OllamaVisionModel(
        model_name="gemma3:latest",  # Use the exact model name from ollama list
        host="http://localhost:11434",
        max_tokens=8096,
        temperature=0.5)

        return self.model
    
    def get_qwen2_5vl_model(self):
        
        self.model = OllamaVisionModel(
        model_name="minicpm-v:latest",  # Use the exact model name from ollama list
        host="http://localhost:11434",
        max_tokens=8096,
        temperature=0.8)

        return self.model
    
    def get_llava_vision_model(self):
        
        self.model = OllamaVisionModel(
        model_name="llava:latest",  # Use the exact model name from ollama list
        host="http://localhost:11434",
        max_tokens=8096,
        temperature=0.8,verbosity_level=2)

        return self.model