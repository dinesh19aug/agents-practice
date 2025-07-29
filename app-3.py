import requests
from PIL import Image
from io import BytesIO
import base64
import tempfile
import os
from smolagents import CodeAgent, LiteLLMModel

def download_images_from_urls(urls):
    """
    Download images from URLs and return them as a list of PIL Image objects
    """
    images = []
    
    for i, url in enumerate(urls):
        try:
            print(f"Downloading image {i+1} from: {url}")
            
            # Download the image
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Convert to PIL Image
            image = Image.open(BytesIO(response.content))
            
            # Convert to RGB if necessary (some images might be in different modes)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            images.append(image)
            print(f"Successfully downloaded image {i+1}")
            
        except Exception as e:
            print(f"Error downloading image {i+1} from {url}: {str(e)}")
            continue
    
    return images

def main():
    # Example image URLs - replace with your desired URLs
    image_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/e/e8/The_Joker_at_Wax_Museum_Plus.jpg", # Joker image
        #"https://upload.wikimedia.org/wikipedia/commons/6/66/Painting_of_Mughal_emperor_Akbar_meeting_Guru_Amar_Das_in_1567_at_Goindwal.jpg"
        "https://upload.wikimedia.org/wikipedia/en/9/98/Joker_%28DC_Comics_character%29.jpg"
    ]
    
    print("Starting image analysis with Ollama LLaVA and smolagents...")
    
    # Step 1: Download images from URLs
    print("\n1. Downloading images...")
    images = download_images_from_urls(image_urls)
    
    if not images:
        print("No images were successfully downloaded. Exiting.")
        return
    
    print(f"Successfully downloaded {len(images)} images")
    
    # Step 2: Create LiteLLMModel for Ollama
    print("\n2. Setting up LiteLLMModel with Ollama...")
    
    # Configure the model to use Ollama's LLaVA with proper settings for images
    model = LiteLLMModel(
        model_id="ollama/llava",  # Ollama LLaVA model
        api_base="http://localhost:11434",  # Default Ollama API endpoint
        flatten_messages_as_text=False  # Critical: Must be False for image support
    )
    
    # Step 3: Create CodeAgent
    print("\n3. Creating CodeAgent...")
    agent = CodeAgent(
        tools=[],  # No additional tools needed for this task
        model=model,
        planning_interval=1,  # Adjust planning interval as needed
        verbosity_level=1,  # Set verbosity level for debugging
        max_steps=1  # Limit the number of steps to prevent infinite loops
    )
    
    # Step 4: Analyze each image using agent.run() approach
    print("\n4. Analyzing images with agent.run()...")
    
    
    try:
        print("Using agent.run() with PIL images...")
        
        for i, img in enumerate(images):
            print(f"\n--- agent.run() Analysis of Image {i+1} ---")
            
            # Ensure img is a PIL Image object
            if not isinstance(img, Image.Image):
                print(f"Warning: Expected PIL Image, got {type(img)}")
                continue
            list_img = [img]  # Wrap in a list to match expected input
            # Try passing PIL image directly to agent.run()
            response = agent.run(
                "Describe this image in detail. What do you see? Include details about objects, people, scenery, colors, and any other notable features.",
                images=list_img  # Pass PIL image directly
            )
            print(f"Image {i+1} Analysis:")
            print(response)
            print("-" * 50)
            
    except Exception as e:
        print(f"agent.run() with PIL images failed: {str(e)}")
        
        

if __name__ == "__main__":
    # Required dependencies check
    try:
        import smolagents
        print("smolagents imported successfully")
    except ImportError:
        print("Please install smolagents: pip install smolagents")
        exit(1)
    
    try:
        import requests
        from PIL import Image
        print("Required dependencies available")
    except ImportError:
        print("Please install required dependencies:")
        print("pip install requests pillow")
        exit(1)
    
    main()