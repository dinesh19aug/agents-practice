

from io import BytesIO
from PIL import Image
import requests
import base64
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


try:
  import litellm

  image_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/6/66/Painting_of_Mughal_emperor_Akbar_meeting_Guru_Amar_Das_in_1567_at_Goindwal.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/e/e8/The_Joker_at_Wax_Museum_Plus.jpg", # Joker image
        #"https://upload.wikimedia.org/wikipedia/en/9/98/Joker_%28DC_Comics_character%29.jpg"
    ]
    
  print("Starting image analysis with Ollama LLaVA and smolagents...")
    
  # Step 1: Download images from URLs
  print("\n1. Downloading images...")
  images = download_images_from_urls(image_urls)
  for i, img in enumerate(images):
    print(f"\n--- Direct LiteLLM Analysis of Image {i+1} ---")
                        
    # Ensure img is a PIL Image object
    if not isinstance(img, Image.Image):
      print(f"Warning: Expected PIL Image, got {type(img)}")
      continue
                        
                        # Convert PIL image to base64
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                       
    response = litellm.completion(
       model="ollama/llava",
       messages=[{
          "role": "user",
          "content": [
             {
                "type": "text", 
                "text": "Describe this image in detail. What do you see? Include details about objects, people, scenery, colors, and any other notable features."
              },
              {
                "type": "image_url",
                "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                              }
              }
             ]
         }],
          api_base="http://localhost:11434"
     )
                        
    print(f"Image {i+1} Analysis:")
    print(response.choices[0].message.content)
    print("-" * 50)
      
except Exception as e4:
  print(f"All approaches failed. Final error: {str(e4)}")
finally:
  print("Image analysis completed.")
    