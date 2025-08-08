import gradio as gr
import cv2
import numpy as np
from detector import IconDetector

model_path = "icon_classifier.pth"
detector = IconDetector(model_path=model_path)

def process_image(input_img):
    if isinstance(input_img, str):
        input_img = cv2.imread(input_img)
    else:
        input_img = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)
    detections = detector.detect_and_classify(input_img)
    annotated = detector.annotate_image(input_img, detections)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated_rgb

demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="numpy"),
    title="Infrastructure Diagram Icon Detector",
    description="Upload an architecture diagram. The app will detect and label icons like servers, databases, NAS, and load balancers."
)

if __name__ == "__main__":
    demo.launch()
