# detector.py (PyTorch version)
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models

class IconDetector:
    def __init__(self, model_path="icon_classifier.pth", class_names=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names or ["database", "load_balancer", "nas", "server", "browser"]

        # Load model
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Image transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    def detect_candidates(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def classify_region(self, img, bbox):
        x, y, w, h = bbox
        roi = img[y:y+h, x:x+w]
        roi_tensor = self.transform(roi).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(roi_tensor)
            probs = torch.softmax(outputs, dim=1)
            class_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, class_idx].item()

        label = self.class_names[class_idx]
        return label, confidence

    def detect_and_classify(self, img):
        detections = []
        contours = self.detect_candidates(img)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < 100:
                continue
            label, confidence = self.classify_region(img, (x, y, w, h))
            detections.append({
                "label": label,
                "confidence": confidence,
                "bbox": (x, y, w, h)
            })
        return detections

    def annotate_image(self, img, detections):
        annotated = img.copy()
        for det in detections:
            x, y, w, h = det["bbox"]
            label = f"{det['label']} ({det['confidence']:.2f})"
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return annotated