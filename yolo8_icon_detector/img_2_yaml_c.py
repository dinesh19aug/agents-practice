import cv2
import numpy as np
import yaml
import json
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
#from torchvision.models import resnet18
from torchvision import models
import os

class InfrastructureDetector:
    def __init__(self):
        # Load classifier model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.classifier = models.resnet18(weights=None)    
        self.classifier = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        #self.classifier.fc = nn.Linear(self.classifier.fc.in_features, 5)  # Adjust number of classes if needed
        self.classifier.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.classifier.fc.in_features, 5))
        self.classifier.load_state_dict(torch.load("yolo8_icon_detector/component_classifier.pth", map_location=self.device))
        self.classifier.to(self.device)
        self.classifier.eval()

        with open("yolo8_icon_detector/component_labels.txt") as f:
            self.class_labels = [line.strip() for line in f]

        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

    def preprocess_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")

        self.original_img = img.copy()
        self.img_height, self.img_width = img.shape[:2]
        self.img_area = self.img_width * self.img_height

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return img, hsv

    """
    def create_combined_mask(self, hsv):
        gray = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return mask
    """
    def create_combined_mask(self, hsv):
        gray = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # --- Work on a COPY for arrow detection ---
        arrow_detect_img = mask.copy()

        # Try to detect straight lines
        edges = cv2.Canny(arrow_detect_img, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                            minLineLength=30, maxLineGap=5)
        arrows_mask = np.zeros_like(mask)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(arrows_mask, (x1, y1), (x2, y2), 255, 3)

        # Detect thin curved lines via skeletonization
        try:
            skeleton = cv2.ximgproc.thinning(arrow_detect_img)
        except AttributeError:
            skeleton = arrow_detect_img.copy()

        contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            length = cv2.arcLength(cnt, False)
            area = cv2.contourArea(cnt)
            avg_width = area / (length + 1e-5)

            if avg_width < 5 and length > 30:
                cv2.drawContours(arrows_mask, [cnt], -1, 255, -1)

        # Subtract arrows from ORIGINAL mask
        mask_no_arrows = cv2.bitwise_and(mask, cv2.bitwise_not(arrows_mask))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))
        mask_no_arrows = cv2.morphologyEx(mask_no_arrows, cv2.MORPH_CLOSE, vertical_kernel)

        return mask_no_arrows

    def extract_shape_features(self, contour):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        relative_area = area / self.img_area
        aspect_ratio = w / float(h) if h > 0 else 0

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        extent = area / (w * h) if (w * h) > 0 else 0
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

        return {
            'area': area,
            'relative_area': relative_area,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'extent': extent,
            'circularity': circularity,
            'bbox': (x, y, w, h)
        }

    def classify_icon_with_model(self, img_crop):
        pil_img = Image.fromarray(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
        tensor = self.img_transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.classifier(tensor)
            predicted_idx = torch.argmax(outputs, dim=1).item()
            confidence = torch.softmax(outputs, dim=1)[0][predicted_idx].item()
            return self.class_labels[predicted_idx], confidence

    def detect_components(self, img_path):
        img, hsv = self.preprocess_image(img_path)
        mask = self.create_combined_mask(hsv)
        cv2.imwrite("yolo8_icon_detector/debug_mask.jpg", mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for contour in contours:
            features = self.extract_shape_features(contour)
            if features['relative_area'] < 0.0005:
                continue

            x, y, w, h = features['bbox']
            img_crop = self.original_img[y:y+h, x:x+w]
            debug_dir = "yolo8_icon_detector/debug_crops"
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f"crop_{x}_{y}_{w}x{h}.jpg"), img_crop)
            comp_type, confidence = self.classify_icon_with_model(img_crop)
            print(f"[DEBUG] Detected contour at ({x},{y}) size=({w}x{h}) -> classified as '{comp_type}' with confidence {confidence:.2f}")

            if confidence >= 0.6:
                detections.append({
                    'type': comp_type,
                    'confidence': confidence,
                    'bbox': (x, y, w, h)
                })
                print(f"[FINAL] {comp_type} at ({x},{y}) size=({w}x{h}) confidence={confidence:.2f}")
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, f"{comp_type} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return img, detections

    def save_results(self, img, detections, output_prefix="output"):
        cv2.imwrite(f"yolo8_icon_detector/{output_prefix}_annotated.jpg", img)
        data = {
            'components': detections
        }
        with open(f"yolo8_icon_detector/{output_prefix}.yaml", "w") as f:
            yaml.dump(data, f)
        with open(f"yolo8_icon_detector/{output_prefix}.json", "w") as f:
            json.dump(data, f, indent=2)
        return data


def main(image_path="yolo8_icon_detector/img_yaml_8.png"):
    detector = InfrastructureDetector()
    try:
        annotated_img, detections = detector.detect_components(image_path)
        results = detector.save_results(annotated_img, detections)

        print("\n✅ Detection complete!")
        print(f"📄 Total components detected: {len(detections)}")
        print("📄 Output files:")
        print("   - yolo8_icon_detector/output.yaml, yolo8_icon_detector/output.json")
        print("   - yolo8_icon_detector/output_annotated.jpg")
        print("   - yolo8_icon_detector/debug_mask.jpg")

        return results
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()