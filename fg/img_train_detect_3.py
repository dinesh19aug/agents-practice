import os
import cv2
import yaml
from collections import defaultdict

def preprocess(img):
    """Convert image to grayscale, blur, then edge-detect for robust matching."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def template_match(cropped_img, template, threshold=0.75, scales=[0.8, 1.0, 1.2]):
    cropped_edges = preprocess(cropped_img)
    best_val = 0

    for scale in scales:
        scaled_template = cv2.resize(template, None, fx=scale, fy=scale)
        template_edges = preprocess(scaled_template)

        if cropped_edges.shape[0] < template_edges.shape[0] or cropped_edges.shape[1] < template_edges.shape[1]:
            continue

        result = cv2.matchTemplate(cropped_edges, template_edges, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        best_val = max(best_val, max_val)

    return best_val >= threshold

def detect_regions(image, min_area=800):
    """Detect candidate rectangular regions based on contours to reduce search space."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area >= min_area:
            boxes.append((x, y, x + w, y + h))
    return boxes

def main():
    image_path = "/Users/dinesharora/Desktop/dinesh/leraning/agents-course/agents-practice/fg/img_yaml_2.png"
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Template directories with draw.io icons exactly matching your diagram
    template_dirs = {
        "server": "/Users/dinesharora/Desktop/dinesh/leraning/agents-course/agents-practice/fg/templates/server",
        "database": "/Users/dinesharora/Desktop/dinesh/leraning/agents-course/agents-practice/fg/templates/database",
        "load_balancer": "/Users/dinesharora/Desktop/dinesh/leraning/agents-course/agents-practice/fg/templates/load_balancer",
        "nas": "/Users/dinesharora/Desktop/dinesh/leraning/agents-course/agents-practice/fg/templates/nas"
    }

    # Load templates
    templates = {}
    for label, folder in template_dirs.items():
        templates[label] = []
        for file in os.listdir(folder):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(folder, file)
                img = cv2.imread(path)
                if img is not None:
                    templates[label].append(img)
                    print(f"Loaded template '{file}' for label '{label}', shape={img.shape}")

    # Detect candidate regions in the image
    boxes = detect_regions(image)
    print(f"🔍 Total candidate regions found: {len(boxes)}")

    component_counts = defaultdict(int)
    debug_img = image.copy()

    for (x1, y1, x2, y2) in boxes:
        cropped = image[y1:y2, x1:x2]
        matched = False

        for label, template_list in templates.items():
            for template in template_list:
                if template_match(cropped, template):
                    print(f"✅ Matched {label} at ({x1}, {y1})")
                    component_counts[label] += 1
                    color = (0, 255, 0)  # green box for matches
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(debug_img, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    matched = True
                    break
            if matched:
                break

        if not matched:
            print(f"❌ No match found at ({x1}, {y1})")
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red box for no match

    # Save component counts to YAML
    with open("detected_components.yaml", "w") as f:
        yaml.dump(dict(component_counts), f)

    print("📄 YAML saved to detected_components.yaml")

    # Save and show debug image with detections
    debug_path = "debug_template_match.png"
    cv2.imwrite(debug_path, debug_img)
    print(f"🖼️ Debug image saved to {debug_path}")
    cv2.imshow("Detected Components", debug_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
