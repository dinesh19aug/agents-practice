import cv2
import numpy as np
import yaml
import json





## --- Step 1: Load and preprocess ---
img = cv2.imread("img_yaml.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# --- Step 2: Create blue mask (adjusted to catch all blue shades) ---
lower_blue = np.array([90, 30, 50])
upper_blue = np.array([140, 255, 255])
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Clean up noise
kernel = np.ones((5, 5), np.uint8)
blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

# --- Step 3: Component counts ---
counts = {
    "browser": 0,
    "load_balancer": 0,
    "servers": 0,
    "database": {
        "type": "Oracle DB",
        "count": 0
    },
    "nas": {
        "type": "NAS FS",
        "count": 0
    },
    "agents": {
        "count": 1
    }
}

# --- Step 4: Analyze all blue objects ---
contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    ar = w / float(h)
    area = cv2.contourArea(cnt)

    # DEBUG: Print shape info
    print(f"[DEBUG] ({x},{y}) size=({w}x{h}) ar={ar:.2f} area={area:.2f}")

    if area < 800:
        continue  # skip noise

    # Debug draw all boxes
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)

    # --- Classify component ---
    if 30 < w < 100 and 30 < h < 100 and 0.8 < ar < 1.3:

        counts["servers"] += 1
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "Server", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    elif w > 120 and h < 90 and ar > 1.5:
        counts["database"]["count"] += 1
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 140, 255), 2)
        cv2.putText(img, "Oracle DB", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)

    elif 50 < w < 100 and 50 < h < 100 and ar < 1.2:
        counts["nas"]["count"] += 1
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, "NAS", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

# --- Step 5: Save debug images ---
cv2.imwrite("debug_blue_mask.jpg", blue_mask)
cv2.imwrite("debug_final_labeled.jpg", img)

# --- Step 6: Export YAML and JSON ---
yaml_output = {"components": counts}

with open("output.yaml", "w") as f:
    yaml.dump(yaml_output, f)

with open("output.json", "w") as f:
    json.dump(yaml_output, f, indent=2)

print("✅ Detection complete!")
print("📄 Output: output.yaml, output.json")
print("🖼️  Debug: debug_final_labeled.jpg, debug_blue_mask.jpg")
