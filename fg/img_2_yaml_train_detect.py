import cv2
import numpy as np
import yaml
import json
import os
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path
import glob

def get_images_from_folder(folder_path):
    """Return a list of image file paths from a folder (png, jpg, jpeg)"""
    exts = ('*.png', '*.jpg', '*.jpeg')
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder_path, ext), recursive=True))
    print(f"DEBUG: Found {len(files)} images in '{folder_path}': {files}")
    return files

class TemplateBasedDetector:
    def __init__(self, template_dir="templates"):
        self.template_dir = Path(template_dir)
        self.templates = {
            'server': [],
            'database': [],
            'nas': [],
            'load_balancer': []
        }
        self.feature_extractors = []
        self.trained_model = None
        self.template_features = {}

        # Color ranges for initial filtering
        self.color_ranges = {
            'blue': ([90, 50, 50], [130, 255, 255]),
            'light_blue': ([80, 40, 100], [110, 255, 255]),
            'purple': ([130, 60, 60], [160, 255, 255]),
            'cyan': ([85, 50, 80], [105, 255, 255])
        }

        # Load existing templates if available
        self.load_templates()

    def create_template_directory(self):
        """Create template directory structure"""
        self.template_dir.mkdir(exist_ok=True)
        for comp_type in self.templates.keys():
            (self.template_dir / comp_type).mkdir(exist_ok=True)

        print(f"📁 Template directory created: {self.template_dir}")
        print("📋 To add templates:")
        print("   1. Place server images in: templates/server/")
        print("   2. Place database images in: templates/database/")
        print("   3. Place NAS images in: templates/nas/")
        print("   4. Place load balancer images in: templates/load_balancer/")

    def extract_template_from_image(self, image_path, save_dir=None, interactive=True):
        """Interactive tool to extract templates from full images"""
        # Try different ways to load the image
        img = None
        image_path_str = str(image_path)

        # Debug: Print image path and check if file exists
        print(f"🔍 Attempting to load image: {image_path_str}")

        if not os.path.exists(image_path_str):
            print(f"❌ File does not exist: {image_path_str}")
            return []

        # Try loading image
        img = cv2.imread(image_path_str)

        if img is None:
            print(f"❌ Could not load image with cv2.imread: {image_path_str}")
            # Try with different flags
            img = cv2.imread(image_path_str, cv2.IMREAD_COLOR)
            if img is None:
                print(f"❌ Failed with IMREAD_COLOR flag as well")
                return []

        print(f"✅ Image loaded successfully: {img.shape}")
        print(f"   Image dimensions: {img.shape[1]}x{img.shape[0]} pixels")

        # Check if image is too large and resize if needed
        max_display_size = 1200
        display_img = img.copy()
        scale_factor = 1.0

        if max(img.shape[:2]) > max_display_size:
            scale_factor = max_display_size / max(img.shape[:2])
            new_width = int(img.shape[1] * scale_factor)
            new_height = int(img.shape[0] * scale_factor)
            display_img = cv2.resize(img, (new_width, new_height))
            print(f"📏 Resized for display: {new_width}x{new_height} (scale: {scale_factor:.2f})")

        clone = display_img.copy()
        templates_extracted = []
        current_selection = None

        def mouse_callback(event, x, y, flags, param):
            nonlocal clone, templates_extracted, current_selection, display_img

            if event == cv2.EVENT_LBUTTONDOWN:
                # Start selection
                param['start'] = (x, y)
                param['selecting'] = True
                current_selection = None
                print(f"🎯 Selection started at: ({x}, {y})")

            elif event == cv2.EVENT_MOUSEMOVE and param.get('selecting'):
                # Draw current selection rectangle
                temp_clone = display_img.copy()

                # Draw existing selections
                for i, template_info in enumerate(templates_extracted):
                    x1, y1, x2, y2 = template_info['display_coords']
                    cv2.rectangle(temp_clone, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(temp_clone, f"Template {i + 1}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Draw current selection
                start = param['start']
                cv2.rectangle(temp_clone, start, (x, y), (0, 0, 255), 2)
                clone = temp_clone

            elif event == cv2.EVENT_LBUTTONUP and param.get('selecting'):
                # End selection
                end = (x, y)
                start = param['start']

                x1, y1 = min(start[0], end[0]), min(start[1], end[1])
                x2, y2 = max(start[0], end[0]), max(start[1], end[1])

                if x2 - x1 > 10 and y2 - y1 > 10:  # Minimum size
                    # Convert display coordinates back to original image coordinates
                    orig_x1 = int(x1 / scale_factor)
                    orig_y1 = int(y1 / scale_factor)
                    orig_x2 = int(x2 / scale_factor)
                    orig_y2 = int(y2 / scale_factor)

                    # Extract template from original image
                    template = img[orig_y1:orig_y2, orig_x1:orig_x2]

                    template_info = {
                        'template': template,
                        'display_coords': (x1, y1, x2, y2),
                        'original_coords': (orig_x1, orig_y1, orig_x2, orig_y2)
                    }
                    templates_extracted.append(template_info)

                    # Draw rectangle on clone
                    cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(clone, f"Template {len(templates_extracted)}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    print(
                        f"✅ Extracted template {len(templates_extracted)}: {x2 - x1}x{y2 - y1} (display) -> {orig_x2 - orig_x1}x{orig_y2 - orig_y1} (original)")
                else:
                    print("⚠️  Selection too small, minimum size is 10x10 pixels")

                param['selecting'] = False

        if interactive:
            print("\n🖱️  INSTRUCTIONS:")
            print("   • Click and drag to select components")
            print("   • Press 'q' to finish and proceed")
            print("   • Press 'r' to reset all selections")
            print("   • Press 's' to show current selections")
            print("   • Make sure to select each type of component you want to detect")

            # Create window with proper flags
            window_name = 'Extract Templates - Click and Drag to Select'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

            # Set window size
            window_width = min(1000, display_img.shape[1])
            window_height = min(700, display_img.shape[0])
            cv2.resizeWindow(window_name, window_width, window_height)

            mouse_params = {'selecting': False, 'start': None}
            cv2.setMouseCallback(window_name, mouse_callback, mouse_params)

            print(f"🖼️  Displaying image in window: {window_name}")

            while True:
                cv2.imshow(window_name, clone)
                key = cv2.waitKey(30) & 0xFF  # Increased wait time for better responsiveness

                if key == ord('q'):
                    print("✅ Finished selecting templates")
                    break
                elif key == ord('r'):
                    clone = display_img.copy()
                    templates_extracted = []
                    print("🔄 Reset all selections")
                elif key == ord('s'):
                    print(f"📊 Current selections: {len(templates_extracted)}")
                    for i, template_info in enumerate(templates_extracted):
                        coords = template_info['original_coords']
                        print(
                            f"   Template {i + 1}: {coords[2] - coords[0]}x{coords[3] - coords[1]} at ({coords[0]}, {coords[1]})")
                elif key == 27:  # ESC key
                    print("❌ Cancelled by user")
                    templates_extracted = []
                    break

            cv2.destroyAllWindows()

        # Save templates if save_dir provided
        if save_dir and templates_extracted:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            for i, template_info in enumerate(templates_extracted):
                template = template_info['template']
                template_path = save_path / f"template_{i:03d}.png"
                cv2.imwrite(str(template_path), template)
                print(f"💾 Saved: {template_path}")

        # Return just the template images for compatibility
        return [t['template'] for t in templates_extracted]

    def load_templates(self):
        """Load templates from the template directory and its subdirectories."""
        if not self.template_dir.exists():
            self.create_template_directory()

        # Iterate through all subdirectories under template_dir
        for subdir in self.template_dir.iterdir():
            if subdir.is_dir():
                # Load images from each subdirectory
                for img_path in subdir.glob("*.png"):
                    label = subdir.name  # Use the subdirectory name as the label
                    template = cv2.imread(str(img_path))
                    if template is not None:
                        self.templates[label].append({
                            'image': template,
                            'path': str(img_path),
                            'features': None
                        })
                        print(f"✅ Loaded template for {label} from {img_path}")

    def extract_visual_features(self, img):
        """Extract multiple visual features from an image"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        else:
            gray = img
            hsv = None

        features = {}

        # 1. HOG features (shape/edge patterns)
        hog = cv2.HOGDescriptor((32, 32), (8, 8), (4, 4), (4, 4), 9)
        resized = cv2.resize(gray, (32, 32))
        hog_features = hog.compute(resized).flatten()
        features['hog'] = hog_features

        # 2. Color histogram
        if hsv is not None:
            hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
            color_features = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
            features['color'] = color_features

        # 3. Texture features (LBP-like)
        texture_features = self.extract_texture_features(gray)
        features['texture'] = texture_features

        # 4. Geometric features
        contours, _ = cv2.findContours(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            geometric_features = self.extract_geometric_features(largest_contour, gray.shape)
            features['geometric'] = geometric_features
        else:
            features['geometric'] = np.zeros(10)

        # Combine all features
        combined = np.concatenate([
            features['hog'],
            features['color'] if 'color' in features else np.zeros(96),
            features['texture'],
            features['geometric']
        ])

        return combined

    def extract_texture_features(self, gray_img):
        """Extract texture features using local binary patterns approach"""
        # Simple texture analysis
        h, w = gray_img.shape
        features = []

        # Edge density
        edges = cv2.Canny(gray_img, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        features.append(edge_density)

        # Gradient magnitude statistics
        grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        features.extend([
            np.mean(magnitude),
            np.std(magnitude),
            np.median(magnitude)
        ])

        # Intensity statistics
        features.extend([
            np.mean(gray_img),
            np.std(gray_img),
            np.median(gray_img)
        ])

        return np.array(features)

    def extract_geometric_features(self, contour, img_shape):
        """Extract geometric features from contour"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h) if h > 0 else 0
        extent = area / (w * h) if (w * h) > 0 else 0

        # Convex hull
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        # Circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

        # Relative size
        img_area = img_shape[0] * img_shape[1]
        relative_area = area / img_area

        return np.array([
            aspect_ratio, extent, solidity, circularity, relative_area,
            w / img_shape[1], h / img_shape[0],  # Relative width/height
            area, perimeter, len(contour)  # Absolute measures
        ])

    def extract_template_features(self):
        """Extract features from all loaded templates"""
        print("🔍 Extracting features from templates...")

        for comp_type, templates in self.templates.items():
            for template_data in templates:
                if template_data['features'] is None:
                    features = self.extract_visual_features(template_data['image'])
                    template_data['features'] = features

        print("✅ Template features extracted")

    def find_candidate_regions(self, img):
        """Find potential component regions using color-based segmentation"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Create combined mask for all relevant colors
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for color_name, (lower, upper) in self.color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        img_area = img.shape[0] * img.shape[1]

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 0.0005 * img_area or area > 0.1 * img_area:  # Size filter
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Extract region with some padding
            padding = max(5, min(w, h) // 4)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img.shape[1], x + w + padding)
            y2 = min(img.shape[0], y + h + padding)

            region = img[y1:y2, x1:x2]

            candidates.append({
                'region': region,
                'bbox': (x, y, w, h),
                'original_bbox': (x1, y1, x2 - x1, y2 - y1),
                'contour': contour
            })

        return candidates

    def match_template(self, candidate_region):
        """Match a candidate region against all templates"""
        if not any(len(templates) > 0 for templates in self.templates.values()):
            return None, 0.0

        # Extract features from candidate
        candidate_features = self.extract_visual_features(candidate_region)

        best_match = None
        best_score = 0.0
        best_type = None

        for comp_type, templates in self.templates.items():
            for template_data in templates:
                if template_data['features'] is None:
                    continue

                # Calculate similarity
                template_features = template_data['features']

                # Cosine similarity
                similarity = cosine_similarity([candidate_features], [template_features])[0][0]

                # Additional template matching for visual confirmation
                template_img = template_data['image']

                # Resize candidate to match template size for template matching
                template_h, template_w = template_img.shape[:2]
                candidate_resized = cv2.resize(candidate_region, (template_w, template_h))

                # Template matching
                if len(template_img.shape) == 3:
                    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
                    candidate_gray = cv2.cvtColor(candidate_resized, cv2.COLOR_BGR2GRAY)
                else:
                    template_gray = template_img
                    candidate_gray = candidate_resized

                # Normalized cross correlation
                ncc = cv2.matchTemplate(candidate_gray, template_gray, cv2.TM_CCOEFF_NORMED)[0][0]

                # Combined score (weighted)
                combined_score = 0.7 * similarity + 0.3 * max(0, ncc)

                if combined_score > best_score:
                    best_score = combined_score
                    best_match = template_data
                    best_type = comp_type

        return best_type, best_score

    def detect_components(self, img_path, confidence_threshold=0.6):
        """Main detection function using template matching"""
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")

        # Check if we have templates
        if not any(len(templates) > 0 for templates in self.templates.values()):
            print("⚠️  No templates loaded! Please add templates first.")
            print("💡 Use extract_template_from_image() or load templates from directory")
            return img, {}, []

        # Find candidate regions
        candidates = self.find_candidate_regions(img)
        print(f"🔍 Found {len(candidates)} candidate regions")

        # Initialize counters
        counts = {
            "browser": 0,
            "load_balancer": 0,
            "servers": 0,
            "database": {
                "type": "Database",
                "count": 0
            },
            "nas": {
                "type": "NAS Storage",
                "count": 0
            },
            "agents": {
                "count": 0
            }
        }

        detections = []

        # Match each candidate against templates
        for i, candidate in enumerate(candidates):
            comp_type, confidence = self.match_template(candidate['region'])

            if comp_type and confidence >= confidence_threshold:
                x, y, w, h = candidate['bbox']

                # Update counts
                if comp_type == 'server':
                    counts['servers'] += 1
                    label = f"Server ({confidence:.2f})"
                    color = (0, 255, 0)
                elif comp_type == 'database':
                    counts['database']['count'] += 1
                    label = f"Database ({confidence:.2f})"
                    color = (0, 140, 255)
                elif comp_type == 'load_balancer':
                    counts['load_balancer'] += 1
                    label = f"Load Balancer ({confidence:.2f})"
                    color = (255, 0, 255)
                elif comp_type == 'nas':
                    counts['nas']['count'] += 1
                    label = f"NAS ({confidence:.2f})"
                    color = (255, 255, 0)
                else:
                    continue

                # Draw detection
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                detections.append({
                    'type': comp_type,
                    'confidence': confidence,
                    'bbox': (x, y, w, h)
                })

                print(f"✅ {comp_type} detected at ({x},{y}) confidence: {confidence:.3f}")

        return img, counts, detections

    def save_results(self, img, counts, output_prefix="output"):
        """Save detection results"""
        # Save annotated image
        cv2.imwrite(f"{output_prefix}_detected.jpg", img)

        # Save YAML output
        yaml_output = {"components": counts}
        with open(f"{output_prefix}.yaml", "w") as f:
            yaml.dump(yaml_output, f, default_flow_style=False)

        # Save JSON output
        with open(f"{output_prefix}.json", "w") as f:
            json.dump(yaml_output, f, indent=2)

        return yaml_output

    def train_from_examples(self, training_images):
        """Train detector from example images with interactive labeling"""
        print("🎯 Interactive training mode")
        print("Instructions:")
        print("1. For each image, select components by clicking and dragging")
        print("2. You'll be asked to label each selection")
        print("3. Press 'q' when done with an image")

        for img_path in training_images:
            print(f"\n📸 Processing: {img_path}")

            # Extract templates interactively
            templates = self.extract_template_from_image(img_path, interactive=True)

            # Label each template
            for i, template in enumerate(templates):
                cv2.imshow(f'Label Template {i + 1}', cv2.resize(template, (200, 200)))
                print(f"Template {i + 1} - Enter type (server/database/nas/load_balancer) or 'skip':")
                label = input().strip().lower()
                cv2.destroyAllWindows()

                if label in self.templates:
                    # Save template
                    template_dir = self.template_dir / label
                    template_dir.mkdir(parents=True, exist_ok=True)

                    template_path = template_dir / f"template_{len(self.templates[label]):03d}.png"
                    cv2.imwrite(str(template_path), template)

                    # Add to templates
                    self.templates[label].append({
                        'image': template,
                        'path': str(template_path),
                        'features': None
                    })

                    print(f"✅ Added {label} template")

        # Extract features for all new templates
        self.extract_template_features()
        print("🎓 Training complete!")


# Utility functions
def test_image_loading(image_path):
    """Test if an image can be loaded and displayed properly"""
    print(f"🔍 Testing image loading for: {image_path}")

    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ File does not exist: {image_path}")
        return False

    # Try to load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return False

    print(f"✅ Image loaded successfully!")
    print(f"   Dimensions: {img.shape[1]}x{img.shape[0]} pixels")
    print(f"   Channels: {img.shape[2] if len(img.shape) == 3 else 1}")
    print(f"   Data type: {img.dtype}")

    # Try to display image
    try:
        window_name = "Image Test - Press any key to close"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Resize for display if too large
        display_img = img.copy()
        if max(img.shape[:2]) > 800:
            scale = 800 / max(img.shape[:2])
            new_w = int(img.shape[1] * scale)
            new_h = int(img.shape[0] * scale)
            display_img = cv2.resize(img, (new_w, new_h))
            print(f"   Resized for display: {new_w}x{new_h}")

        cv2.imshow(window_name, display_img)
        print("🖼️  Image displayed in window. Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("✅ Image display test successful!")
        return True

    except Exception as e:
        print(f"❌ Error displaying image: {e}")
        return False


# Usage functions
def setup_detector_with_training(training_images=None):
    """Set up detector with training data"""
    detector = TemplateBasedDetector()

    if not detector.template_dir.exists():
        detector.create_template_directory()

    if training_images:
        detector.train_from_examples(training_images)
    else:
        detector.load_templates()

    return detector


def detect_infrastructure(image_path, confidence_threshold=0.6):
    """Main detection function"""
    detector = TemplateBasedDetector()

    try:
        # Detect components
        annotated_img, counts, detections = detector.detect_components(
            image_path, confidence_threshold
        )

        # Save results
        results = detector.save_results(annotated_img, counts)

        print("\n🎯 Template-based Detection Results:")
        print("📊 Component Summary:")
        for comp_type, count in counts.items():
            if isinstance(count, dict):
                print(f"   {comp_type.title()}: {count['count']}")
            else:
                print(f"   {comp_type.title()}: {count}")

        print(f"\n📄 Total detections: {len(detections)}")
        print("📄 Output files:")
        print("   - output.yaml, output.json")
        print("   - output_detected.jpg")

        return results

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Example usage
    print("🚀 Template-Based Infrastructure Detector")
    print("\n📋 Setup Options:")
    print("1. Basic detection (requires existing templates)")
    print("2. Interactive training mode")
    print("3. Test image loading")

    choice = input("Choose option (1, 2, or 3): ").strip()

    if choice == "3":
        # Test image loading
        image_path = input("Enter image path to test: ").strip()
        if image_path:
            test_image_loading(image_path)

    elif choice == "2":
        # Training mode
        print("\n📚 Interactive Training Mode")
        print("You can enter a folder name to use all images inside, or comma-separated image paths.")

        training_images_input = input("Enter training image folder or image paths (comma-separated): ").strip()

        training_images = []
        if training_images_input:
            if os.path.isdir(training_images_input):
                training_images = get_images_from_folder(training_images_input)
                print(f"📂 Found {len(training_images)} images in folder '{training_images_input}'")
                print(f"DEBUG: training_images = {training_images}")
            else:
                training_images = [img.strip() for img in training_images_input.split(',') if img.strip()]

            if not training_images:
                print("❌ No images found. Please check the folder or file paths.")
                exit(1)

            print(f"\n🔍 Testing first image: {training_images[0]}")
            if not test_image_loading(training_images[0]):
                print("❌ Cannot load the first image. Please check the file path and format.")
                exit(1)

            print("\nImage test passed! Proceeding with interactive training...")
            input("Press Enter to continue...")

            detector = setup_detector_with_training(training_images)

            test_image = input("Enter test image path (or press Enter to skip): ").strip()
            if test_image:
                detect_infrastructure(test_image)
        else:
            print("❌ No training images provided.")
    else:
        # Basic detection
        image_path = input("Enter image path (default: img_yaml.png): ").strip()
        if not image_path:
            image_path = "img_yaml.png"

        # Test image first
        if not test_image_loading(image_path):
            print("❌ Cannot load the image. Please check the file path and format.")
            exit(1)

        detect_infrastructure(image_path)