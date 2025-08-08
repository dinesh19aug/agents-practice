import cv2
import numpy as np
import yaml
import json
from collections import defaultdict
import math


class InfrastructureDetector:
    def __init__(self):
        # More precise color ranges for better detection
        self.color_ranges = {
            'blue': {
                'lower': np.array([100, 80, 80]),  # More restrictive blue
                'upper': np.array([130, 255, 255]),
                'components': ['servers', 'database']
            },
            'light_blue': {
                'lower': np.array([85, 40, 100]),  # Light blue/cyan for NAS
                'upper': np.array([105, 255, 255]),
                'components': ['nas', 'storage']
            },
            'purple': {
                'lower': np.array([130, 60, 60]),  # Purple for load balancers
                'upper': np.array([160, 255, 255]),
                'components': ['load_balancer']
            }
        }

        # More specific classification rules
        self.classification_rules = {
            'server': {
                'min_area_ratio': 0.0008,  # Minimum 0.08% of image
                'max_area_ratio': 0.05,  # Maximum 5% of image
                'aspect_ratio_range': (0.8, 1.3),  # Nearly square
                'min_extent': 0.65,  # Should fill most of bounding box
                'shape_type': 'cube',
                'color': 'blue'
            },
            'database': {
                'min_area_ratio': 0.002,  # Larger than servers typically
                'max_area_ratio': 0.08,
                'aspect_ratio_range': (1.3, 2.5),  # Cylindrical/elliptical
                'min_extent': 0.6,
                'shape_type': 'cylinder',
                'color': 'blue'
            },
            'nas': {
                'min_area_ratio': 0.0005,
                'max_area_ratio': 0.04,
                'aspect_ratio_range': (0.9, 1.4),  # Squarish but can be slightly rectangular
                'min_extent': 0.6,
                'shape_type': 'storage',
                'color': 'light_blue'
            },
            'load_balancer': {
                'min_area_ratio': 0.001,
                'max_area_ratio': 0.06,
                'aspect_ratio_range': (1.8, 4.0),  # More rectangular
                'min_extent': 0.7,
                'shape_type': 'rectangle',
                'color': 'purple'
            }
        }

    def preprocess_image(self, img_path):
        """Load and preprocess the image"""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")

        self.original_img = img.copy()
        self.img_height, self.img_width = img.shape[:2]
        self.img_area = self.img_width * self.img_height

        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return img, hsv

    def create_combined_mask(self, hsv):
        """Create a combined mask for all relevant colors"""
        combined_mask = np.zeros((self.img_height, self.img_width), dtype=np.uint8)

        for color_name, color_info in self.color_ranges.items():
            mask = cv2.inRange(hsv, color_info['lower'], color_info['upper'])
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Clean up noise with adaptive morphological operations
        kernel_size = max(3, min(9, int(min(self.img_width, self.img_height) / 80)))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Remove noise
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        # Fill gaps
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        return combined_mask

    def get_color_at_position(self, hsv, x, y, w, h):
        """Determine the dominant color in a region"""
        # Sample the center region of the bounding box
        center_x, center_y = x + w // 2, y + h // 2
        sample_size = min(w // 3, h // 3, 10)  # Sample a small region around center

        x1 = max(0, center_x - sample_size)
        y1 = max(0, center_y - sample_size)
        x2 = min(self.img_width, center_x + sample_size)
        y2 = min(self.img_height, center_y + sample_size)

        region = hsv[y1:y2, x1:x2]

        # Check which color range this region belongs to
        best_match = None
        max_pixels = 0

        for color_name, color_info in self.color_ranges.items():
            mask = cv2.inRange(region, color_info['lower'], color_info['upper'])
            pixel_count = cv2.countNonZero(mask)

            if pixel_count > max_pixels:
                max_pixels = pixel_count
                best_match = color_name

        return best_match if max_pixels > 10 else None

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

        # Ellipse detection
        ellipse_detected = False
        ellipse_params = None
        ellipse_aspect_ratio = 0

        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                (center, axes, angle) = ellipse
                major, minor = max(axes), min(axes)
                ellipse_aspect_ratio = major / minor if minor > 0 else 0

                if 1.1 < ellipse_aspect_ratio < 3.5:
                    ellipse_detected = True
                    ellipse_params = ellipse
            except:
                pass

        return {
            'area': area,
            'relative_area': relative_area,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'extent': extent,
            'circularity': circularity,
            'bbox': (x, y, w, h),
            'perimeter': perimeter,
            'width': w,
            'height': h,
            'ellipse_detected': ellipse_detected,
            'ellipse_params': ellipse_params,
            'ellipse_aspect_ratio': ellipse_aspect_ratio
        }

    def classify_component(self, features, color_name):
        """Classify component with more rigorous scoring"""
        if not color_name:
            return None, 0.0

        best_match = None
        best_score = 0.0

        for comp_type, rules in self.classification_rules.items():
            # Skip if color doesn't match
            if rules['color'] != color_name:
                continue

            score = 0.0
            criteria_met = 0
            total_criteria = 5

            # 1. Area check (mandatory)
            if rules['min_area_ratio'] <= features['relative_area'] <= rules['max_area_ratio']:
                score += 1.0
                criteria_met += 1

            # 2. Aspect ratio check (mandatory)
            ar_min, ar_max = rules['aspect_ratio_range']
            if ar_min <= features['aspect_ratio'] <= ar_max:
                score += 1.0
                criteria_met += 1

            # 3. Extent check (how well it fills bounding box)
            if features['extent'] >= rules['min_extent']:
                score += 1.0
                criteria_met += 1

            # 4. Shape-specific bonus scoring
            if rules['shape_type'] == 'cube':
                # Prefer square-like shapes with good solidity
                if 0.85 <= features['aspect_ratio'] <= 1.15 and features['solidity'] > 0.85:
                    score += 1.0
                    criteria_met += 1
            elif rules['shape_type'] == 'cylinder':
                # Prefer elongated shapes (databases)
                if features['aspect_ratio'] > 1.2 and features['solidity'] > 0.75:
                    score += 0.5
                    criteria_met += 1
                if features.get('ellipse_detected'):
                    score += 1
                    criteria_met += 1
            elif rules['shape_type'] == 'storage':
                # NAS can be square or slightly rectangular
                if 0.8 <= features['aspect_ratio'] <= 1.3 and features['extent'] > 0.6:
                    score += 1.0
                    criteria_met += 1
            elif rules['shape_type'] == 'rectangle':
                # Load balancers are more rectangular
                if features['aspect_ratio'] > 1.5 and features['extent'] > 0.75:
                    score += 1.0
                    criteria_met += 1

            # 5. Size consistency bonus
            # Give bonus for components that are reasonable size
            if 0.002 <= features['relative_area'] <= 0.05:
                score += 1.0
                criteria_met += 1

            confidence = score / total_criteria

            # Require at least 60% confidence and minimum criteria
            if confidence >= 0.6 and criteria_met >= 3:
                if confidence > best_score:
                    best_score = confidence
                    best_match = comp_type

        return best_match, best_score

    def remove_overlapping_detections(self, detections):
        """Remove overlapping detections, keeping the one with higher confidence"""
        if len(detections) <= 1:
            return detections

        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)

        filtered = []
        for current in detections:
            x1, y1, w1, h1 = current['bbox']
            overlap_found = False

            for existing in filtered:
                x2, y2, w2, h2 = existing['bbox']

                # Calculate overlap
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y

                area1 = w1 * h1
                area2 = w2 * h2

                # If overlap is more than 30% of either box, consider it duplicate
                if overlap_area > 0.3 * min(area1, area2):
                    overlap_found = True
                    break

            if not overlap_found:
                filtered.append(current)

        return filtered

    def detect_components(self, img_path):
        """Main detection function with improved accuracy"""
        img, hsv = self.preprocess_image(img_path)

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

        # Create combined mask for all components
        mask = self.create_combined_mask(hsv)

        # Save debug mask
        cv2.imwrite("debug_mask.jpg", mask)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        all_detections = []

        for contour in contours:
            features = self.extract_shape_features(contour)

            # Filter out very small objects
            if features['relative_area'] < 0.0005:  # Less than 0.05% of image
                continue

            x, y, w, h = features['bbox']

            # Determine color at this position
            color_name = self.get_color_at_position(hsv, x, y, w, h)

            # Classify the component
            comp_type, confidence = self.classify_component(features, color_name)

            if comp_type and confidence >= 0.6:  # Higher confidence threshold
                all_detections.append({
                    'type': comp_type,
                    'confidence': confidence,
                    'bbox': (x, y, w, h),
                    'features': features,
                    'color': color_name
                })

                print(
                    f"[CANDIDATE] {comp_type} at ({x},{y}) size=({w}x{h}) confidence={confidence:.2f} color={color_name}")

        # Remove overlapping detections
        filtered_detections = self.remove_overlapping_detections(all_detections)

        # Update counts and draw annotations
        for detection in filtered_detections:
            comp_type = detection['type']
            confidence = detection['confidence']
            x, y, w, h = detection['bbox']

            if comp_type == 'server':
                counts['servers'] += 1
                label = f"Server ({confidence:.2f})"
                color = (0, 255, 0)  # Green
            elif comp_type == 'database':
                counts['database']['count'] += 1
                label = f"Database ({confidence:.2f})"
                color = (0, 140, 255)  # Orange
            elif comp_type == 'load_balancer':
                counts['load_balancer'] += 1
                label = f"Load Balancer ({confidence:.2f})"
                color = (255, 0, 255)  # Magenta
            elif comp_type == 'nas':
                counts['nas']['count'] += 1
                label = f"NAS ({confidence:.2f})"
                color = (255, 255, 0)  # Cyan
            else:
                continue

            # Draw bounding box and label
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            print(f"[FINAL] {comp_type} at ({x},{y}) size=({w}x{h}) confidence={confidence:.2f}")

        return img, counts, filtered_detections

    def save_results(self, img, counts, output_prefix="output"):
        """Save detection results"""
        # Save annotated image
        cv2.imwrite(f"{output_prefix}_annotated.jpg", img)

        # Save YAML output
        yaml_output = {"components": counts}
        with open(f"{output_prefix}.yaml", "w") as f:
            yaml.dump(yaml_output, f, default_flow_style=False)

        # Save JSON output
        with open(f"{output_prefix}.json", "w") as f:
            json.dump(yaml_output, f, indent=2)

        return yaml_output


# Usage function
def main(image_path="img_yaml_2.png"):
    detector = InfrastructureDetector()

    try:
        # Detect components
        annotated_img, counts, components = detector.detect_components(image_path)

        # Save results
        results = detector.save_results(annotated_img, counts)

        print("\n✅ Detection complete!")
        print("📊 Component Summary:")
        print(f"   Servers: {counts['servers']}")
        print(f"   Databases: {counts['database']['count']}")
        print(f"   Load Balancers: {counts['load_balancer']}")
        print(f"   NAS Storage: {counts['nas']['count']}")

        print(f"\n📄 Total components detected: {len(components)}")
        print("📄 Output files:")
        print("   - output.yaml, output.json")
        print("   - output_annotated.jpg")
        print("   - debug_mask.jpg")

        return results

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()