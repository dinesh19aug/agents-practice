import os

def generate_labels_file(data_dir, output_file="yolo8_icon_detector/component_labels.txt"):
    # List subdirectories in alphabetical order
    classes = sorted([
        name for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name))
    ])

    with open(output_file, "w") as f:
        for cls in classes:
            f.write(f"{cls}\n")

    print(f"✅ Generated {output_file} with {len(classes)} classes.")
    for idx, cls in enumerate(classes):
        print(f"  {idx}: {cls}")

if __name__ == "__main__":
    # Change this if your path is different
    generate_labels_file("yolo8_icon_detector/data")