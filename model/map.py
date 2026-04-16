import os
from collections import Counter

# Paths
LABELS_DIR = "labels"
CLASSES_FILE = "classes.txt"


def load_classes(classes_file: str) -> list[str]:
    if not os.path.exists(classes_file):
        raise FileNotFoundError(f"Could not find '{classes_file}'")

    with open(classes_file, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]

    if not classes:
        raise ValueError(f"'{classes_file}' is empty")

    return classes


def analyze_yolo_dataset(labels_dir: str, class_names: list[str]):
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"Could not find folder '{labels_dir}'")

    box_counts = Counter()    # total boxes per class
    image_counts = Counter()  # number of images containing each class

    txt_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]

    if not txt_files:
        raise ValueError(f"No .txt label files found in '{labels_dir}'")

    empty_files = 0
    invalid_lines = 0

    for filename in txt_files:
        filepath = os.path.join(labels_dir, filename)
        classes_in_this_image = set()

        with open(filepath, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            empty_files += 1
            continue

        for line in lines:
            parts = line.split()

            # YOLO format: class_id x_center y_center width height
            if len(parts) < 5:
                invalid_lines += 1
                continue

            try:
                class_id = int(parts[0])
            except ValueError:
                invalid_lines += 1
                continue

            if class_id < 0 or class_id >= len(class_names):
                invalid_lines += 1
                continue

            box_counts[class_id] += 1
            classes_in_this_image.add(class_id)

        for class_id in classes_in_this_image:
            image_counts[class_id] += 1

    return {
        "total_label_files": len(txt_files),
        "empty_files": empty_files,
        "invalid_lines": invalid_lines,
        "box_counts": box_counts,
        "image_counts": image_counts,
    }


def print_report(result: dict, class_names: list[str]):
    total_files = result["total_label_files"]
    empty_files = result["empty_files"]
    invalid_lines = result["invalid_lines"]
    box_counts = result["box_counts"]
    image_counts = result["image_counts"]

    total_boxes = sum(box_counts.values())

    print("\n=== YOLO DATASET REPORT ===")
    print(f"Total label files: {total_files}")
    print(f"Empty label files: {empty_files}")
    print(f"Invalid lines: {invalid_lines}")
    print(f"Total bounding boxes: {total_boxes}")

    print("\n--- Bounding boxes per class ---")
    for class_id, class_name in enumerate(class_names):
        count = box_counts[class_id]
        pct = (count / total_boxes * 100) if total_boxes > 0 else 0
        print(f"{class_name}: {count} boxes ({pct:.2f}%)")

    print("\n--- Images containing each class ---")
    for class_id, class_name in enumerate(class_names):
        count = image_counts[class_id]
        pct = (count / total_files * 100) if total_files > 0 else 0
        print(f"{class_name}: {count} images ({pct:.2f}%)")

    print("\n--- Balance check (by boxes) ---")
    counts = [box_counts[i] for i in range(len(class_names))]
    nonzero_counts = [c for c in counts if c > 0]

    if not nonzero_counts:
        print("No valid annotations found.")
        return

    min_count = min(nonzero_counts)
    max_count = max(nonzero_counts)

    if min_count == 0:
        print("At least one class has zero annotations.")
        return

    imbalance_ratio = max_count / min_count

    print(f"Min boxes in a class: {min_count}")
    print(f"Max boxes in a class: {max_count}")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}")

    if imbalance_ratio <= 1.15:
        print("Dataset looks well balanced.")
    elif imbalance_ratio <= 1.5:
        print("Dataset is slightly imbalanced.")
    else:
        print("Dataset is strongly imbalanced and may bias training.")


def main():
    try:
        class_names = load_classes(CLASSES_FILE)
        result = analyze_yolo_dataset(LABELS_DIR, class_names)
        print_report(result, class_names)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()