"""
PKLot Dataset Tester
====================
Extracts, validates, and tests the PKLot YOLOv8 dataset.
Tests include: structure validation, label integrity, sample visualization, and model evaluation.
"""

import zipfile
import os
import sys
import yaml
import random
import time
from pathlib import Path
from collections import Counter

# ── Configuration ──────────────────────────────────────────────────
ZIP_PATH = Path(r"C:\Users\Dell\Desktop\IEEE-BHTC\PKLot raw.v6i.yolov8.zip")
EXTRACT_DIR = Path(r"C:\Users\Dell\Desktop\IEEE-BHTC\data\pklot_dataset")
RESULTS_DIR = Path(r"C:\Users\Dell\Desktop\IEEE-BHTC\data\pklot_test_results")


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_pass(msg):
    print(f"  [PASS] {msg}")


def print_fail(msg):
    print(f"  [FAIL] {msg}")


def print_info(msg):
    print(f"  [INFO] {msg}")


# ── Step 1: Extract Dataset ───────────────────────────────────────
def extract_dataset():
    print_header("STEP 1: Extracting Dataset")
    if not ZIP_PATH.exists():
        print_fail(f"ZIP file not found: {ZIP_PATH}")
        sys.exit(1)

    print_info(f"ZIP size: {ZIP_PATH.stat().st_size / (1024**2):.1f} MB")

    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.iterdir()):
        print_info("Dataset already extracted. Skipping extraction.")
        return

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    print_info("Extracting... (this may take a minute)")
    start = time.time()
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(EXTRACT_DIR)
    elapsed = time.time() - start
    print_pass(f"Extracted in {elapsed:.1f}s")


# ── Step 2: Validate Structure ────────────────────────────────────
def validate_structure():
    print_header("STEP 2: Validating Dataset Structure")
    errors = []

    # Check data.yaml
    yaml_path = EXTRACT_DIR / "data.yaml"
    if not yaml_path.exists():
        print_fail("data.yaml not found!")
        errors.append("Missing data.yaml")
        return None, errors

    with open(yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)

    print_info(f"Classes ({data_cfg.get('nc', '?')}): {data_cfg.get('names', [])}")
    print_pass("data.yaml found and parsed")

    # Check splits
    expected_splits = ['train', 'valid', 'test']
    split_stats = {}
    for split in expected_splits:
        img_dir = EXTRACT_DIR / split / "images"
        lbl_dir = EXTRACT_DIR / split / "labels"

        if not img_dir.exists():
            print_fail(f"Missing {split}/images directory")
            errors.append(f"Missing {split}/images")
            continue
        if not lbl_dir.exists():
            print_fail(f"Missing {split}/labels directory")
            errors.append(f"Missing {split}/labels")
            continue

        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        labels = list(lbl_dir.glob("*.txt"))
        split_stats[split] = {"images": len(images), "labels": len(labels)}
        print_pass(f"{split}: {len(images)} images, {len(labels)} labels")

        # Check for mismatches
        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}
        missing_labels = img_stems - lbl_stems
        orphan_labels = lbl_stems - img_stems

        if missing_labels:
            print_info(f"  {split}: {len(missing_labels)} images without labels (may be background/negatives)")
        if orphan_labels:
            print_fail(f"  {split}: {len(orphan_labels)} orphan labels (no matching image)")
            errors.append(f"{split}: orphan labels")

    return data_cfg, errors


# ── Step 3: Validate Labels ──────────────────────────────────────
def validate_labels(data_cfg):
    print_header("STEP 3: Validating Label Integrity")
    nc = data_cfg.get('nc', 2)
    errors = []
    class_dist = Counter()
    total_boxes = 0
    bad_files = 0

    for split in ['train', 'valid', 'test']:
        lbl_dir = EXTRACT_DIR / split / "labels"
        if not lbl_dir.exists():
            continue

        for lbl_file in lbl_dir.glob("*.txt"):
            try:
                with open(lbl_file, 'r') as f:
                    lines = f.readlines()
                for line_num, line in enumerate(lines, 1):
                    parts = line.strip().split()
                    if len(parts) == 0:
                        continue
                    if len(parts) != 5:
                        errors.append(f"{lbl_file.name}:{line_num} - expected 5 values, got {len(parts)}")
                        bad_files += 1
                        continue

                    cls_id = int(parts[0])
                    x_c, y_c, w, h = map(float, parts[1:])

                    if cls_id < 0 or cls_id >= nc:
                        errors.append(f"{lbl_file.name}:{line_num} - invalid class {cls_id} (nc={nc})")
                        bad_files += 1

                    # Validate bbox ranges
                    for val, name in [(x_c, 'x'), (y_c, 'y'), (w, 'w'), (h, 'h')]:
                        if val < 0 or val > 1:
                            errors.append(f"{lbl_file.name}:{line_num} - {name}={val} out of [0,1]")
                            bad_files += 1

                    class_dist[cls_id] += 1
                    total_boxes += 1
            except Exception as e:
                errors.append(f"{lbl_file.name} - parse error: {e}")
                bad_files += 1

    print_info(f"Total bounding boxes: {total_boxes:,}")
    print_info(f"Class distribution:")
    class_names = data_cfg.get('names', [])
    for cls_id, count in sorted(class_dist.items()):
        name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        pct = (count / total_boxes * 100) if total_boxes > 0 else 0
        print_info(f"  {cls_id} ({name}): {count:,} boxes ({pct:.1f}%)")

    if bad_files == 0:
        print_pass(f"All labels valid ({total_boxes:,} boxes checked)")
    else:
        print_fail(f"{bad_files} label issues found")
        for e in errors[:10]:
            print(f"    -> {e}")

    return errors


# ── Step 4: Check Sample Images ──────────────────────────────────
def check_sample_images():
    print_header("STEP 4: Checking Sample Images")
    try:
        import cv2
    except ImportError:
        print_info("OpenCV not available, skipping image checks")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    img_dir = EXTRACT_DIR / "test" / "images"
    lbl_dir = EXTRACT_DIR / "test" / "labels"

    if not img_dir.exists():
        print_fail("test/images not found")
        return

    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    if not images:
        print_fail("No test images found")
        return

    samples = random.sample(images, min(5, len(images)))
    sizes = []
    colors = [(0, 255, 0), (0, 0, 255)]  # green=empty, red=occupied
    class_names = ['empty', 'occupied']

    for img_path in samples:
        img = cv2.imread(str(img_path))
        if img is None:
            print_fail(f"Cannot read image: {img_path.name}")
            continue

        h, w = img.shape[:2]
        sizes.append((w, h))
        print_pass(f"{img_path.name}: {w}x{h}, {img.shape[2]}ch")

        # Draw bboxes if label exists
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:])
                    x1 = int((xc - bw / 2) * w)
                    y1 = int((yc - bh / 2) * h)
                    x2 = int((xc + bw / 2) * w)
                    y2 = int((yc + bh / 2) * h)
                    color = colors[cls_id] if cls_id < len(colors) else (255, 255, 0)
                    label = class_names[cls_id] if cls_id < len(class_names) else f"cls{cls_id}"
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        out_path = RESULTS_DIR / f"sample_{img_path.stem[:30]}.jpg"
        cv2.imwrite(str(out_path), img)

    unique_sizes = set(sizes)
    if len(unique_sizes) == 1:
        print_pass(f"All sample images have consistent size: {unique_sizes.pop()}")
    else:
        print_info(f"Multiple image sizes detected: {unique_sizes}")

    print_pass(f"Sample visualizations saved to: {RESULTS_DIR}")


# ── Step 5: Quick Model Test ─────────────────────────────────────
def quick_model_test(data_cfg):
    print_header("STEP 5: Quick YOLOv8 Model Validation Run")
    try:
        from ultralytics import YOLO
    except ImportError:
        print_fail("ultralytics not installed. Run: pip install ultralytics")
        return

    # Fix data.yaml paths to be absolute
    fixed_yaml = EXTRACT_DIR / "data_fixed.yaml"
    fixed_cfg = dict(data_cfg)
    fixed_cfg['train'] = str(EXTRACT_DIR / "train" / "images")
    fixed_cfg['val'] = str(EXTRACT_DIR / "valid" / "images")
    fixed_cfg['test'] = str(EXTRACT_DIR / "test" / "images")

    with open(fixed_yaml, 'w') as f:
        yaml.dump(fixed_cfg, f)

    print_info(f"Fixed data.yaml written to: {fixed_yaml}")

    # Run validation on pretrained yolov8n with the dataset
    # First, train for just 1 epoch to verify data loading works
    print_info("Running 1-epoch training sanity check (verifies data loading)...")
    model = YOLO("yolov8n.pt")

    try:
        results = model.train(
            data=str(fixed_yaml),
            epochs=1,
            imgsz=320,
            batch=8,
            workers=0,
            device='cpu',
            project=str(RESULTS_DIR),
            name='sanity_check',
            exist_ok=True,
            verbose=False,
        )
        print_pass("Training sanity check passed! Data loads correctly.")
        print_info(f"Training results saved to: {RESULTS_DIR / 'sanity_check'}")
    except Exception as e:
        print_fail(f"Training sanity check failed: {e}")
        return

    # Run validation
    print_info("Running validation on the test split...")
    try:
        val_results = model.val(
            data=str(fixed_yaml),
            split='test',
            imgsz=320,
            batch=8,
            workers=0,
            device='cpu',
            project=str(RESULTS_DIR),
            name='val_test',
            exist_ok=True,
            verbose=False,
        )
        print_pass("Validation completed successfully!")

        # Print metrics
        metrics = val_results
        print_info(f"  mAP50:    {metrics.box.map50:.4f}")
        print_info(f"  mAP50-95: {metrics.box.map:.4f}")
        print_info(f"  Precision: {metrics.box.mp:.4f}")
        print_info(f"  Recall:    {metrics.box.mr:.4f}")

    except Exception as e:
        print_fail(f"Validation failed: {e}")


# ── Step 6: Summary ──────────────────────────────────────────────
def print_summary(data_cfg, struct_errors, label_errors):
    print_header("TEST SUMMARY")
    total_errors = len(struct_errors) + len(label_errors)

    if total_errors == 0:
        print_pass("ALL TESTS PASSED!")
    else:
        print_fail(f"{total_errors} total issues found")

    print()
    print("  Dataset: PKLot Raw v6 (YOLOv8 format)")
    print(f"  Classes: {data_cfg.get('nc', '?')} - {data_cfg.get('names', [])}")
    print(f"  Structure errors: {len(struct_errors)}")
    print(f"  Label errors: {len(label_errors)}")
    print(f"  Results dir: {RESULTS_DIR}")
    print()


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  PKLot Dataset Tester - YOLOv8 Format")
    print("="*60)

    extract_dataset()
    data_cfg, struct_errors = validate_structure()
    if data_cfg is None:
        print_fail("Cannot continue without valid data.yaml")
        sys.exit(1)

    label_errors = validate_labels(data_cfg)
    check_sample_images()

    # Ask user if they want the model test (takes time)
    if "--full" in sys.argv:
        quick_model_test(data_cfg)
    else:
        print_header("STEP 5: Quick YOLOv8 Model Validation Run")
        print_info("Skipped. Run with --full flag to include model training/validation test")
        print_info("  python test_pklot_dataset.py --full")

    print_summary(data_cfg, struct_errors, label_errors)
