import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from rfdetr import RFDETRMedium


def load_coco_annotation(annotation_path: Path):
    with open(annotation_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = {
        int(cat["id"]): cat["name"]
        for cat in coco.get("categories", [])
    }

    if not categories:
        raise ValueError(f"No categories found in: {annotation_path}")

    images = coco.get("images", [])
    if not images:
        raise ValueError(f"No images found in: {annotation_path}")

    return coco, images, categories



def resolve_image_path(split_dir: Path, dataset_dir: Path, file_name: str) -> Path:
    candidates = [
        split_dir / file_name,
        dataset_dir / file_name,
        Path(file_name),
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Image not found for file_name={file_name}. Tried:\n"
        + "\n".join(str(p) for p in candidates)
    )


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = [float(x) for x in box]
    return [
        x1,
        y1,
        max(0.0, x2 - x1),
        max(0.0, y2 - y1),
    ]

def convert_category_id(raw_class_id, categories, class_id_base: str):
    raw_class_id = int(raw_class_id)
    cat_ids = sorted(categories.keys())

    if class_id_base == "zero":
        if raw_class_id < 0 or raw_class_id >= len(cat_ids):
            return None
        return cat_ids[raw_class_id]

    if class_id_base == "one":
        if raw_class_id not in categories:
            return None
        return raw_class_id

    raise ValueError(f"Unknown class_id_base: {class_id_base}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument(
        "--class-id-base",
        choices=["zero", "one"],
        default="one",
        help="RF-DETR raw class_id base. For this AlphaDent COCO dataset, use one if raw class_id already equals COCO category_id.",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    split_dir = dataset_dir / args.split
    annotation_path = split_dir / "_annotations.coco.json"
    checkpoint_path = Path(args.checkpoint)
    out_path = Path(args.out)

    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {annotation_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("===== ENV CHECK =====")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
    else:
        raise RuntimeError("CUDA is not available.")

    print("===== LOAD COCO ANNOTATION =====")
    coco, images, categories = load_coco_annotation(annotation_path)

    expected_categories = {
        1: "Abrasion",
        2: "Caries",
    }

    if categories != expected_categories:
        raise ValueError(
            f"Category mismatch.\n"
            f"Expected: {expected_categories}\n"
            f"Found:    {categories}"
        )

    num_classes = args.num_classes if args.num_classes is not None else len(categories)

    print(f"dataset: {dataset_dir}")
    print(f"split: {args.split}")
    print(f"annotation: {annotation_path}")
    print(f"num images: {len(images)}")
    print(f"categories: {categories}")
    print(f"num_classes: {num_classes}")

    print("===== LOAD RF-DETR MODEL =====")
    print(f"checkpoint: {checkpoint_path}")

    model = RFDETRMedium(
        num_classes=num_classes,
        pretrain_weights=str(checkpoint_path),
    )

    raw_predictions = []
    raw_class_ids = []

    print("===== START RF-DETR TEST INFERENCE =====")

    for idx, img_info in enumerate(images, start=1):
        image_id = int(img_info["id"])
        file_name = img_info["file_name"]

        image_path = resolve_image_path(split_dir, dataset_dir, file_name)
        image = Image.open(image_path).convert("RGB")

        detections = model.predict(image, threshold=args.threshold)

        xyxy = detections.xyxy
        scores = detections.confidence
        class_ids = detections.class_id

        for box, score, raw_class_id in zip(xyxy, scores, class_ids):
            raw_class_id = int(raw_class_id)

            raw_class_ids.append(raw_class_id)

            raw_predictions.append(
                {
                    "image_id": image_id,
                    "raw_class_id": raw_class_id,
                    "bbox_xyxy": [float(x) for x in box],
                    "score": float(score),
                }
            )

        if idx % 20 == 0 or idx == len(images):
            print(
                f"processed {idx}/{len(images)}, "
                f"raw_predictions={len(raw_predictions)}"
            )

    print("===== CLASS ID BASE CHECK =====")
    raw_counter = Counter(raw_class_ids)
    print("raw class_id counts:", dict(sorted(raw_counter.items())))

    class_id_base = args.class_id_base
    print(f"selected class_id_base: {class_id_base}")

    predictions = []
    skipped = 0
    skipped_counter = Counter()
    skipped_examples = []

    for pred in raw_predictions:
        category_id = convert_category_id(
            raw_class_id=pred["raw_class_id"],
            categories=categories,
            class_id_base=class_id_base,
        )

        if category_id is None:
            skipped += 1
            skipped_counter[pred["raw_class_id"]] += 1

            if len(skipped_examples) < 5:
                skipped_examples.append(pred)

            continue

        predictions.append(
            {
                "image_id": pred["image_id"],
                "category_id": int(category_id),
                "bbox": xyxy_to_xywh(pred["bbox_xyxy"]),
                "score": float(pred["score"]),
            }
        )

    print("skipped invalid raw_class_id predictions:", skipped)
    print("skipped raw_class_id counts:", dict(sorted(skipped_counter.items())))
    if skipped_examples:
        print("skipped examples:", skipped_examples)

    saved_counter = Counter(p["category_id"] for p in predictions)
    print("saved category_id counts:", dict(sorted(saved_counter.items())))

    invalid_saved = [
        p for p in predictions
        if p["category_id"] not in categories
    ]

    if invalid_saved:
        raise ValueError(
            f"Saved predictions contain invalid category_id examples: "
            f"{invalid_saved[:5]}"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False)

    print("===== DONE =====")
    print(f"Saved predictions: {out_path}")
    print(f"num raw predictions: {len(raw_predictions)}")
    print(f"num saved predictions: {len(predictions)}")


if __name__ == "__main__":
    main()