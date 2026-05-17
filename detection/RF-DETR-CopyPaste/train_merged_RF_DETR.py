import argparse
import json
from pathlib import Path

import torch
from rfdetr import RFDETRMedium


EXPECTED_CATEGORIES = {
    1: "Abrasion",
    2: "Caries",
}


def check_dataset(dataset_dir: Path):
    print("===== CHECK DATASET =====")
    print(f"dataset_dir: {dataset_dir}")

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    for split in ["train", "valid", "test"]:
        split_dir = dataset_dir / split
        ann_path = split_dir / "_annotations.coco.json"

        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        if not ann_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {ann_path}")

        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        categories = {
            int(cat["id"]): cat["name"]
            for cat in coco.get("categories", [])
        }

        if categories != EXPECTED_CATEGORIES:
            raise ValueError(
                f"[{split}] category mismatch.\n"
                f"Expected: {EXPECTED_CATEGORIES}\n"
                f"Found:    {categories}"
            )
        
        bad_annotations = [
            ann for ann in coco.get("annotations", [])
            if int(ann["category_id"]) not in EXPECTED_CATEGORIES
        ]

        if bad_annotations:
            raise ValueError(
                f"[{split}] invalid annotation category_id found. "
                f"Examples: {bad_annotations[:5]}"
            )

        num_images = len(coco.get("images", []))
        num_annotations = len(coco.get("annotations", []))

        print(
            f"[{split}] images={num_images}, "
            f"annotations={num_annotations}, "
            f"categories={categories}"
        )

    print("Dataset check passed.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--resolution", type=int, default=672)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--eval-interval", type=int, default=1)

    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("===== ENV CHECK =====")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
    else:
        raise RuntimeError("CUDA is not available. Run this script on a GPU compute node.")

    check_dataset(dataset_dir)

    if args.num_classes != len(EXPECTED_CATEGORIES):
        raise ValueError(
            f"num_classes mismatch. "
            f"args.num_classes={args.num_classes}, "
            f"expected={len(EXPECTED_CATEGORIES)}, "
            f"categories={EXPECTED_CATEGORIES}"
        )

    print("===== TRAIN CONFIG =====")
    print("model: RF-DETR Base")
    print(f"dataset: {dataset_dir}")
    print(f"output_dir: {output_dir}")
    print(f"num_classes: {args.num_classes}")
    print(f"epochs: {args.epochs}")
    print(f"batch_size: {args.batch_size}")
    print(f"grad_accum_steps: {args.grad_accum_steps}")
    print(f"effective_batch_size: {args.batch_size * args.grad_accum_steps}")
    print(f"lr: {args.lr}")
    print(f"lr_encoder: {args.lr_encoder}")
    print(f"weight_decay: {args.weight_decay}")
    print(f"resolution: {args.resolution}")
    print(f"seed: {args.seed}")

    model = RFDETRMedium(num_classes=args.num_classes)

    model.train(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        weight_decay=args.weight_decay,
        resolution=args.resolution,
        device="cuda",
        seed=args.seed,
        use_ema=True,
        checkpoint_interval=args.checkpoint_interval,
        eval_interval=args.eval_interval,
        log_per_class_metrics=True,
        progress_bar=True,
        tensorboard=False,
        wandb=False,
    )

    print("===== TRAINING FINISHED =====")

    best_ckpt = output_dir / "checkpoint_best_total.pth"
    if best_ckpt.exists():
        print(f"SUCCESS: best checkpoint found: {best_ckpt}")
    else:
        print(f"[WARN] checkpoint_best_total.pth not found in {output_dir}")
        print("Output files:")
        for p in sorted(output_dir.glob("*")):
            print(" -", p)


if __name__ == "__main__":
    main()