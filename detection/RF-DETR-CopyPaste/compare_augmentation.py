import json
import cv2
import numpy as np
from pathlib import Path

# 경로
orig_dir = Path('/data/thdud23/datasets/AlphaDent_2class/train')
aug_dir = Path('/data/thdud23/datasets/AlphaDent_2class_seg_aug/train')
output_dir = Path('/data/thdud23/outputs/compare_seg_aug_final')
output_dir.mkdir(exist_ok=True)

# COCO 로드
with open(orig_dir / '_annotations.coco.json') as f:
    orig_coco = json.load(f)

with open(aug_dir / '_annotations.coco.json') as f:
    aug_coco = json.load(f)

# 이미지별 annotation 매핑
orig_anns = {}
for ann in orig_coco['annotations']:
    img_id = ann['image_id']
    if img_id not in orig_anns:
        orig_anns[img_id] = []
    orig_anns[img_id].append(ann)

aug_anns = {}
for ann in aug_coco['annotations']:
    img_id = ann['image_id']
    if img_id not in aug_anns:
        aug_anns[img_id] = []
    aug_anns[img_id].append(ann)

# Caries 증가한 이미지 찾기
increased = []
for img_info in orig_coco['images']:
    img_id = img_info['id']
    filename = img_info['file_name']
    
    orig_caries = sum(1 for a in orig_anns.get(img_id, []) if a['category_id'] == 2)
    aug_caries = sum(1 for a in aug_anns.get(img_id, []) if a['category_id'] == 2)
    
    if aug_caries > orig_caries:
        increased.append((filename, aug_caries - orig_caries))

increased.sort(key=lambda x: x[1], reverse=True)

print(f"Found {len(increased)} images with increased Caries")
print("Visualizing top 5...")

# 상위 5개 시각화
for filename, diff in increased[:5]:
    # 이미지 로드
    orig_img = cv2.imread(str(orig_dir / filename))
    aug_img = cv2.imread(str(aug_dir / filename))
    
    if orig_img is None or aug_img is None:
        continue
    
    # 해당 이미지 ID 찾기
    img_id = next(img['id'] for img in orig_coco['images'] if img['file_name'] == filename)
    
    # Bbox 그리기
    for ann in orig_anns.get(img_id, []):
        x, y, w, h = [int(v) for v in ann['bbox']]
        color = (0, 255, 0) if ann['category_id'] == 1 else (0, 0, 255)
        cv2.rectangle(orig_img, (x, y), (x+w, y+h), color, 2)
    
    for ann in aug_anns.get(img_id, []):
        x, y, w, h = [int(v) for v in ann['bbox']]
        color = (0, 255, 0) if ann['category_id'] == 1 else (0, 0, 255)
        cv2.rectangle(aug_img, (x, y), (x+w, y+h), color, 2)
    
    # 텍스트
    cv2.putText(orig_img, 'ORIGINAL', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(aug_img, f'AUGMENTED (+{diff})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 합치기
    combined = np.hstack([orig_img, aug_img])
    
    # 저장
    output_path = output_dir / filename
    cv2.imwrite(str(output_path), combined)
    print(f"Saved: {filename} (Caries +{diff})")

print(f"Done! Check: {output_dir}")
