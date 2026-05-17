import os
import json
import cv2
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import shutil

# ==================== CONFIG ====================
INPUT_BASE = "/data/thdud23/datasets/AlphaDent"
OUTPUT_BASE = "/data/thdud23/datasets/AlphaDent_2class_seg_aug"

# Copy-Paste 파라미터
SCALES = [0.8, 1.0, 1.2]
NUM_PASTE_PER_IMAGE = 4  # 증가
PASTE_PROB = 0.72  # 증가
MARGIN = 50

# 클래스 매핑
CLASS_MAPPING = {
    0: 1,  # Abrasion → class 1
    1: None,  # Filling → 제거
    2: None,  # Crown → 제거
    3: 2,  # Caries_1 → class 2
    4: 2,  # Caries_2 → class 2
    5: 2,  # Caries_3 → class 2
    6: 2,  # Caries_4 → class 2
    7: 2,  # Caries_5 → class 2
    8: 2,  # Caries_6 → class 2
}

NEW_CATEGORIES = [
    {"id": 1, "name": "Abrasion"},
    {"id": 2, "name": "Caries"}
]

# ==================== UTILS ====================

def polygon_to_mask(polygon, image_shape):
    """YOLO polygon을 binary mask로 변환"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    # Polygon 좌표 변환 (normalized → absolute)
    h, w = image_shape[:2]
    points = []
    for i in range(0, len(polygon), 2):
        x = int(polygon[i] * w)
        y = int(polygon[i+1] * h)
        points.append([x, y])
    
    points = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [points], 1)
    
    return mask


def create_teeth_mask(label_path, image_shape):
    """모든 치아 segmentation을 합쳐서 전체 치아 영역 마스크 생성"""
    teeth_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    if not label_path.exists():
        return teeth_mask
    
    h, w = image_shape[:2]
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            
            old_class = int(parts[0])
            new_class = CLASS_MAPPING.get(old_class)
            
            # Abrasion과 Caries만 (치아 영역)
            if new_class in [1, 2]:
                polygon = list(map(float, parts[1:]))
                mask = polygon_to_mask(polygon, image_shape)
                teeth_mask = np.maximum(teeth_mask, mask)
    
    return teeth_mask


def extract_object_with_mask(image, mask):
    """Mask를 사용해 객체만 추출"""
    # Bounding box 계산
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None, None, None
    
    x, y, w, h = cv2.boundingRect(coords)
    
    # 객체 영역 crop
    obj_image = image[y:y+h, x:x+w].copy()
    obj_mask = mask[y:y+h, x:x+w].copy()
    
    return obj_image, obj_mask, (x, y, w, h)


def polygon_to_bbox(polygon, width, height):
    """Polygon을 bbox로 변환"""
    xs = [polygon[i] * width for i in range(0, len(polygon), 2)]
    ys = [polygon[i] * height for i in range(1, len(polygon), 2)]
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def check_overlap(bbox1, bbox2, iou_threshold=0.1):
    """두 bbox가 겹치는지 확인"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return False
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    iou = intersection / float(area1 + area2 - intersection)
    
    return iou > iou_threshold


def find_valid_paste_position(image_shape, obj_shape, existing_bboxes, teeth_mask, margin=50):
    """치아 마스크 안에서 겹치지 않는 유효한 위치 찾기"""
    img_h, img_w = image_shape[:2]
    obj_h, obj_w = obj_shape[:2]
    
    # 붙일 수 있는 범위
    max_y = img_h - obj_h - margin
    max_x = img_w - obj_w - margin
    
    if max_y < margin or max_x < margin:
        return None
    
    for attempt in range(100):
        y = random.randint(margin, max_y)
        x = random.randint(margin, max_x)
        
        # 치아 마스크 안에 충분히 들어가는지 확인
        region_mask = teeth_mask[y:y+obj_h, x:x+obj_w]
        overlap_ratio = np.sum(region_mask) / (obj_h * obj_w)
        
        if overlap_ratio < 0.5:  # 50% 이상 치아 영역이어야 함
            continue
        
        # 새 bbox
        new_bbox = [x, y, obj_w, obj_h]
        
        # 기존 객체들과 겹치는지 확인
        overlap = False
        for existing_bbox in existing_bboxes:
            if check_overlap(new_bbox, existing_bbox):
                overlap = True
                break
        
        if not overlap:
            return (y, x)
    
    return None


def paste_object(target_image, obj_image, obj_mask, position):
    """객체를 target 이미지에 붙여넣기"""
    y, x = position
    obj_h, obj_w = obj_image.shape[:2]
    
    # 타겟 영역
    target_region = target_image[y:y+obj_h, x:x+obj_w]
    
    # Mask 적용하여 붙이기
    mask_3ch = cv2.merge([obj_mask, obj_mask, obj_mask])
    target_region[mask_3ch > 0] = obj_image[mask_3ch > 0]
    
    target_image[y:y+obj_h, x:x+obj_w] = target_region
    
    return target_image

# ==================== MAIN LOGIC ====================

def load_objects_from_dataset(input_base, split):
    """데이터셋에서 모든 객체 추출 (Caries만)"""
    objects = []
    
    image_dir = Path(input_base) / "images_split" / split
    label_dir = Path(input_base) / "labels_split" / split
    
    image_files = sorted(image_dir.glob("*.jpg"))
    
    print(f"Loading {split} objects...")
    
    for img_path in tqdm(image_files):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        h, w = image.shape[:2]
        label_path = label_dir / f"{img_path.stem}.txt"
        
        if not label_path.exists():
            continue
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                
                old_class = int(parts[0])
                new_class = CLASS_MAPPING.get(old_class)
                
                # Caries만 추출
                if new_class != 2:
                    continue
                
                polygon = list(map(float, parts[1:]))
                
                # Mask 생성
                mask = polygon_to_mask(polygon, image.shape)
                
                # 객체 추출
                result = extract_object_with_mask(image, mask)
                if result[0] is None:
                    continue
                
                obj_img, obj_mask, bbox = result
                
                # 저장
                objects.append({
                    'image': obj_img,
                    'mask': obj_mask,
                    'bbox': bbox,
                    'category_id': new_class
                })
    
    print(f"Loaded {len(objects)} Caries objects from {split}")
    return objects


def process_image_with_copypaste(image_path, label_path, caries_objects, apply_prob=0.6):
    """단일 이미지 처리 + Copy-Paste 적용"""
    
    # 이미지 로드
    image = cv2.imread(str(image_path))
    if image is None:
        return None, []
    
    h, w = image.shape[:2]
    
    # 치아 전체 마스크 생성 (중요!)
    teeth_mask = create_teeth_mask(label_path, image.shape)
    
    # 기존 annotation 로드
    annotations = []
    existing_bboxes = []
    
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                
                old_class = int(parts[0])
                new_class = CLASS_MAPPING.get(old_class)
                
                if new_class is None:
                    continue
                
                polygon = list(map(float, parts[1:]))
                bbox = polygon_to_bbox(polygon, w, h)
                
                annotations.append({
                    'category_id': new_class,
                    'bbox': bbox,
                    'area': bbox[2] * bbox[3]
                })
                
                existing_bboxes.append(bbox)
    
    # Copy-Paste 적용 여부 결정
    if random.random() > apply_prob or len(caries_objects) == 0:
        return image, annotations
    
    # Caries 객체 복사-붙여넣기
    num_paste = random.randint(1, NUM_PASTE_PER_IMAGE)
    
    for _ in range(num_paste):
        # 랜덤하게 Caries 선택
        src_obj = random.choice(caries_objects)
        
        # 랜덤 스케일 선택
        scale = random.choice(SCALES)
        
        # 스케일 적용
        obj_img = src_obj['image']
        obj_mask = src_obj['mask']

        if scale != 1.0:
            new_h = int(obj_img.shape[0] * scale)
            new_w = int(obj_img.shape[1] * scale)
            
            if new_h < 10 or new_w < 10:
                continue
            
            obj_img = cv2.resize(obj_img, (new_w, new_h))
            obj_mask = cv2.resize(obj_mask, (new_w, new_h))

        # 치아 마스크 안에서 유효한 위치 찾기 (수정!)
        position = find_valid_paste_position(
            image.shape, 
            obj_img.shape, 
            existing_bboxes,
            teeth_mask,  # 치아 마스크 전달
            margin=MARGIN
        )
        
        if position is None:
            continue
        
        # 붙여넣기
        image = paste_object(image, obj_img, obj_mask, position)
        
        # Annotation 추가
        y, x = position
        obj_h, obj_w = obj_img.shape[:2]
        new_bbox = [x, y, obj_w, obj_h]
        
        annotations.append({
            'category_id': 2,
            'bbox': new_bbox,
            'area': obj_w * obj_h
        })
        
        existing_bboxes.append(new_bbox)
    
    return image, annotations


def process_split(input_base, output_base, split, caries_objects):
    """전체 split 처리"""
    
    image_dir = Path(input_base) / "images_split" / split
    label_dir = Path(input_base) / "labels_split" / split
    output_split = Path(output_base) / split
    output_split.mkdir(parents=True, exist_ok=True)
    
    # COCO format
    coco = {
        'images': [],
        'annotations': [],
        'categories': NEW_CATEGORIES
    }
    
    image_id = 1
    ann_id = 1
    
    image_files = sorted(image_dir.glob("*.jpg"))
    
    print(f"\nProcessing {split} split with Copy-Paste...")
    
    for img_path in tqdm(image_files):
        label_path = label_dir / f"{img_path.stem}.txt"
        
        # Copy-Paste 적용
        augmented_image, annotations = process_image_with_copypaste(
            img_path, 
            label_path, 
            caries_objects,
            apply_prob=PASTE_PROB
        )
        
        if augmented_image is None or len(annotations) == 0:
            continue
        
        # 이미지 저장
        output_img_path = output_split / img_path.name
        cv2.imwrite(str(output_img_path), augmented_image)
        
        # 이미지 정보
        h, w = augmented_image.shape[:2]
        coco['images'].append({
            'id': image_id,
            'file_name': img_path.name,
            'width': w,
            'height': h
        })
        
        # Annotation 추가
        for ann in annotations:
            coco['annotations'].append({
                'id': ann_id,
                'image_id': image_id,
                'category_id': ann['category_id'],
                'bbox': ann['bbox'],
                'area': ann['area'],
                'iscrowd': 0
            })
            ann_id += 1
        
        image_id += 1
    
    # COCO JSON 저장
    output_ann = output_split / "_annotations.coco.json"
    with open(output_ann, 'w') as f:
        json.dump(coco, f, indent=2)
    
    # 통계
    num_abrasion = sum(1 for ann in coco['annotations'] if ann['category_id'] == 1)
    num_caries = sum(1 for ann in coco['annotations'] if ann['category_id'] == 2)
    
    print(f"✅ {split}: {len(coco['images'])} images, {len(coco['annotations'])} annotations")
    print(f"   Abrasion: {num_abrasion}, Caries: {num_caries}")

# ==================== MAIN ====================

def main():
    print("="*60)
    print("Copy-Paste with Teeth Segmentation Mask Constraint")
    print("="*60)
    print(f"Input: {INPUT_BASE}")
    print(f"Output: {OUTPUT_BASE}")
    print(f"Scales: {SCALES}")
    print(f"Paste prob: {PASTE_PROB}")
    print(f"Num paste per image: {NUM_PASTE_PER_IMAGE}")
    print("="*60)
    
    # Train에서 Caries 객체 추출
    caries_objects = load_objects_from_dataset(INPUT_BASE, 'train')
    
    # 각 split 처리
    for split in ['train', 'valid', 'test']:
        process_split(INPUT_BASE, OUTPUT_BASE, split, caries_objects)
    
    print("\n" + "="*60)
    print("✅ Segmentation-Aware Copy-Paste Complete!")
    print(f"Check: {OUTPUT_BASE}")
    print("="*60)

if __name__ == "__main__":
    main()