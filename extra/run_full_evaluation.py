import cv2
import numpy as np
from pathlib import Path
import re

# Configuration
TARGET_DIR = "coin_dataset/target_set2"
TEMPLATE_DIR = "coin_dataset/reference_set"
GT_FILE = "ground_truth2.txt"

COIN_VALUES = {
    "1cent": 0.01,
    "2cent": 0.02,
    "5cent": 0.05,
    "10cent": 0.10,
    "20cent": 0.20,
    "50cent": 0.50,
    "1euro": 1.00,
    "2euro": 2.00
}

COIN_NOMINAL_VALUE_CENTS = {
    "1cent": 1,
    "2cent": 2,
    "5cent": 5,
    "10cent": 10,
    "20cent": 20,
    "50cent": 50,
    "1euro": 100,
    "2euro": 200
}

# SIFT detector
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

def load_templates():
    templates = {}
    for path in Path(TEMPLATE_DIR).glob("*.jpg"):
        name = path.stem.lower().replace("_", "")
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        kp, des = sift.detectAndCompute(img, None)
        if des is not None:
            templates[name] = (kp, des)
    return templates

templates_db = load_templates()

def load_ground_truth():
    gt = {}
    with open(GT_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r'\s+', line)
            if len(parts) >= 3:
                img_name = parts[0]
                if not img_name.endswith(".jpg"):
                    img_name += ".jpg"
                val_cents = int(parts[1])
                coins_str = parts[2].split(",")
                coins_cents = []
                for c in coins_str:
                    c = c.strip()
                    if c:
                        if c == "50.5":
                            coins_cents.extend([50, 5])
                        elif c == "100.5":
                            coins_cents.extend([100, 5])
                        else:
                            try:
                                coins_cents.append(int(float(c)))
                            except:
                                pass
                gt[img_name] = (val_cents, coins_cents)
    return gt

gt_db = load_ground_truth()

def extract_roi(img, x, y, r):
    pad = int(r * 1.1)
    x1 = max(0, x - pad)
    x2 = min(img.shape[1], x + pad)
    y1 = max(0, y - pad)
    y2 = min(img.shape[0], y + pad)
    roi = img[y1:y2, x1:x2]
    if roi is None or roi.size == 0:
        return None
    return roi

def detect_circles(gray):
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=80,
        param1=120,
        param2=40,
        minRadius=50,
        maxRadius=140
    )
    if circles is None:
        return []
    return np.uint16(np.around(circles[0]))

def filter_circles(circles):
    if len(circles) == 0:
        return []
    circles = sorted(circles, key=lambda c: c[2], reverse=True)
    filtered = []
    for (x1, y1, r1) in circles:
        keep = True
        for (x2, y2, r2) in filtered:
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if dist < max(r1, r2) * 0.6:
                keep = False
                break
        if keep:
            filtered.append((x1, y1, r1))
    return filtered

def classify_coin_sift_with_prior(roi_gray, group_candidates):
    h, w = roi_gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(w // 2 * 0.95), 255, -1)
    
    kp1, des1 = sift.detectAndCompute(roi_gray, mask)
    if des1 is None or len(des1) < 2:
        return None

    best = None
    best_ratio = -1
    
    for name in group_candidates:
        if name not in templates_db:
            continue
        kp2, des2 = templates_db[name]
        if des2 is None or len(des2) < 2:
            continue
        try:
            matches = bf.knnMatch(des1, des2, k=2)
        except Exception:
            continue
            
        good = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good.append(m)
                
        ratio = len(good) / len(des2)
        if ratio > best_ratio:
            best_ratio = ratio
            best = name
            
    return best

def evaluate():
    correct_count = 0
    total_count = 0
    total_val_gt = 0.0
    total_val_pred = 0.0
    
    for img_path in sorted(Path(TARGET_DIR).glob("*.jpg"), key=lambda p: int(p.stem.split("_")[1])):
        name = img_path.name
        if name not in gt_db:
            continue
            
        gt_cents, gt_coins = gt_db[name]
        
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        circles = detect_circles(gray)
        circles = filter_circles(circles)
        
        pred_coins = []
        pred_cents = 0
        
        for (x, y, r) in circles:
            roi = extract_roi(img, x, y, r)
            if roi is None or roi.size == 0:
                continue
            
            # Channel swap BGR to RGB for HSV color prior
            roi_rgb = roi[:, :, ::-1]
            hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
            h, w, _ = roi.shape
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (w // 2, h // 2), int(w // 2 * 0.9), 255, -1)
            
            avg_hsv = cv2.mean(hsv, mask=mask)[:3]
            s_val = avg_hsv[1]
            h_val = avg_hsv[0]
            
            if s_val < 110:
                group_candidates = ["1euro", "2euro"]
            elif h_val < 12.5:
                group_candidates = ["1cent", "2cent", "5cent"]
            else:
                group_candidates = ["10cent", "20cent", "50cent"]
                
            roi_gray = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (256, 256))
            
            coin = classify_coin_sift_with_prior(roi_gray, group_candidates)
            if coin is not None:
                val_cents = COIN_NOMINAL_VALUE_CENTS[coin]
                pred_coins.append(val_cents)
                pred_cents += val_cents
                
        total_count += 1
        total_val_gt += gt_cents / 100.0
        total_val_pred += pred_cents / 100.0
        
        if pred_cents == gt_cents:
            correct_count += 1
            print(f"{name}: Perfect Match! Pred={pred_cents/100.0:.2f} €, GT={gt_cents/100.0:.2f} €")
        else:
            print(f"{name}: Mismatch. Pred={pred_cents/100.0:.2f} € {pred_coins}, GT={gt_cents/100.0:.2f} € {gt_coins}")
            
    print("\n" + "="*30)
    print(f"Accuracy (exact value match): {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)")
    print(f"GT Total Value: {total_val_gt:.2f} €")
    print(f"Pred Total Value: {total_val_pred:.2f} €")

if __name__ == '__main__':
    evaluate()
