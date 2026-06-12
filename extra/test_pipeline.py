import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Configuration
TARGET_DIR = "coin_dataset/target_set"
TEMPLATE_DIR = "coin_dataset/reference_set"

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

# SIFT detector
sift = cv2.SIFT_create()

# FLANN matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

def load_templates():
    templates = {}
    for path in Path(TEMPLATE_DIR).glob("*.jpg"):
        name = path.stem.lower().replace("_", "")
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256))
        kp, des = sift.detectAndCompute(img, None)
        if des is not None:
            templates[name] = (kp, des, img)
    return templates

templates_db = load_templates()

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

def preprocess(roi):
    if roi is None or roi.shape[0] < 20 or roi.shape[1] < 20:
        return None
    roi = cv2.resize(roi, (256, 256))
    return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

def detect_circles(gray):
    # Gaussian Blur (Lab 1 reference: cv2.GaussianBlur)
    sigma = 1.5
    k_size = int(np.ceil((3 * sigma)) * 2 + 1)
    blur = cv2.GaussianBlur(gray, (k_size, k_size), sigma)
    
    # Hough Circles (param1 is the higher threshold for Canny edge detector)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=60,
        param1=100,
        param2=30,
        minRadius=40,
        maxRadius=120
    )
    if circles is None:
        return []
    return np.uint16(np.around(circles[0]))

def classify_coin_sift(roi_gray):
    # Circular mask to exclude background noise (# extra)
    h, w = roi_gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(w // 2 * 0.95), 255, -1)
    
    kp1, des1 = sift.detectAndCompute(roi_gray, mask)
    if des1 is None or len(des1) < 2:
        return None

    best = None
    best_matches = 0
    
    for name, (kp2, des2, _) in templates_db.items():
        if des2 is None:
            continue
        try:
            matches = flann.knnMatch(des1, des2, k=2)
        except Exception:
            continue
            
        good = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < 0.7 * n.distance:
                good.append(m)
                
        if len(good) > best_matches:
            best_matches = len(good)
            best = name
            
    if best_matches < 5:  # Rejection threshold
        return None
    return best

def run_test():
    for img_path in Path(TARGET_DIR).glob("*.jpg"):
        print(f"\nProcessing {img_path.name}")
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        circles = detect_circles(gray)
        print(f"Detected {len(circles)} circles")
        
        detected_coins = []
        total_val = 0.0
        
        for (x, y, r) in circles:
            roi = extract_roi(img, x, y, r)
            roi_gray = preprocess(roi)
            if roi_gray is None:
                continue
            
            coin = classify_coin_sift(roi_gray)
            if coin is not None:
                val = COIN_VALUES[coin]
                detected_coins.append((coin, val))
                total_val += val
                print(f"  Coin at ({x}, {y}) r={r}: {coin} ({val} €)")
            else:
                print(f"  Coin at ({x}, {y}) r={r}: Unclassified")
                
        print(f"Total Value: {total_val:.2f} €")

if __name__ == '__main__':
    run_test()
