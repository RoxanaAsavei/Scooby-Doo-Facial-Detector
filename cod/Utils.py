import cv2 as cv
import numpy as np 
from collections import defaultdict

def clip01(img):
    return np.clip(img, 0.0, 1.0).astype(np.float32)

def aug_blur(img):
    return cv.GaussianBlur(img, (3, 3), 0)

def aug_rotate(img, degrees = 5):
    h, w = img.shape[0], img.shape[1]
    center = (w / 2.0, h / 2.0)
    M = cv.getRotationMatrix2D(center, degrees, 1.0)
    return cv.warpAffine(
        img, M, (w, h),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_REFLECT_101
    )

def aug_brightness_contrast(img, brightness=0.05, contrast=0.15):
    b = np.random.uniform(-brightness, brightness)
    c = np.random.uniform(1.0-contrast, 1.0+contrast)
    out = img * c + b
    return clip01(out)

def aug_shift(img, max_shift=4):
    h, w = img.shape[:2]
    dx = np.random.randint(-max_shift, max_shift + 1)
    dy = np.random.randint(-max_shift, max_shift + 1)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    out = cv.warpAffine(img, M, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)
    return clip01(out)

def intersection_over_union(bbox_a, bbox_b):
    x_a = max(bbox_a[0], bbox_b[0])
    y_a = max(bbox_a[1], bbox_b[1])
    x_b = min(bbox_a[2], bbox_b[2])
    y_b = min(bbox_a[3], bbox_b[3])
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)
    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    return iou

def group_detections_by_filename(detections, file_names):
        d = defaultdict(list)
        for det, fname in zip(detections, file_names):
            d[str(fname)].append(det)
        # convertim listele in np arrays
        return {k: np.stack(v, axis=0) for k, v in d.items()}