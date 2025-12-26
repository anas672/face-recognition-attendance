import cv2
import numpy as np

# ================= LIVENESS CONFIG =================
MOTION_THRESHOLD = 15


# ---------- IMAGE AUGMENTATION ----------

def adjust_brightness(image, value):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    v = v.astype(np.int16)
    v = np.clip(v + value, 0, 255).astype(np.uint8)

    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB)


def adjust_contrast(image, alpha):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)


def rotate_image(image, angle):
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, m, (w, h))


def slight_blur(image):
    return cv2.GaussianBlur(image, (3, 3), 0)


def generate_variations(image):
    variations = [image, cv2.flip(image, 1)]

    for b in [-20, 20]:
        variations.append(adjust_brightness(image, b))

    for c in [0.9, 1.1]:
        variations.append(adjust_contrast(image, c))

    for a in [-5, 5]:
        variations.append(rotate_image(image, a))

    variations.append(slight_blur(image))
    return variations


# ---------- LIVENESS CHECK ----------

def is_live_face(current_gray, prev_gray):
    if prev_gray is None:
        return False, current_gray

    current_gray = cv2.resize(
        current_gray,
        (prev_gray.shape[1], prev_gray.shape[0])
    )

    diff = cv2.absdiff(current_gray, prev_gray)
    motion_score = diff.mean()

    return motion_score > MOTION_THRESHOLD, current_gray
