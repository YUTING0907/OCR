import cv2
import numpy as np
from rembg import remove
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

'''
rembg 使用 MODNet 或类似的语义分割模型，已经在 大量带有透明通道的图像数据 上训练过。

主要用来 区分前景（主体）和背景
'''
def show(title, img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def remove_background(image):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    result = remove(pil_img)
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

def mock_detect_corners(image):
    h, w = image.shape[:2]
    return np.array([[50, 50], [w-50, 40], [w-40, h-50], [60, h-40]], dtype="float32")

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def perspective_transform(image, corners):
    rect = order_points(corners)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxWidth = int(max(widthA, widthB))
    maxHeight = int(max(heightA, heightB))
    dst = np.array([
        [0, 0], [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def preprocess_id_card(image_path):
    image = cv2.imread(image_path)
    show("原图", image)

    enhanced = apply_clahe(image)
    show("CLAHE增强后", enhanced)

    no_bg = remove_background(enhanced)
    show("背景去除后", no_bg)

    corners = mock_detect_corners(no_bg)
    corrected = perspective_transform(no_bg, corners)
    show("透视矫正后", corrected)

    return corrected
