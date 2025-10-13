# preprocessing.py
import cv2
import numpy as np

# ---- CONFIG ----
IMAGE_PATH = "CD.jpeg"
OUTPUT_PATH = "warped.jpg"
OUTPUT_SIZE = (600, 780)  # width x height

# ---- UTILITY ----
def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# ---- PREPROCESSING FUNCTION ----
def preprocess_sheet(image_path, output_size):
    image = cv2.imread(image_path)
    orig = image.copy()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    sheet_contour = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            sheet_contour = approx
            break
    
    if sheet_contour is None:
        raise Exception("❌ Could not find the answer sheet contour!")
    
    pts = sheet_contour.reshape(4,2)
    rect = order_points(pts)
    
    dst = np.array([
        [0,0],
        [output_size[0]-1, 0],
        [output_size[0]-1, output_size[1]-1],
        [0, output_size[1]-1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, output_size)

    return warped

# ---- MAIN ----
if __name__ == "__main__":
    preprocess_sheet(IMAGE_PATH, OUTPUT_PATH, OUTPUT_SIZE)
