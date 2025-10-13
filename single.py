# single_image_process.py

import cv2
import numpy as np
from preprocess import preprocess_sheet, OUTPUT_SIZE
from bubble_detection import detect_bubbles, group_bubbles, detect_answers, visualize_answers_with_key, calculate_score, OPTION_LETTERS, ANSWER_KEY

INPUT_IMAGE = "AB.jpeg" 
OUTPUT_IMAGE = "annotated.jpg"  

NAME_BOX_SIZE = (OUTPUT_SIZE[0], 120) 

def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def extract_name_box(image, top_ratio=0.40):
    h, w = image.shape[:2]
    roi = image[0:int(h*top_ratio), 0:w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    name_contour = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            name_contour = approx
            break

    if name_contour is None:
        warped_name = cv2.resize(roi, NAME_BOX_SIZE)
    else:
        pts = name_contour.reshape(4,2)
        rect = order_points(pts)
        dst = np.array([
            [0,0],
            [NAME_BOX_SIZE[0]-1,0],
            [NAME_BOX_SIZE[0]-1, NAME_BOX_SIZE[1]-1],
            [0, NAME_BOX_SIZE[1]-1]
        ], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped_name = cv2.warpPerspective(roi, M, NAME_BOX_SIZE)

    return warped_name

def add_name_box_and_score(warped_sheet, name_box, score, score_height=50):
    h_sheet, w_sheet = warped_sheet.shape[:2]
    h_name, w_name = name_box.shape[:2]

    canvas_height = h_sheet + h_name + score_height
    canvas = 255 * np.ones((canvas_height, w_sheet, 3), dtype=np.uint8)

    canvas[0:h_name, 0:w_sheet] = name_box

    score_y_start = h_name
    score_y_end = h_name + score_height
    cv2.rectangle(canvas, (0, score_y_start), (w_sheet, score_y_end), (200, 200, 200), -1) 
    cv2.putText(canvas, f"Score: {score}/{len(ANSWER_KEY)}", (20, score_y_start + int(score_height*0.7)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    canvas[score_y_end:score_y_end+h_sheet, 0:w_sheet] = warped_sheet

    return canvas

if __name__ == "__main__":
    try:
        print(f"📄 Processing: {INPUT_IMAGE}")

        original = cv2.imread(INPUT_IMAGE)

        name_box = extract_name_box(original)

        warped = preprocess_sheet(INPUT_IMAGE, OUTPUT_SIZE)

        bubble_list, thresh = detect_bubbles(warped)
        QUESTIONS = group_bubbles(bubble_list)

        answers = detect_answers(QUESTIONS, thresh)

        score = calculate_score(answers, ANSWER_KEY)
        print(f"\n🎯 Total Score: {score}/{len(ANSWER_KEY)}")

        visualize_answers_with_key(warped, QUESTIONS, answers, ANSWER_KEY)
        
        final_image = add_name_box_and_score(warped, name_box, score)

        cv2.imwrite(OUTPUT_IMAGE, final_image)
        print(f"✅ Annotated sheet saved as {OUTPUT_IMAGE}")

    except Exception as e:
        print(f"❌ Failed to process {INPUT_IMAGE}: {e}")
