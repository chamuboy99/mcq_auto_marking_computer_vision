# bubble_detection.py
import cv2
import numpy as np

# ---- CONFIG ----
BUBBLES_PER_Q = 4
OPTION_LETTERS = ['A','B','C','D']

# ---- FUNCTIONS ----
def detect_bubbles(warped):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 25)
    
    # 4️⃣ Morphological opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    circles = cv2.HoughCircles(thresh,
                               cv2.HOUGH_GRADIENT,
                               dp=1.2,
                               minDist=20,
                               param1=50,
                               param2=30,
                               minRadius=25,
                               maxRadius=55)
    
    if circles is None:
        raise Exception("❌ No bubbles detected!")
    
    circles = np.uint16(np.around(circles))
    bubble_list = circles[0].tolist()
    print(f"✅ Total bubbles detected: {len(bubble_list)}")
    return bubble_list, thresh

def group_bubbles(bubble_list):
    bubble_list.sort(key=lambda x: x[1])
    QUESTIONS = []
    for i in range(0, len(bubble_list), BUBBLES_PER_Q):
        q_bubbles = bubble_list[i:i+BUBBLES_PER_Q]
        q_bubbles.sort(key=lambda x: x[0])
        QUESTIONS.append(q_bubbles)
    print(f"✅ Total questions detected: {len(QUESTIONS)}")
    return QUESTIONS

def detect_answers(QUESTIONS, thresh, fill_threshold=70):
    answers = []

    for q_bubbles in QUESTIONS:
        bubble_vals = []
        for idx, (x, y, r) in enumerate(q_bubbles):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.circle(mask, (x, y), r, 255, -1)
            mean_val = cv2.mean(thresh, mask=mask)[0]
            bubble_vals.append((idx, mean_val))
        
        filled_bubbles = [idx for idx, val in bubble_vals if val > fill_threshold]

        if len(filled_bubbles) == 1:
            answers.append(filled_bubbles[0])
        else:
            answers.append(filled_bubbles)

    return answers


ANSWER_KEY = [1, 3, 0, 2, 1, 0, 0, 2, 1, 2]

def visualize_answers_with_key(warped, QUESTIONS, answers, answer_key):
    for q_idx, q_bubbles in enumerate(QUESTIONS):
        ans = answers[q_idx]
        correct_idx = answer_key[q_idx]

        if ans is None:
            cv2.circle(warped, (q_bubbles[correct_idx][0], q_bubbles[correct_idx][1]),
                       q_bubbles[correct_idx][2], (255, 255, 0), 2)
            continue

        if isinstance(ans, list):
            for idx in ans:
                cv2.circle(warped, (q_bubbles[idx][0], q_bubbles[idx][1]), q_bubbles[idx][2], (255,0,0), 2) 
            cv2.circle(warped, (q_bubbles[correct_idx][0], q_bubbles[correct_idx][1]),
                       q_bubbles[correct_idx][2], (255, 255, 0), 2)
            continue

        for idx, (x, y, r) in enumerate(q_bubbles):
            if idx == ans:
                color = (0, 255, 0) if ans == correct_idx else (0, 0, 255)
                cv2.circle(warped, (x, y), r, color, 2)
            cv2.putText(warped, OPTION_LETTERS[idx], (x-10, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        if ans != correct_idx:
            cv2.circle(warped, (q_bubbles[correct_idx][0], q_bubbles[correct_idx][1]),
                       q_bubbles[correct_idx][2], (255, 255, 0), 2)


def calculate_score(answers, answer_key):
    score = 0
    for i, ans in enumerate(answers):
        correct = answer_key[i]
        if isinstance(ans, list):
            continue
        elif ans == correct:
            score += 1
    return score




