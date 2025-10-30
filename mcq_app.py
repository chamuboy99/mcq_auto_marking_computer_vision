import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QLabel, 
                             QVBoxLayout, QFileDialog, QProgressBar, QTextEdit,
                             QDialog, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from preprocess import preprocess_sheet, OUTPUT_SIZE
from bubble_detection import detect_bubbles, group_bubbles, detect_answers, visualize_answers_with_key, calculate_score, OPTION_LETTERS, ANSWER_KEY

NAME_BOX_SIZE = (OUTPUT_SIZE[0], 120)
SCORE_HEIGHT = 50

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

def add_name_box_and_score(warped_sheet, name_box, score):
    h_sheet, w_sheet = warped_sheet.shape[:2]
    h_name, w_name = name_box.shape[:2]

    canvas_height = h_name + SCORE_HEIGHT + h_sheet
    canvas = 255 * np.ones((canvas_height, w_sheet, 3), dtype=np.uint8)

    # Name box
    canvas[0:h_name, 0:w_sheet] = name_box

    # Score bar
    score_y_start = h_name
    score_y_end = h_name + SCORE_HEIGHT
    cv2.rectangle(canvas, (0, score_y_start), (w_sheet, score_y_end), (200, 200, 200), -1)
    cv2.putText(canvas, f"Score: {score}/{len(ANSWER_KEY)}", (20, score_y_start + int(SCORE_HEIGHT*0.7)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Annotated sheet
    canvas[score_y_end:score_y_end+h_sheet, 0:w_sheet] = warped_sheet

    return canvas

def cv2_to_qpixmap(cv_img, max_width=600, max_height=500):
    """Convert OpenCV image to QPixmap for display with optional resizing"""
    img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape

    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h, 1.0)

    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_img = cv2.resize(img_rgb, (new_w, new_h))

    bytes_per_line = ch * new_w
    qimg = QImage(resized_img.data, new_w, new_h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

# ------------------ Batch Processing Thread ------------------

class BatchWorker(QThread):
    log_signal = pyqtSignal(str)

    def __init__(self, input_folder, output_folder):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder

    def run(self):
        os.makedirs(self.output_folder, exist_ok=True)
        files = [f for f in os.listdir(self.input_folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        for filename in files:
            input_path = os.path.join(self.input_folder, filename)
            output_path = os.path.join(self.output_folder, f"annotated_{filename}")
            try:
                self.log_signal.emit(f"📄 Processing: {filename}")
                original = cv2.imread(input_path)
                name_box = extract_name_box(original)
                warped = preprocess_sheet(input_path, OUTPUT_SIZE)
                bubble_list, thresh = detect_bubbles(warped)
                QUESTIONS = group_bubbles(bubble_list)
                answers = detect_answers(QUESTIONS, thresh)
                score = calculate_score(answers, ANSWER_KEY)
                visualize_answers_with_key(warped, QUESTIONS, answers, ANSWER_KEY)
                final_image = add_name_box_and_score(warped, name_box, score)
                cv2.imwrite(output_path, final_image)
                self.log_signal.emit(f"✅ Saved: {output_path}\n")
            except Exception as e:
                self.log_signal.emit(f"❌ Failed: {filename} - {e}\n")

# ------------------ Answer Key Edit Dialog ------------------

class AnswerKeyDialog(QDialog):
    def __init__(self, current_key):
        super().__init__()
        self.setWindowTitle("Edit Answer Key")
        self.layout = QVBoxLayout()
        
        self.text_edit = QTextEdit()
        self.text_edit.setText(','.join([OPTION_LETTERS[i] for i in current_key]))
        self.layout.addWidget(self.text_edit)
        
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_key)
        self.layout.addWidget(self.save_btn)
        
        self.setLayout(self.layout)
        self.new_key = None
    
    def save_key(self):
        text = self.text_edit.toPlainText().replace(" ", "")
        letters = text.split(",")
        if len(letters) != len(ANSWER_KEY):
            QMessageBox.warning(self, "Error", f"Answer key must have {len(ANSWER_KEY)} entries!")
            return
        try:
            indices = [OPTION_LETTERS.index(l.upper()) for l in letters]
        except ValueError:
            QMessageBox.warning(self, "Error", "Only letters A-D allowed!")
            return
        self.new_key = indices
        self.accept()

# ------------------ GUI Application ------------------

class MCQApp(QWidget):
    def __init__(self):
        super().__init__()

        with open("mcq_style.qss", "r") as f:
            self.setStyleSheet(f.read())

        self.setWindowTitle("MCQ Sheet Processor")
        self.setGeometry(100, 100, 900, 700)

        self.layout = QVBoxLayout()

        self.single_btn = QPushButton("Single Sheet")
        self.single_btn.clicked.connect(self.single_process)
        self.layout.addWidget(self.single_btn)

        self.clear_btn = QPushButton("Clear Image")
        self.clear_btn.clicked.connect(self.clear_image)
        self.clear_btn.hide()  # hidden until an image is displayed
        self.layout.addWidget(self.clear_btn)

        self.batch_btn = QPushButton("Multiple Sheets")
        self.batch_btn.clicked.connect(self.batch_process)
        self.layout.addWidget(self.batch_btn)

        self.capture_btn = QPushButton("Capture From Camera")
        self.capture_btn.clicked.connect(self.capture_process)
        self.layout.addWidget(self.capture_btn)

        self.edit_key_btn = QPushButton("Edit Answers")
        self.edit_key_btn.clicked.connect(self.edit_answer_key)
        self.layout.addWidget(self.edit_key_btn)

        self.image_label = QLabel("Annotated image will appear here")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.layout.addWidget(self.log_text)

        self.setLayout(self.layout)

    def append_log(self, message):
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def clear_image(self):
        self.image_label.clear()
        self.image_label.setText("Annotated image will appear here")
        self.clear_btn.hide()
        self.append_log("🧹 Image preview cleared.")

    # ---------------- Single Image ----------------
    def single_process(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.jpg *.jpeg *.png)")
        if file_name:
            self.single_process_image(file_name, display_preview=True)

    def single_process_image(self, file_name, display_preview=False):
        try:
            self.append_log(f"📄 Processing: {file_name}")
            original = cv2.imread(file_name)
            name_box = extract_name_box(original)
            warped = preprocess_sheet(file_name, OUTPUT_SIZE)
            bubble_list, thresh = detect_bubbles(warped)
            QUESTIONS = group_bubbles(bubble_list)
            answers = detect_answers(QUESTIONS, thresh)
            score = calculate_score(answers, ANSWER_KEY)
            visualize_answers_with_key(warped, QUESTIONS, answers, ANSWER_KEY)
            final_image = add_name_box_and_score(warped, name_box, score)
            save_name = os.path.join(os.path.dirname(file_name), "annotated_" + os.path.basename(file_name))
            cv2.imwrite(save_name, final_image)
            self.append_log(f"✅ Saved: {save_name}")
            if display_preview:
                self.image_label.setPixmap(cv2_to_qpixmap(final_image, max_width=600, max_height=500))
                self.clear_btn.show()
        except Exception as e:
            self.append_log(f"❌ Failed: {e}")

    # ---------------- Batch Process ----------------
    def batch_process(self):
        input_folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        output_folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if input_folder and output_folder:
            self.worker = BatchWorker(input_folder, output_folder)
            self.worker.log_signal.connect(self.append_log)
            self.worker.start()

    # ---------------- Capture ----------------
    def capture_process(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.append_log("❌ Cannot access camera")
            return
        self.append_log("📷 Capturing... Press 's' to snap, 'q' to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Camera", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('s'):
                save_path = "captured_sheet.jpg"
                cv2.imwrite(save_path, frame)
                self.append_log(f"📄 Captured: {save_path}")
                self.single_process_image(save_path, display_preview=True)
                break
            elif key & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    # ---------------- Edit Answer Key ----------------
    def edit_answer_key(self):
        global ANSWER_KEY  # declare at the top
        dialog = AnswerKeyDialog(ANSWER_KEY)
        if dialog.exec_():
            if dialog.new_key:
                ANSWER_KEY = dialog.new_key
                self.append_log(f"✅ Updated Answer Key: {', '.join([OPTION_LETTERS[i] for i in ANSWER_KEY])}")


# ------------------ Run ------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MCQApp()
    window.show()
    sys.exit(app.exec_())
