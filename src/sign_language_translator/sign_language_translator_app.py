import os
import sys

import cv2
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

from detection_model_load import LoadModel
from hand_camera import HandCamera


class SignLanguageTranslator(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("main.ui", self)
        self.setFixedSize(1000, 800)

        self.slideshow_label = self.findChild(QtWidgets.QLabel, "slideshowLabel")
        self.sign_label = self.findChild(QtWidgets.QLabel, "signLabel")
        self.video_label = self.findChild(QtWidgets.QLabel, 'videoLabel')
        self.prediction_label = self.findChild(QtWidgets.QLabel, 'predictionLabel')
        self.text_box = self.findChild(QtWidgets.QTextEdit, 'textBox')

        self.save_to_text_btn = self.findChild(QtWidgets.QPushButton, "saveToTextButton")
        self.prev_btn = self.findChild(QtWidgets.QPushButton, "prevButton")
        self.next_btn = self.findChild(QtWidgets.QPushButton, "nextButton")
        self.add_btn = self.findChild(QtWidgets.QPushButton, 'addButton')
        self.space_btn = self.findChild(QtWidgets.QPushButton, 'spaceButton')
        self.delete_btn = self.findChild(QtWidgets.QPushButton, 'deleteButton')
        self.reset_btn = self.findChild(QtWidgets.QPushButton, 'resetButton')
        self.auto_btn = self.findChild(QtWidgets.QPushButton, 'autoButton')
        self.case_btn = self.findChild(QtWidgets.QPushButton, 'caseButton')

        self.save_to_text_btn.clicked.connect(self.save_to_text)
        self.add_btn.clicked.connect(self.add_prediction)
        self.space_btn.clicked.connect(lambda: self.text_box.insertPlainText(" "))
        self.delete_btn.clicked.connect(self.delete_last)
        self.reset_btn.clicked.connect(lambda: self.text_box.clear())
        self.auto_btn.setCheckable(True)
        self.auto_btn.toggled.connect(self.toggle_auto_mode)
        self.case_btn.setCheckable(True)
        self.case_btn.toggled.connect(self.toggle_case_mode)
        self.prev_btn.clicked.connect(self.show_prev_image)
        self.next_btn.clicked.connect(self.show_next_image)

        self.camera = HandCamera()
        self.model = LoadModel()

        self.images = [os.path.join("../../slides", f) for f in os.listdir("../../slides")
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.images.sort()
        self.index = 0
        self.show_image()

        self.prediction = "..."
        self.auto_mode = False
        self.uppercase_mode = False
        self.auto_counter = 0
        self.auto_interval = 30

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def save_to_text(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Transcript As",
            "",
            "Text Files (*.txt);;All Files (*)",
            options=options
        )
        if file_path:
            if not file_path.lower().endswith('.txt'):
                file_path += '.txt'
            with open(file_path, 'w', encoding='utf-8') as f:
                transcript = self.text_box.toPlainText()
                f.write(transcript)
            QtWidgets.QMessageBox.information(
                self,
                "Saved",
                f"Transcript saved successfully to:\n{file_path}"
            )

    def show_image(self):
        if not self.images:
            self.slideshow_label.setText("No images found")
            self.sign_label.setText("No image")
            return
        img_path = self.images[self.index]
        pixmap = QPixmap(img_path)
        self.slideshow_label.setPixmap(pixmap.scaled(self.slideshow_label.size(), aspectRatioMode=1))
        base_name = os.path.basename(img_path)
        name_without_ext = os.path.splitext(base_name)[0]
        self.sign_label.setText(name_without_ext)

    def show_next_image(self):
        if self.images:
            self.index = (self.index + 1) % len(self.images)
            self.show_image()

    def show_prev_image(self):
        if self.images:
            self.index = (self.index - 1) % len(self.images)
            self.show_image()

    def update_frame(self):
        frame, results = self.camera.read_frame()
        if frame is None:
            return
        pred = "..."
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            coords = [c for pt in lm.landmark for c in (pt.x, pt.y)]
            pred = self.model.predict(coords)
            self.camera.draw(frame, lm)
        self.prediction_label.setText(f"Prediction: {pred}")

        self.prediction = pred
        if self.auto_mode:
            self.auto_counter += 1
            if self.auto_counter >= self.auto_interval:
                self.add_prediction()
                self.auto_counter = 0

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        qt_img = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    def add_prediction(self):
        if not self.prediction or self.prediction == "...":
            return

        if self.prediction.lower() == 'space':
            self.text_box.insertPlainText(" ")
            return
        if self.prediction.lower() == 'del':
            self.delete_last()
            return
        if self.prediction.lower() == 'nothing':
            return

        char = self.prediction
        char = char.upper() if self.uppercase_mode else char.lower()
        self.text_box.insertPlainText(char)

    def delete_last(self):
        text = self.text_box.toPlainText()
        self.text_box.setPlainText(text[:-1])

    def toggle_auto_mode(self, checked):
        self.auto_mode = checked
        self.auto_btn.setText("⚙️ Auto: ON" if checked else "⚙️ Auto")
        self.auto_counter = 0

    def toggle_case_mode(self, checked):
        self.uppercase_mode = checked
        self.case_btn.setText("Aa: UPPER" if checked else "Aa: lower")

    def closeEvent(self, event):
        self.camera.cap.release()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = SignLanguageTranslator()
    window.show()
    sys.exit(app.exec_())
