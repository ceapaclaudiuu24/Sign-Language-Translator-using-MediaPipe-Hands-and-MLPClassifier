import cv2
import mediapipe as mp


class HandCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1)
        self.mp_draw = mp.solutions.drawing_utils

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        return frame, results

    def draw(self, frame, landmarks):
        self.mp_draw.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)

    def release(self):
        self.cap.release()
