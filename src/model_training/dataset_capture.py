import cv2
import os
import numpy as np
import time

DATASET_DIR = '../../dataset'
LABELS = [chr(c) for c in range(ord('A'), ord('Z')+1)] + [str(d) for d in range(10)] + ['nothing', 'space', 'del']     # List your sign labels here
NUM_IMAGES = 500
IMG_WIDTH, IMG_HEIGHT = 200, 200
CAPTURE_KEY = ord('c')
NEXT_LABEL_KEY = ord('n')
QUIT_KEY = ord('q')
DELAY_BETWEEN_CAPTURES = 0.05

def setup_dirs():
    os.makedirs(DATASET_DIR, exist_ok=True)
    for lbl in LABELS:
        os.makedirs(os.path.join(DATASET_DIR, lbl), exist_ok=True)

def get_hand_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([20, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask

def capture_dataset():
    setup_dirs()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    label_idx, count = 0, 0
    total_labels = len(LABELS)
    current_label = LABELS[label_idx]

    print(f"Ready to capture for label '{current_label}'.")
    print("Press 'c' to start automatic capture, 'n' for next label, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        size = min(w, h) // 2
        x, y = (w - size)//2, (h - size)//2

        cv2.rectangle(frame, (x,y), (x+size,y+size), (0,255,0), 2)
        info = f"Label: {current_label} ({count}/{NUM_IMAGES})"
        cv2.putText(frame, info, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.imshow('Capture', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == CAPTURE_KEY:
            print(f"Capturing {NUM_IMAGES} images for label '{current_label}'...")
            while count < NUM_IMAGES:
                ret, frame = cap.read()
                if not ret:
                    break
                roi = frame[y:y+size, x:x+size]
                img = cv2.resize(roi, (IMG_WIDTH, IMG_HEIGHT))
                path = os.path.join(DATASET_DIR, current_label, f"{count:04d}.png")
                cv2.imwrite(path, img)
                count += 1
                print(f"Captured {count}/{NUM_IMAGES} for label '{current_label}'.")
                preview = frame.copy()
                cv2.rectangle(preview, (x,y), (x+size,y+size), (0,255,0), 2)
                cv2.putText(preview, f"Capturing... {count}/{NUM_IMAGES}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow('Capture', preview)
                cv2.waitKey(1)
                time.sleep(DELAY_BETWEEN_CAPTURES)
            print(f"Finished capturing for label '{current_label}'. Press 'n' to proceed.")
        elif key == NEXT_LABEL_KEY:
            if label_idx < total_labels - 1:
                label_idx += 1
                current_label = LABELS[label_idx]
                count = 0
                print(f"Switched to label '{current_label}'. Press 'c' to begin capturing.")
            else:
                print("All labels completed. Exiting.")
                break
        elif key == QUIT_KEY:
            print("Quitting capture.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    capture_dataset()