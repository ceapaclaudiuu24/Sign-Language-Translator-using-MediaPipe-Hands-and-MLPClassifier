import os
import string
import cv2
import mediapipe as mp
import numpy as np
import absl.logging
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
absl.logging.set_verbosity(absl.logging.ERROR)

DATASET_PATH = '../../dataset'
OUTPUT_PATH = '../../model/hand_landmarks_dataset.npy'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

X, y = [], []
total = 0
start_time = time.time()

label_map = {char: idx for idx, char in enumerate(string.ascii_uppercase)}

label_map.update({str(d): idx + 26 for idx, d in enumerate(range(10))})

label_map.update({
    'del': 36,
    'nothing': 37,
    'space': 38
})

for label in sorted(os.listdir(DATASET_PATH)):
    folder_path = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(folder_path):
        continue

    print(f"Processing letter '{label}'...")
    count = 0

    for img_file in os.listdir(folder_path):
        if not img_file.endswith('.png'):
            continue

        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            data = []
            for lm in landmarks.landmark:
                data.extend([lm.x, lm.y])
            X.append(data)
            y.append(label_map[label])
            count += 1
            total += 1
            print(f"{img_file} : landmarks extracted")
        else:
            print(f"{img_file} : no hand detected")

    print(f"Finished letter '{label}' : {count} valid samples\n")

X = np.array(X)
y = np.array(y)

np.save(OUTPUT_PATH, {'X': X, 'y': y})
elapsed = time.time() - start_time
print(f"Saved {len(X)} samples with 42 features each to '{OUTPUT_PATH}' in {elapsed:.2f} seconds.")
