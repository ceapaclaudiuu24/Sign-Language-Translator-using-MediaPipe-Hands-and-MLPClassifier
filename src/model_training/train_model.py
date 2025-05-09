import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import string

data = np.load('../../hand_landmarks_dataset.npy', allow_pickle=True).item()
X, y = data['X'], data['y']

label_map = {char: idx for idx, char in enumerate(string.ascii_uppercase)}
label_map.update({str(i): idx + 26 for idx, i in enumerate(range(10))})
label_map.update({'del': 36, 'nothing': 37, 'space': 38})

index_to_label = {v: k for k, v in label_map.items()}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=[index_to_label[i] for i in sorted(set(y))]))

joblib.dump({
    'model': model,
    'label_map': index_to_label
}, 'sign_language_model.pkl')

print("Model and label map saved to 'sign_language_model.pkl'")
