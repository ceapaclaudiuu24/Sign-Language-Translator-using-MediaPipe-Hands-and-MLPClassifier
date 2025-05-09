import joblib


class LoadModel:
    def __init__(self, path="../../model/sign_language_model.pkl"):
        data = joblib.load(path)
        self.model = data['model']
        self.index_to_label = data['label_map']

    def predict(self, landmarks):
        idx = self.model.predict([landmarks])[0]
        return self.index_to_label.get(idx, "?")
