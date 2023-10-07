import numpy as np
from joblib import load
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    test_features = np.load(open("data/test_features.npy", "rb"))
    test_targets = np.load(open("data/test_targets.npy", "rb"))

    rfc = load("models/rfc100.joblib")

    predicted_targets = rfc.predict(test_features)
    np.save(open("data/predicted_targets.npy", "wb"), predicted_targets)

    accuracy = accuracy_score(test_targets, predicted_targets)
    print(accuracy)
