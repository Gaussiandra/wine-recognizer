import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    RANDOM_STATE = 1122

    train_features = np.load(open("data/train_features.npy", "rb"))
    train_targets = np.load(open("data/train_targets.npy", "rb"))

    rfc = RandomForestClassifier(random_state=RANDOM_STATE)
    rfc.fit(train_features, train_targets)

    dump(rfc, "models/rfc100.joblib")
