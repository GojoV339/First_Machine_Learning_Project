import joblib
import numpy as np

MODEL_PATH = "models/model.joblib"
PREPROCESSOR_PATH = "models/preprocessor.joblib"


def load_artifacts():
    """
    Load the preprocessor nd model from disk.
    """
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    return preprocessor, model

# Order of features expected by the model
FEATURE_ORDER = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

# Iris target class names (index -> name)
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def prepare_input(sample: dict) -> np.ndarray:
    """
    Convert a simple dict into a 2D numpy array
    in the correct feature order
    """
    
    values = [sample[feature] for feature in FEATURE_ORDER]
    return np.array(values).reshape(1,-1)

def predict_single(sample: dict):
    """
    Given a dict with 4 iris features return predicted class and prob's
    """
    
    preprocessor, model = load_artifacts()
    X = prepare_input(sample)
    X_proc = preprocessor.transform(X)
    
    
    proba = model.predict_proba(X_proc)[0]
    pred_class_idx = np.argmax(proba)
    pred_class_name = CLASS_NAMES[pred_class_idx]
    
    return pred_class_name,proba

if __name__ == "__main__":
    sample = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    pred_class, proba = predict_single(sample)
    print("Input sample:", sample)
    print("Predicted class:", pred_class)
    print("Probabilities:", proba)