from data import load_data
from features import DataTransformation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
import joblib
import os

def train():
    X_train,X_test,y_train,y_test = load_data()
    
    
    preprocessor = DataTransformation.build_preprocessor()
    preprocessor = DataTransformation.fit_preprocessor(preprocessor,X_train)
    
    X_train_proc = DataTransformation.transform_with_preprocessor(preprocessor,X_train)
    X_test_proc = DataTransformation.transform_with_preprocessor(preprocessor,X_test)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_proc, y_train)
    
    y_pred = model.predict(X_test_proc)
    
    print(f"Logistic Regression model accuracy: {accuracy_score(y_test,y_pred)}")
    print(classification_report(y_test,y_pred))
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, "models/preprocessor.joblib")
    joblib.dump(model, "models/model.joblib")
    
    print("Saved Preprocessor and model to 'models/' directory.")
    
if __name__ == "__main__":
    train()