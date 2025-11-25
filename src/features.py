from data import load_data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

class DataTransformation:
    @staticmethod
    def build_preprocessor():
        """
       A simple Scaling pipeline for numeric features 
        """
        pipeline = Pipeline(
            steps = [
                ("scaler",StandardScaler())
            ]
        )
        
        return pipeline
            
    def fit_preprocessor(preprocessor, X_train):
        """
        Fit the preprocessor on X_train
        """
        return preprocessor.fit(X_train)
        
    def transform_with_preprocessor(preprocessor, X):
        """
        Transform X using the fitted preprocessor
        """
        return preprocessor.transform(X)
            