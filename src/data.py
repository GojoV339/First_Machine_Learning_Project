from sklearn import datasets
from sklearn.model_selection import train_test_split

def load_data(test_size: float = 0.1):
    """
    This function loads the data 
    and splits the data and returns 
    splitted data
    """
    
    data = datasets.load_iris()
    X = data.data
    y = data.target
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=42)
    return X_train,X_test,y_train,y_test



