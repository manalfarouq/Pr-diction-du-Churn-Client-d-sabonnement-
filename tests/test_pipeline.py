import pandas as pd
import pytest
from pipeline import split_data


df = pd.read_csv("../data/data.csv")

@pytest.fixture
def data_split():
    
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y

def test_split_data_dimensions(data_split):

    X, y = data_split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Vérifier que X et y ont la même longueur
    assert len(X) == len(y)
    
    # Vérifier que le split conserve la correspondance lignes/étiquettes
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
