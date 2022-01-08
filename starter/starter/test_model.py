import pytest
import os
import pandas as pd
import pickle
#from starter.utils import ROOT_DIR
from sklearn.model_selection import train_test_split
from starter.ml.model import train_model, inference, compute_model_metrics
from starter.ml.data import process_data
from sklearn.exceptions import NotFittedError
from numpy import dtype
import numpy as np

@pytest.fixture(scope='session')
def load_data():
    path_to_data = os.path.join(os.getcwd(), "data","clean_data.csv")
    data = pd.read_csv(path_to_data)
    return data

@pytest.fixture(scope='session')
def load_model():
    model = pickle.load(open(os.path.join(os.getcwd(), "starter", "model", "unit_test_rf_model.pkl"), 'rb'))
    return model


def test_train_model(load_data):
    print(os.getcwd())
    data = load_data

    train, test = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    rf_model = train_model(X_train, y_train)
    with open(os.path.join(os.getcwd(), "starter", "model", "unit_test_rf_model.pkl"), 'wb') as file:
        pickle.dump(rf_model, file)
    
    try:
        rf_model.predict(X_test)
    except NotFittedError as e:
        print(repr(e))
    


def test_computer_model_metrics():
    assert 1 == 1
    preds = np.full((10, 1), 1)
    y = np.full((10, 1), 0)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision.dtype == dtype('float64')
    assert recall.dtype == dtype('float64')
    assert fbeta.dtype == dtype('float64')


def test_save_load(load_data, load_model):
    assert 1 == 1
    rf_model = load_model
    data = load_data

    train, test = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    
    try:
        rf_model.predict(X_test)
    except NotFittedError as e:
        print(repr(e))

