import pytest
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from starter.starter.ml.model import train_model, inference, compute_model_metrics
from starter.starter.ml.data import process_data
from sklearn.exceptions import NotFittedError
from numpy import dtype
import numpy as np


@pytest.fixture(scope='session')
def load_data():
    path_to_data = os.path.join(os.getcwd(), "data", "clean_data.csv")
    data = pd.read_csv(path_to_data)
    return data


@pytest.fixture(scope='session')
def load_model_and_encoder():
    model = pickle.load(open(os.path.join(os.getcwd(), "starter", "model", "rf_model.pkl"), 'rb'))
    encoder = pickle.load(open(os.path.join(os.getcwd(), "starter", "model", "encoder.pkl"), 'rb'))
    lb = pickle.load(open(os.path.join(os.getcwd(), "starter", "model", "label_binarizer.pkl"), 'rb'))
    return model, encoder, lb


def test_train_model_gives_binary_predictions(load_data):
    data = load_data

    train, test = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    _, lr_model = train_model(X_train, y_train)

    try:
        pred = lr_model.predict(X_test)
    except NotFittedError as e:
        print(repr(e))
    
    # Predictions can either be 0 or 1 (salary less or more than 50K)
    assert np.all((pred == 0)|(pred == 1))


def test_computer_model_metrics():
    assert 1 == 1
    preds = np.full((10, 1), 1)
    y = np.full((10, 1), 0)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision.dtype == dtype('float64')
    assert recall.dtype == dtype('float64')
    assert fbeta.dtype == dtype('float64')


def test_model_inference_above_than_50k(load_model_and_encoder):
    model, encoder, lb = load_model_and_encoder
    
    attributes = {
        "age": 45,
        "workclass": "Private",
        "fnlgt": 280464,
        "education": "Doctorate",
        "education_num": 10,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 510,
        "capital_loss": 0,
        "hours_per_week": 60,
        "native_country": "United-States"
    }
    
    data = pd.DataFrame(attributes, index=[0])

    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]

    X, _, _, _ = process_data(
        data, categorical_features=cat_features, training=False, encoder=encoder, lb=lb
    )

    preds = inference(model, X)
    is_greater_than_50k = int(preds[0])
    assert is_greater_than_50k


def test_model_inference_less_than_50k(load_model_and_encoder):
    model, encoder, lb = load_model_and_encoder
    
    attributes = {
        "age": 21,
        "workclass": "Private",
        "fnlgt": 101509,
        "education": "Some-college",
        "education_num": 3,
        "marital_status": "Married-civ-spouse",
        "occupation": "Other-services",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 50,
        "hours_per_week": 10,
        "native_country": "United-States"
    }
    
    data = pd.DataFrame(attributes, index=[0])

    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]

    X, _, _, _ = process_data(
        data, categorical_features=cat_features, training=False, encoder=encoder, lb=lb
    )

    preds = inference(model, X)
    is_greater_than_50k = int(preds[0])
    assert is_greater_than_50k == 0
