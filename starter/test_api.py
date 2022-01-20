from fastapi.testclient import TestClient
from main import app
from fastapi.testclient import TestClient
import pytest
from main import app

client = TestClient(app)


def test_get_malformed():
    r = client.get("/invalid_addr")
    assert r.status_code != 200


def test_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == ["Welcome to this machine learning app to predict wether someone is making above 50K in salary"]


def test_api_inference_above_50k():
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
    response = client.post("/inference", json=attributes)
    assert response.status_code == 200
    assert response.json() == ["Predicted salary, based on input attributes, is greater than 50K"]


def test_api_inference_less_than_50k():
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
    response = client.post("/inference", json=attributes)
    assert response.status_code == 200
    assert response.json() == ["Predicted salary, based on input attributes, is less than 50K"]
