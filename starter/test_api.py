from fastapi.testclient import TestClient
from main import app
from numpy import dtype, reshape
from fastapi.testclient import TestClient
import pytest

from main import app

client = TestClient(app)


def test_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json().dtype == dtype.str 


def test_inference_above_50k():
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
    response = client.post("/items", json=attributes)
    assert response.status_code == 200
    assert "Salary greater than" in response.json() 