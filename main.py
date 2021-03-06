from fastapi import FastAPI
from pydantic import BaseModel
from starter.starter.ml.model import inference
import pickle
from starter.starter.ml.data import process_data
import pandas as pd
import os


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()

class Attributes(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
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
        }


@app.get("/")
async def get_root():
    return {"Welcome to this machine learning app to predict wether someone is making above 50K in salary"}


@app.post("/inference")
async def model_inference(attributes: Attributes):

    model = pickle.load(open(os.path.join(os.getcwd(), "starter", "model", "rf_model.pkl"), 'rb'))
    encoder = pickle.load(open(os.path.join(os.getcwd(), "starter", "model", "encoder.pkl"), 'rb'))
    lb = pickle.load(open(os.path.join(os.getcwd(), "starter", "model", "label_binarizer.pkl"), 'rb'))

    attributes_dict = attributes.dict(by_alias=True)
    data = pd.DataFrame(attributes_dict, index=[0])

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
    if is_greater_than_50k:
        return {"Predicted salary, based on input attributes, is greater than 50K"}
    else:
        return {"Predicted salary, based on input attributes, is less than 50K"}


@app.get("/items/{item_id}")
async def get_items(item_id: int, count: int = 1):
    return {"fetch": f"Fetched {count} of {item_id}"}
