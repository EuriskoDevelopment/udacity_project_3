from fastapi import FastAPI
from pydantic import BaseModel

from typing import Union 
from pydantic import BaseModel

from starter.starter.ml.model import inference
import pickle
#from starter.utils import ROOT_DIR
from sklearn.model_selection import train_test_split
from starter.starter.ml.data import process_data
import pandas as pd
import os

# Declare the data object with its components and their type.
class TaggedItem(BaseModel):
    name: str
    tags: Union[str, list] 
    item_id: int

app = FastAPI()

#def load_data():
#
#    return model, data


@app.get("/")
async def get_root():
    return {"welcome to this amazing app"}

# This allows sending of data (our TaggedItem) via POST to the API.
@app.post("/inference/")
async def model_inference():

    model = pickle.load(open(os.path.join(os.getcwd(), "starter", "model", "lr_model.pkl"), 'rb'))
    encoder = pickle.load(open(os.path.join(os.getcwd(), "starter", "model", "encoder.pkl"), 'rb'))
    lb = pickle.load(open(os.path.join(os.getcwd(), "starter", "model", "label_binarizer.pkl"), 'rb'))
    path_to_data = os.path.join(os.getcwd(), "data","clean_data.csv")
    data = pd.read_csv(path_to_data)

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
    
    X, _, _, _ = process_data(
        data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    preds = inference(model, X)
    return {"Prediction output": int(preds[0])}

@app.get("/items/{item_id}")
async def get_items(item_id: int, count: int = 1):
    return {"fetch": f"Fetched {count} of {item_id}"}