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

app = FastAPI()

# Declare the data object with its components and their type.
class TaggedItem(BaseModel):
    name: str
    tags: Union[str, list] 
    item_id: int

class Attributes(BaseModel):
    age : int
    workclass : str
    fnlgt : int
    education : str
    education_num : int
    marital_status : str
    occupation : str
    relationship : str
    race : str
    sex : str
    capital_gain : int
    capital_loss : int
    hours_per_week : int
    native_country : str



@app.get("/")
async def get_root():
    return {"welcome to this machine learning app to predict wether someone is making above 50K in salary"}

# This allows sending of data (our TaggedItem) via POST to the API.
@app.post("/inference/")
async def model_inference(attributes: Attributes):

    model = pickle.load(open(os.path.join(os.getcwd(), "starter", "model", "lr_model.pkl"), 'rb'))
    encoder = pickle.load(open(os.path.join(os.getcwd(), "starter", "model", "encoder.pkl"), 'rb'))
    lb = pickle.load(open(os.path.join(os.getcwd(), "starter", "model", "label_binarizer.pkl"), 'rb'))

    attributes_dict = attributes.dict(by_alias=True)
    data = pd.DataFrame(attributes_dict, index=[0])

    #path_to_data = os.path.join(os.getcwd(), "data","clean_data.csv")
    #data = pd.read_csv(path_to_data)

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
    return {"Salary greater than 50K": int(preds[0])}

@app.get("/items/{item_id}")
async def get_items(item_id: int, count: int = 1):
    return {"fetch": f"Fetched {count} of {item_id}"}