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
import subprocess


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    os.system("dvc remote add -df s3-bucket s3://euriskoudacityproject3")
    print("AWS set up")
    dvc_output = subprocess.run(
    ["dvc", "pull"], capture_output=True, text=True)
    print(dvc_output.stdout)
    print(dvc_output.stderr)
    if dvc_output.returncode != 0:
        print("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


app = FastAPI()

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
    return {"Welcome to this machine learning app to predict wether someone is making above 50K in salary"}

# This allows sending of data (our TaggedItem) via POST to the API.
@app.post("/inference")
async def model_inference(attributes: Attributes):

    model = pickle.load(open(os.path.join(os.getcwd(), "starter", "model", "rf_model.pkl"), 'rb'))
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
    is_greater_than_50k = int(preds[0])
    if is_greater_than_50k:
        return {"Predicted salary, based on input attributes, is greater than 50K"}
    else:
        return {"Predicted salary, based on input attributes, is less than 50K"}

@app.get("/items/{item_id}")
async def get_items(item_id: int, count: int = 1):
    return {"fetch": f"Fetched {count} of {item_id}"}