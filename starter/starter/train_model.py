# Script to train machine learning model.
import os
import pandas as pd
import pickle
import logging

from sklearn import linear_model
from starter.starter.ml.model import train_model, inference, compute_model_metrics, slice_performance
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from starter.starter.ml.data import process_data
import csv

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()
# Add code to load in the data.


path_to_data = os.path.join(os.getcwd(), "data","clean_data.csv")
data = pd.read_csv(path_to_data)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
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

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
with open(os.path.join(os.getcwd(), "starter", "model", "label_binarizer.pkl"), 'wb') as f:
    pickle.dump(lb, f)
with open(os.path.join(os.getcwd(), "starter", "model", "encoder.pkl"), 'wb') as f:
    pickle.dump(encoder, f)

# Train and save a model.
logger.info("Train model")
_, lr_model = train_model(X_train, y_train)
logger.info("Save model")
with open(os.path.join(os.getcwd(), "starter", "model", "lr_model.pkl"), 'wb') as file:
    pickle.dump(lr_model, file)

y_pred = inference(lr_model, X_test)

# Compute r2 and MAE
logger.info("Scoring")
r_squared = lr_model.score(X_test, y_test)

#y_pred = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

logger.info(f"Score: {r_squared}")
logger.info(f"MAE: {mae}")

#precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
#logger.info(f"precision: {precision}")
#logger.info(f"recall: {recall}")
#logger.info(f"fbeta: {fbeta}")

slice_performance(lr_model, data, encoder, lb, os.path.join(os.getcwd(), "starter", "model", "slice_performance.csv"))
