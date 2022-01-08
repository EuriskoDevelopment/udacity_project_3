from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestRegressor
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    #rf_config = {"n_estimators": 101,
    #            "max_depth": 10,
    #            "min_samples_split": 4,
    #            "min_samples_leaf": 3,
    #            "n_jobs": -1,
    #            "criterion": "mae",
    #            "max_features": 0.5,
    #            "oob_score": True}
    rf_pipe = RandomForestRegressor()
    logger.info("Fitting")
    rf_pipe.fit(X_train,y_train)

    return rf_pipe


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds
