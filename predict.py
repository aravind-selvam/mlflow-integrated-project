"""Prediction module"""

import os
import pickle

import mlflow
import pandas as pd
import requests
from flask import Flask, flash, jsonify, request, render_template
from pymongo import MongoClient

EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "used-car-prediction")
MLFLOW_ENABLED = os.getenv("MLFLOW_ENABLED", "False") == "True"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DEFAULT_MODEL_ENABLED = os.getenv("DEFAULT_MODEL_ENABLED", "True") == "True"
MONITORING_ENABLED = os.getenv("MONITORING_ENABLED", "False") == "True"
EVIDENTLY_SERVICE_URI = os.getenv("EVIDENTLY_SERVICE_URI", "http://localhost:8085")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
if not os.getenv("MLFLOW_S3_ENDPOINT_URL"):
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

if MONITORING_ENABLED:
    mongo_client = MongoClient(MONGODB_URI)
    db = mongo_client.get_database("prediction_service")
    collection = db.get_collection(EXPERIMENT_NAME)


def load_model_from_registry():
    """
    Loads the ML model from the MLFlow registry
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{EXPERIMENT_NAME}/latest"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    print("Loaded model from S3 Bucket")
    return loaded_model


def load_default_model():
    """
    Loads the default ML model from disk
    """
    with open("pickle/model.pkl", "rb") as f_in:
        loaded_model = pickle.load(f_in)
    print("Loaded default model from disk")
    return loaded_model


def load_model():
    """
    Loads the ML model
    """
    try:
        if MLFLOW_ENABLED:
            return load_model_from_registry()

        if DEFAULT_MODEL_ENABLED:
            return load_default_model()
    except:
        if DEFAULT_MODEL_ENABLED:
            return load_default_model()

    return None

def validate_data(record):
    """
    Performs data validation
    """
    if record["vehicle_age"] < 0 or record["vehicle_age"] > 30:
        return "Vehicle_age should be between 0 and 30 years"
    
    if record["mileage"] < 4.0 or record["mileage"] > 33.0:
        return "Mileage should be between 4.0 and 33.0 years"
    
    if record["max_power"] < 38.0 or record["max_power"] > 626.0:
        return "Max power should be between 38.0 and 626.0 years"
    
    if record["seats"] < 1 or record["seats"] > 9:
        return "Seats should be between 1 and 9"
    
    return None

def predict(model, df):
    """
    Predicts the car market value
    """
    predict = model.predict(df)
    return predict[0]


def save_to_db(record, selling_price):
    """
    Saves the prediction data to the Mongo database
    """
    rec = record.copy()
    rec["selling_price"] = selling_price
    collection.insert_one(rec)


# def send_to_evidently_service(record, selling_price):
#     """
#     Sends the prediction data to the Evidently monitoring service
#     """
#     rec = record.copy()
#     rec["selling_price"] = selling_price
#     requests.post(f"{EVIDENTLY_SERVICE_URI}/iterate/---", json=[rec])


def calculate_selling_price(record):
    """
    Calculates the car's selling_price
    """
    selling_price = predict(record)
    if MONITORING_ENABLED:
        save_to_db(record, selling_price)
        send_to_evidently_service(record, selling_price)
    return selling_price

def load_list():
    """
    Get model and brand list from a pickle file
    """
    file = open("pickle/carnamelist.pkl", 'rb')
    model_list = pickle.load(file)
    brand_list = pickle.load(file)
    file.close()
    return model_list, brand_list

def transform_data(dict_data):
    """ Pre-Process data from request data
    """
    input_data = pd.DataFrame.from_dict(dict_data, orient='index').T
    with open("pickle/preprocess.pkl" , "rb")as f:
        preprocessor = pickle.load(f)
    input_data = preprocessor.transform(input_data)
    
    return input_data


app = Flask(EXPERIMENT_NAME)
app.secret_key = os.urandom(24)

model = load_model()


@app.route("/", methods=["GET", "POST"])
def predict_form_endpoint():
    """
    Prediction form endpoint
    """
    model_list, brand_list = load_list()
    
    if request.method == "POST":
        record = {}
        record["brand"] = request.form.get("brand")
        record["model"] = request.form.get("model")
        record["vehicle_age"] = int(request.form.get("vehicle_age"))
        record["km_driven"] = int(request.form.get("km_driven"))
        record["seller_type"] = request.form.get("seller_type")
        record["fuel_type"] = request.form.get("fuel_type")
        record["transmission_type"] = request.form.get("transmission")
        record["mileage"] = float(request.form.get("mileage"))
        record["engine"] = int(request.form.get("engine"))
        record["max_power"] = float(request.form.get("max_power"))
        record["seats"] = int(request.form.get("seats"))
                        
        input_data = transform_data(record)
        
        error_message = validate_data(record)
        if error_message:
            flash(error_message, 'info')
        else:
            selling_price = predict(model, input_data)
            flash(f"is the market selling price of the Car model {record['model']}", selling_price)

    return render_template("index.html", model_list=model_list, brand_list=brand_list)


@app.route("/predict", methods=["POST"])
def predict_json_endpoint():
    """
    Prediction API endpoint
    """
    record = request.get_json()

    error_message = validate_data(record)
    if error_message:
        return jsonify({"Error": error_message})

    selling_price = calculate_selling_price(record)
    return jsonify({"selling_price": selling_price})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)