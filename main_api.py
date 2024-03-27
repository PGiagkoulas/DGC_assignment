from flask import Flask, request, jsonify

from utils import read_data_file
import feature_preparation
import model_training
import model_inference

app = Flask(__name__)


# Endpoint to provide general help or documentation
@app.route('/help', methods=['GET'])
def get_help():
    help_text = "Welcome to the API! Here are the available endpoints:\n" \
                "/help - Get API documentation\n" \
                "/prepare_features - Prepare features for model training\n" \
                "/train_model - Train a machine learning model\n" \
                "/infer_model - Infer using the trained model\n"
    return help_text


# Endpoint to prepare features for model training
@app.route('/prepare_features', methods=['POST'])
def prepare_features():

    return jsonify({"message": "Features prepared successfully"})


# Endpoint to train a machine learning model
@app.route('/train_model', methods=['POST'])
def train_model():
    # Your code to train the model goes here
    model_training.train_model()
    return jsonify({"message": "Model trained successfully"})


# Endpoint to infer using the trained model
@app.route('/infer_model', methods=['POST'])
def infer_model():
    request_data = request.get_json()
    # Your code to perform inference using the trained model goes here
    infer_score = model_inference.infer_on_model()
    return jsonify({"message": "Inference performed successfully", "score": infer_score})


if __name__ == '__main__':
    app.run(debug=True)
    # # feature_preparation.prepare_features()
    # # model_training.train_model()
    # model_inference.infer_on_model()
    # pass
