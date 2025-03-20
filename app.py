from flask import Flask, request, jsonify
import pandas as pd
import os
import pickle
from flask_cors import CORS
from flask_jwt_extended import create_access_token, jwt_required, JWTManager, get_jwt_identity
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize Flask app
app = Flask(__name__)

# CORS
CORS(app)

app.config["JWT_SECRET_KEY"] = "supersecretkey"
jwt = JWTManager(app)

# Load the trained model and vectorizer from disk
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

#for pie chart data
#Load datasets
dataset_folder = os.path.join(os.path.dirname(__file__), "datasets")
bullying_file = os.path.join(dataset_folder,"Aggressive_All.csv")
non_bullying_file = os.path.join(dataset_folder,"Non_Aggressive_All.csv")

bullying_data = pd.read_csv(bullying_file,encoding="utf-8")
non_bullying_data = pd.read_csv(non_bullying_file,encoding="utf-8")

bullying_data["label"] = 1
non_bullying_data["label"] = 0

df = pd.concat([bullying_data,non_bullying_data],ignore_index=True)
df = df.dropna(subset=['Message'])

class_counts = df['label'].value_counts()
total = class_counts.sum()

bullying_percentage = (class_counts[1] / total) * 100 if 1 in class_counts else 0
non_bullying_percentage = (class_counts[0] / total) * 100 if 0 in class_counts else 0

VALID_USERNAME = "Admin"
VALID_PASSWORD = "Admin"

# login
@app.route('/login',methods=['POST'])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    if username == VALID_USERNAME and password == VALID_PASSWORD:
        access_token = create_access_token(identity=username)
        return jsonify({"token" : access_token, "username" : username}),200
    else:
        return jsonify({"message" : "Invalid credentials"}),401

@app.route('/')
def home():
    return "Welcome to the Cyberbullying Detection API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the user (assuming it's sent as JSON)
    input_data = request.get_json()  # JSON input
    input_text = input_data['text']  # Get the text field from the JSON input

    # Vectorize the input text using the loaded vectorizer
    input_text_vec = vectorizer.transform([input_text])

    print("input text:",input_text_vec.shape)
    # Predict using the loaded model
    prediction = model.predict(input_text_vec)

    print("Prediction:", prediction) 

    # Return the prediction result as a response
    result = "Bullying" if prediction[0] == 1 else "Non-Bullying"
    return jsonify({"prediction": result})

@app.route('/data-distribution',methods=['GET'])
def get_data_distribution():
    return jsonify({
    "bullying" : bullying_percentage,
    "nonbullying" : non_bullying_percentage
})

if __name__ == '__main__':
    app.run(debug=True)
