from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    education = request.form["education"]
    experience = int(request.form["experience"])
    job = request.form["job"]
    age = int(request.form["age"])
    gender = request.form["gender"]

    # Convert to numerical features — adjust as per your ML preprocessing
    # You must encode string features the same way you did during training!
    # Here’s just an example (update this as per your model):
    edu_map = {'10th': 0, '12th': 1, 'Bachelors': 2, 'Masters': 3, 'PhD': 4}
    gender_map = {'Male': 0, 'Female': 1}

    features = [
        age,
        edu_map.get(education, 0),
        experience,
        gender_map.get(gender, 0),
        len(job) % 10  # dummy encoding for job title, update as needed
    ]

    prediction = model.predict([features])[0]

    return render_template("index.html", prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
