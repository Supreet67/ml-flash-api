from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Get user input from form
        education = request.form["education"]
        experience = int(request.form["experience"])
        job = request.form["job"]
        age = int(request.form["age"])
        gender = request.form["gender"]

        # Mapping inputs - must match training time!
        edu_map = {'10th': 0, '12th': 1, 'Bachelors': 2, 'Masters': 3, 'PhD': 4}
        gender_map = {'Male': 0, 'Female': 1}
        job_map = {
            'Engineer': 0,
            'Manager': 1,
            'Data Scientist': 2,
            'Teacher': 3,
            'Clerk': 4,
            'Doctor': 5
            # Add more job titles if used during training
        }

        # Basic feature array (can be 5 values)
        raw_features = [
            age,
            edu_map.get(education, 0),
            experience,
            gender_map.get(gender, 0),
            job_map.get(job, 0)
        ]

        # Expand to 14 features (dummy values) to match model input size
        features = raw_features + [0] * (14 - len(raw_features))

        # Make prediction
        prediction = model.predict([features])[0]

        return render_template("index.html", prediction=round(prediction, 2))

    except Exception as e:
        # Return a helpful error message to the user
        return f"<h2 style='color:red;'>❌ Error: {str(e)}</h2>"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
