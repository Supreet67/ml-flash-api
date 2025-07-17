from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load your trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "ML Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)  # expects JSON input
    
    # Example: Suppose your model expects a list of features like [feat1, feat2, feat3]
    features = data['features']  # make sure client sends {"features": [values]}
    
    # Predict using your model
    prediction = model.predict([features])
    
    # Return prediction as JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
