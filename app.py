# Importing required libraries
from flask import Flask, request, render_template
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
from feature import FeatureExtraction  # Assuming this is in feature.py

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
file_path = "pickle/model.pkl"
try:
    with open(file_path, "rb") as file:
        gbc = pickle.load(file)
except FileNotFoundError:
    print(f"Error: {file_path} not found. Please ensure the model file exists.")
    exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        
        # Validate URL
        if not url:
            return render_template('index.html', error="Please enter a URL")
        
        try:
            # Extract features using FeatureExtraction class
            obj = FeatureExtraction(url)
            x = np.array(obj.getFeaturesList()).reshape(1, 30)  # 30 features
            
            # Predict and get probability of being safe
            y_pro_non_phishing = gbc.predict_proba(x)[0, 1]  # Probability of being safe (class 1)
            
            # Render result.html with the safe probability
            return render_template('result.html', url=url, xx=y_pro_non_phishing)
        
        except Exception as e:
            print(f"Prediction error: {e}")
            return render_template('index.html', error="An error occurred while analyzing the URL")
    
    # Render the input form for GET requests
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)