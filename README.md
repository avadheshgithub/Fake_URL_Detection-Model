# 🎯 Phishing URL Detection using Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![ML Project](https://img.shields.io/badge/Machine%20Learning-Phishing%20Detection-red)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Notebook](https://img.shields.io/badge/Platform-Jupyter-informational)

---

## 📌 Overview - 

Phishing is one of the most common cyber-attacks targeting users via malicious links. This project leverages various **machine learning algorithms** to build a predictive model that detects whether a given URL is phishing or legitimate.

> ⚠️ Real-time web security is critical. This project aims to contribute to safer internet browsing using intelligent systems.

> Url - https://fake-url-detection-model-2.onrender.com

---

## 🧠 Models Used

The following models were trained and evaluated using a labeled dataset of phishing and legitimate URLs:

| Algorithm               | Accuracy    | Precision   | Recall      | F1 Score    |
|------------------------|-------------|-------------|-------------|-------------|
| ✅ **Random Forest**        | **97.21%**   | 0.97        | 0.97        | 0.97        |
| ✅ Decision Tree         | 93.11%      | 0.93        | 0.93        | 0.93        |
| ✅ Logistic Regression   | 91.78%      | 0.92        | 0.92        | 0.92        |
| ✅ K-Nearest Neighbors   | 89.92%      | 0.90        | 0.89        | 0.89        |
| ✅ Gaussian NB           | 87.68%      | 0.88        | 0.88        | 0.88        |
| ✅ SVM                   | 94.21%      | 0.94        | 0.94        | 0.94        |

🏆 **Random Forest** was the top-performing model and selected for final deployment.

---

## 🛠️ Tech Stack

| Component         | Tech Used             |
|------------------|-----------------------|
| 👩‍💻 Programming    | Python 3.9+            |
| 📚 Libraries      | Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn |
| 🔍 ML Algorithms  | Random Forest, Decision Tree, Logistic Regression, KNN, SVM, Gaussian NB |
| 📁 Dataset        | Public phishing URL dataset from Kaggle/UCI |
| 📓 Environment    | Jupyter Notebook      |

---

Url : - https://fake-url-detection-model-2.onrender.com

# Web App Interface
![Screenshot 2025-04-02 121417](https://github.com/user-attachments/assets/239f67d8-04b1-4d50-8ff7-a642422d09a7)

![Screenshot 2025-04-02 121505](https://github.com/user-attachments/assets/202ff6d8-9939-4f05-940b-96e53c8c4e46)

## Introduction

The Internet has become an indispensable part of our lives, However, it also has provided opportunities to perform malicious activities like phishing anonymously. Phishers try to deceive their victims by social engineering or creating mockup websites to steal information such as account ID, username, and password from individuals and organizations. Although many methods have been proposed to detect phishing websites, phishing methods have evolved to escape from these detection methods. One of the most successful methods for detecting these malicious activities is Machine Learning. This is because most Phishing attacks have some common characteristics that machine learning methods can identify.


## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/avadheshgithub/Fake_URL_Detection-Model.git

2. Navigate to the project directory:

   ```shell
   cd Phishing-URL-Detection

3. Install the required dependencies:

   ```shell
   pip install -r requirements.txt
   
4. Run the application:

   ```shell
   python app.py

## Directory Tree 
```
├── pickle
│   ├── model.pkl
├── static
    ├──  Images
│      ├── Interface.png
       ├── Result.png
│   ├── styles.css
├── templates
│   ├── index.html
    ├── Result.html
├── Phishing URL Detection.ipynb
├── Procfile
├── README.md
├── app.py
├── feature.py
├── phishing.csv
├── requirements.txt


```

## Technologies Used

1. Python/Flask
2. Numpy
3. Pandas
4. Matplotlib
5. Scikit learn
6. VS Code


## Result

Accuracy of various model used for URL detection
<br>

<br>

||ML Model|	Accuracy|  	f1_score|	Recall|	Precision|
|---|---|---|---|---|---|
0|	Gradient Boosting Classifier|	0.974|	0.977|	0.994|	0.986|
1|	CatBoost Classifier|	        0.972|	0.975|	0.994|	0.989|
2|	XGBoost Classifier| 	        0.969|	0.973|	0.993|	0.984|
3|	Multi-layer Perceptron|	        0.969|	0.973|	0.995|	0.981|
4|	Random Forest|	                0.967|	0.971|	0.993|	0.990|
5|	Support Vector Machine|	        0.964|	0.968|	0.980|	0.965|
6|	Decision Tree|      	        0.960|	0.964|	0.991|	0.993|
7|	K-Nearest Neighbors|        	0.956|	0.961|	0.991|	0.989|
8|	Logistic Regression|        	0.934|	0.941|	0.943|	0.927|
9|	Naive Bayes Classifier|     	0.605|	0.454|	0.292|	0.997|




## Conclusion
Our Project/system is ready to use

All the best | Thank you
