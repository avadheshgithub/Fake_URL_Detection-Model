# Fake URL Detection 

Url : - https://fake-url-detection-model-2.onrender.com

# Web App Interface
![Screenshot 2025-04-02 121417](https://github.com/user-attachments/assets/239f67d8-04b1-4d50-8ff7-a642422d09a7)

![Screenshot 2025-04-02 121505](https://github.com/user-attachments/assets/202ff6d8-9939-4f05-940b-96e53c8c4e46)

## Introduction

The Internet has become an indispensable part of our life, However, It also has provided opportunities to anonymously perform malicious activities like Phishing. Phishers try to deceive their victims by social engineering or creating mockup websites to steal information such as account ID, username, password from individuals and organizations. Although many methods have been proposed to detect phishing websites, Phishers have evolved their methods to escape from these detection methods. One of the most successful methods for detecting these malicious activities is Machine Learning. This is because most Phishing attacks have some common characteristics which can be identified by machine learning methods.


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
