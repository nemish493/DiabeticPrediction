# Diabetes Prediction

Members: Parth Navadiya(22202538) And Nemish Kyada 22212034

Project Title: Diabetes Prediction

Link to repository:  https://mygit.th-deg.de/kickers-as/Diabetes-Prediction


## Description

A Machine Learning Model to predict Diabetes of a patient. This project uses Logistic Regression Model to train on a given dataset. Interactive GUI is developed using PyQt5.

The file [description.txt](description.txt) contains insights and description of each and every variable in dataset.

The Diabetes prediction dataset is a collection of medical and demographic data
from patients, along with their diabetes status (positive or negative).
The data includes features such as age, gender, body mass index (BMI), 
hypertension, heart disease, smoking history, HbA1c level, and blood glucose level. 
This dataset can be used to build machine learning models to predict diabetes 
in patients based on their medical history and demographic information. 
This can be useful for healthcare professionals in identifying patients 
who may be at risk of developing diabetes and in developing personalized 
treatment plans. Additionally, the dataset can be used by researchers to explore 
the relationships between various medical and demographic factors and 
the likelihood of developing diabetes.


There is two graph and one is heatmap and one is histograph in our system predict, patient has diabetes or not so histograph already shown the result and also in run console of pycharm app, print the accuracy and patient details which he/she input in output window. and in every user input it predict patient has diabetes or not with system accuracy.


## Installation

- Download this git repository or clone it to your system using following command:
```
git clone https://mygit.th-deg.de/kickers-as/Diabetes-Prediction
```
`Note: Python3.11 is the recommanded Python version for this project. Install and add it to PATH incase there are any errors. and also setup the virtual environment before the run.`

- Install required python packages from [requirements.txt](requirements.txt) file using following command:
```
pip install -r requirements.txt
```
- Double click and run [test.py](main.py) file to use the prediction model.

## Prerequisites
1. numpy==1.22.3
2. pandas==1.4.1
3. seaborn==0.11.2
4. PyQt6==6.2.3
5. scikit-learn==1.0.2
6. matplotlib==3.5.1


## Work done: 

Navadiya Parth :
1. Integrating Logistic Regression model to main file in order to get predictions.
2. Modifying and Preparing Data with use of numpy arrays.
3. Training Working Logistic Regression Model with help of Scikit-learn.

Nemish kyada :
1. Creating a GUI Interface using PyQt6.
2. Getting User-Inputs using different GUI elements.
3. Worked on Collection and Preprocessing of Dataset using Pandas. 
