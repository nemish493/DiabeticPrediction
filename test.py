import sys
import numpy as np
import pandas as pd
import seaborn as sns
from PyQt6.QtGui import QFont, QDoubleValidator, QIntValidator
from PyQt6.QtWidgets import QApplication, QTabWidget, QLineEdit, QComboBox, QVBoxLayout, QLabel, QPushButton, QWidget, QSpinBox
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Data preprocessing
label_encoder = LabelEncoder()
data = pd.read_csv('diabetes_prediction_dataset.csv')
print("Null Values:\n", data.isnull().sum())  # Print null values for better visibility
data.drop_duplicates(inplace=True)
data['sex'] = label_encoder.fit_transform(data['sex'])
smoking_mapping = {'never': 2, 'No Info': 3, 'current': 0, 'former': 1, 'ever': 4, 'not current': 5}
data['smoking_history'] = data['smoking_history'].map(smoking_mapping)

# Feature scaling
X = data.drop(columns='diabetes', axis=1)
Y = data['diabetes']
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split and model training
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
model = LogisticRegression()
model.fit(x_train, y_train)

# Model evaluation
x_test_predict = model.predict(x_test)
x_test_accu = accuracy_score(x_test_predict, y_test)
print("Accuracy on test data: ", x_test_accu)

# Visualization in a PyQt window
class Window(FigureCanvas):
    def __init__(self, parent):
        # Initialize the matplotlib figure for visualizations
        fig, ax = plt.subplots(2, figsize=(8, 8), gridspec_kw={'hspace': 0.5})
        fig.subplots_adjust(left=0.4, right=1)
        super().__init__(fig)
        self.setParent(parent)

        # Heatmap for correlation visualization
        heatmap = sns.heatmap(data.corr(), annot=True, cmap='YlGnBu', ax=ax[0])
        heatmap_position = [0.20, 0.71, 0.7, 0.25]
        heatmap.set_position(heatmap_position)
        cbar = heatmap.collections[0].colorbar
        cbar_position = [0.92, 0.71, 0.02, 0.25]
        cbar.ax.set_position(cbar_position)

        # Histogram for diabetes scale distribution
        histogram_position = [0.2, 0.10, 0.8, 0.4]
        plt.hist([data[data.diabetes == 1].diabetes, data[data.diabetes == 0].diabetes],
                 color=["green", "red"], label=["label=yes", "label=no"])
        plt.legend()
        ax[1].set_position(histogram_position)
        ax[1].set_ylabel('Number of People')
        ax[1].set_xlabel('Diabetes Scale')

# Main application window
class AppMain(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Diabetes Prediction")
        self.setGeometry(100, 100, 900, 900)
        self.setStyleSheet("background-color: #F0F0F0;")
        Window(self)

# User input and prediction window
class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Diabetes Prediction")
        self.setGeometry(100, 100, 1400, 800)
        self.setStyleSheet("background-color: #FFFFFF;")
        self.font = QFont("Arial", 10)
        self.layout = QVBoxLayout(self)

        # User input form
        self.head = QLabel("Please Enter Patient's details. ", self)
        self.head.setFont(QFont("Arial", 12, weight=QFont.Weight.Bold))
        self.head.setFixedWidth(450)
        self.head.move(1000, 150)

        self.label_sex = QLabel("Sex:", self)
        self.label_sex.move(900, 220)
        self.label_sex.setFont(self.font)
        self.value_sex = QComboBox(self)
        self.value_sex.addItem("Male")
        self.value_sex.addItem("Female")
        self.value_sex.move(1150, 220)
        self.value_sex.resize(150, 30)

        self.label_age = QLabel("Age:", self)
        self.label_age.move(900, 280)
        self.label_age.setFont(self.font)
        self.value_age = QSpinBox(self)
        self.value_age.setMinimum(20)
        self.value_age.setMaximum(100)
        self.value_age.move(1150, 280)
        self.value_age.resize(150, 30)
        self.value_age_label = QLabel("20", self)
        self.value_age_label.move(1600, 280)
        self.value_age_label.setFont(self.font)
        self.value_age.valueChanged.connect(self.update_value_age_label)

        self.label_hypertension = QLabel("Hypertension:", self)
        self.label_hypertension.setFont(self.font)
        self.label_hypertension.move(900, 340)
        self.value_hypertension = QComboBox(self)
        self.value_hypertension.addItem("Yes")
        self.value_hypertension.addItem("No")
        self.value_hypertension.move(1150, 340)
        self.value_hypertension.resize(150, 30)

        self.label_heart_disease = QLabel("Heart Disease:", self)
        self.label_heart_disease.move(900, 400)
        self.label_heart_disease.setFont(QFont("Arial", 10))
        self.label_heart_disease.setFixedWidth(150)
        self.value_heart_disease = QComboBox(self)
        self.value_heart_disease.addItem("Yes")
        self.value_heart_disease.addItem("No")
        self.value_heart_disease.move(1150, 400)
        self.value_heart_disease.resize(150, 30)

        self.label_smoking_history = QLabel("Smoking History:", self)
        self.label_smoking_history.move(900, 460)
        self.label_smoking_history.resize(150, 30)
        self.label_smoking_history.setFont(self.font)
        self.value_smoking_history = QComboBox(self)
        self.value_smoking_history.addItem("current")
        self.value_smoking_history.addItem("former")
        self.value_smoking_history.addItem("never")
        self.value_smoking_history.addItem("No Info")
        self.value_smoking_history.addItem("ever")
        self.value_smoking_history.addItem("not current")
        self.value_smoking_history.resize(150, 30)
        self.value_smoking_history.move(1150, 460)

        self.label_bmi = QLabel("BMI(10.00 - 95.70):", self)
        self.label_bmi.move(900, 520)
        self.label_bmi.setFont(self.font)
        self.label_bmi.setFixedWidth(150)
        self.value_bmi = QLineEdit(self)
        self.value_bmi.setValidator(QDoubleValidator(1.0, 50.0, 2, self))
        self.value_bmi.move(1150, 520)
        self.value_bmi.resize(150, 30)
        bmi_validator = QDoubleValidator(10.0, 95.7, 2, self)

        self.label_hba1c = QLabel("HbA1c Level(3.5 - 9.0):", self)
        self.label_hba1c.setFont(self.font)
        self.label_hba1c.move(900, 580)
        self.value_hba1c = QLineEdit(self)
        self.label_hba1c.setFixedWidth(200)
        self.value_hba1c.resize(150, 30)
        self.value_hba1c.move(1150, 580)
        hba1c_validator = QDoubleValidator(3.5, 9.0, 2, self)

        self.label_blood_glucose = QLabel("Blood Glucose Level (80 - 300):", self)
        self.label_blood_glucose.move(900, 640)
        self.label_blood_glucose.setFixedWidth(250)
        self.label_blood_glucose.setFont(self.font)
        self.value_blood_glucose = QLineEdit(self)
        blood_glucose_validator = QIntValidator(80, 300, self)
        self.value_blood_glucose.resize(150, 30)
        self.value_blood_glucose.move(1150, 640)

        self.button = QPushButton("Submit", self)
        self.button.setCheckable(True)
        self.button.setGeometry(1050, 675, 150, 30)
        self.button.setStyleSheet("background-color: #4CAF50; color: black; font-weight: bold;")
        self.button.setCheckable(True)

        self.button.clicked.connect(self.predict_disease)
        self.value_bmi.setValidator(bmi_validator)
        self.value_hba1c.setValidator(hba1c_validator)
        self.value_blood_glucose.setValidator(blood_glucose_validator)

        self.prediction = QLabel("", self)
        self.prediction.setGeometry(900, 715, 300, 50)
        self.prediction.setFont(QFont("Arial", 12, weight=QFont.Weight.Bold))
        self.prediction.setStyleSheet(
            "color: #FF0000; background-color: #bcd95f; border: 1px solid #E0E0E0; border-radius: 5px; padding: 5px;")

        self.tab = QWidget(self)
        self.addTab(main, "Prediction Model")

    def update_value_age_label(self, value):
        self.value_age_label.setText(str(value))

    # Process user input and make predictions
    def predict_disease(self):
        try:
            # Retrieve user input
            sex = self.value_sex.currentText()
            age = self.value_age.value()
            blood_glucose = self.value_blood_glucose.text()
            hypertension = self.value_hypertension.currentText()
            heart_disease = self.value_heart_disease.currentText()
            bmi = self.value_bmi.text()
            hba1c = self.value_hba1c.text()
            smoking_history = self.value_smoking_history.currentText()

            # Check if all required fields are filled
            if not blood_glucose or not bmi or not hba1c:
                self.prediction.setText("Please fill all the details!")
                return

            # Convert categorical data to numerical format
            sex = label_encoder.transform([sex])[0]
            hypertension = 1 if hypertension == "Yes" else 0
            heart_disease = 1 if heart_disease == "Yes" else 0
            smoking_history = smoking_mapping.get(smoking_history, 0)

            # Create an array from the user input
            user_input = [int(sex), age, int(hypertension), int(heart_disease), int(smoking_history), float(bmi),
                          float(hba1c), int(blood_glucose)]

            # Reshape and scale the input data
            user_input = np.asarray(user_input).reshape(1, -1)
            user_input = scaler.transform(user_input)

            # Make a prediction
            predict = model.predict(user_input)

            # Display the prediction result
            if predict[0] == 0:
                self.prediction.setText("Patient doesn't have Diabetes :)")
            else:
                self.prediction.setText("Patient has Diabetes!!")

        except Exception as e:
            # If an error occurs, print it to the console and show an error message in the GUI
            print(f"Error during prediction: {e}")
            self.prediction.setText("Error in prediction. Check the console for details.")


app = QApplication(sys.argv)
main = AppMain()
final = MainWindow()
main.show()
final.show()
sys.exit(app.exec())
