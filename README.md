# Diabetes_Predictions
# 🩺 Diabetes Risk Prediction App

A web-based application to predict **diabetes risk probability** using machine learning, built with **Streamlit** and trained on public health survey data.

> ⚠️ This application is intended for **educational and informational purposes only** and **is not a medical diagnosis tool**.

## 🚀 Live Demo
🔗 **Streamlit App**:  
👉 _Add your Streamlit Cloud URL here_

## 📌 Overview

This application allows users to:
- Input personal, health, lifestyle, and socioeconomic data
- Automatically calculate BMI from height & weight
- Predict the **probability of diabetes risk** using a trained ML pipeline
- View results in a clean, step-by-step (wizard-style) interface

The model is trained on a **balanced diabetes dataset** to avoid biased predictions.


## 🧠 Machine Learning Model

- **Algorithm**: Logistic Regression
- **Training Approach**:
  - End-to-end **Scikit-learn Pipeline**
  - Feature scaling handled inside the pipeline
  - Consistent preprocessing between training & inference
- **Output**:
  - Probability of diabetes risk (not just binary yes/no)

### Why Pipeline?
Using a pipeline ensures:
- No feature mismatch
- No double scaling
- Stable predictions in production (Streamlit Cloud)


## 📊 Dataset

- **Source**: BRFSS 2015 (Behavioral Risk Factor Surveillance System)
- **Target**: Diabetes (binary)
- **Key Features**:
  - Health indicators (BMI, blood pressure, cholesterol)
  - Lifestyle habits (smoking, physical activity, diet)
  - Mental & physical health days
  - Socioeconomic factors (education, income)

> The dataset used for training is **balanced** to prevent model bias toward the majority class.


## 🧩 Application Flow (UX)

1. **Personal Information**
   - Age, gender, height, weight
   - BMI calculated automatically

2. **Medical History**
   - Blood pressure, cholesterol, heart disease, stroke
   - General, mental, and physical health

3. **Lifestyle Habits**
   - Physical activity
   - Diet (fruits & vegetables)
   - Smoking & alcohol consumption

4. **Social & Economic Factors**
   - Education level
   - Income range
   - Healthcare access

5. **Prediction Result**
   - Risk probability
   - Visual indicator
   - Health recommendations


## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **Scikit-learn**
- **Pandas & NumPy**
- **Joblib**

## ⚙️ Installation & Local Run

```bash
# Clone repository
git clone https://github.com/USERNAME/Diabetes_Predictions.git
cd Diabetes_Predictions

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py


