# 🚖 AI-Powered Taxi Fare Prediction System

![Banner](docs/taxi_banner.png)

---

## 📌 Project Overview

This project is an **end-to-end AI system** designed to **predict taxi fares** using real-world trip data. It combines **data preprocessing, feature engineering, machine learning model training, and deployment** into a fully functional **Streamlit web application**.

The model has been trained on the **Taxi Trip Fare Data 2023** dataset from Kaggle, capturing **trip-level details such as passenger count, distance, duration, and payment type**. The goal is to provide **accurate fare predictions** that can be integrated into ride-hailing services, transport analytics dashboards, or cost estimation tools.

---

## 🔑 Key Features

- 📊 **Data Preprocessing**: Cleaning missing values, filtering extreme trips, and applying **One-Hot Encoding** for categorical variables (Payment Type: Card/Cash).
- 🤖 **Machine Learning Model**: A **Random Forest Regressor** trained for robust, non-linear relationships between features and fare amounts.
- 📉 **Performance Metrics**: Evaluation with **R² Score** and **Mean Absolute Error (MAE)** to measure accuracy.
- 🗂 **Model Persistence**: Saving and reusing trained models with `joblib`.
- 🖥 **Interactive Streamlit Web App**: Real-time prediction by entering passenger count, distance, duration, and payment type.
- 📈 **Visualization Tools**: Graphs for training loss curves and **Actual vs Predicted fares** scatter plots.

---

## 🛠️ Tech Stack

- **Language**: Python 🐍
- **Libraries & Frameworks**:
  - `pandas` → Data manipulation & preprocessing
  - `numpy` → Numerical computations
  - `scikit-learn` → ML model training & evaluation (Random Forest Regressor)
  - `matplotlib` → Visualization (training curve, scatter plots)
  - `joblib` → Model saving & loading
  - `streamlit` → Web application deployment

---

## 📂 Project Structure

├── data/ # Dataset (raw and preprocessed taxi trip data)
├── model/ # Saved trained model (taxifare_model.pkl)
├── docs/ # Graphs and images (training curves, scatter plots)
│ ├── training_curve.png
│ ├── scatter_plot.png
│ └── taxi_banner.png
├── training.py # Model training & evaluation script
├── webapp.py # Streamlit web application
├── requirements.txt # Dependencies
└── README.md # Project documentation

yaml
Copy code

---

## ⚙️ Workflow

### 1️⃣ Data Preprocessing

- Removed outliers in trip distance & duration (NYC taxi trips usually < 5km).
- Converted categorical features (`payment_type`) into **one-hot encoded variables** (`payment_type_Cash`, `payment_type_Card`).
- Normalized input ranges for numerical stability.

### 2️⃣ Model Training

- Model: **Random Forest Regressor**
- Training/Test Split: **80/20**
- Evaluation Metrics:
  - **R² Score**: `0.85`
  - **Mean Absolute Error (MAE)**: `0.48`

### 3️⃣ Visualization

#### 📉 Training Curve (Error vs Iterations)

![Training Curve](docs/training_curve.png)

#### 📊 Actual vs Predicted Fares

![Scatter Plot](docs/scatter_plot.png)

---

## 🚀 How to Run

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/taxi-fare-prediction.git
cd taxi-fare-prediction
2️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Train the Model (Optional, pre-trained model included)
bash
Copy code
python training.py
4️⃣ Launch the Web App
bash
Copy code
streamlit run webapp.py
🎛️ Streamlit Web App Features
The app allows you to input real-world parameters:

🧑 Passenger Count (1–10)

📏 Trip Distance (km)

⏱ Trip Duration (minutes)

💳 Payment Type (Cash or Card)

Once submitted, the trained model outputs the predicted taxi fare instantly.

📊 Model Performance
Metric	Value
R² Score	0.85
MAE	0.48

The following scatter plot demonstrates the strong correlation between actual fares vs predicted fares:


🌍 Real-World Applications
Ride-hailing platforms (Uber, Lyft, Careem) for dynamic fare prediction

Fleet operators for cost forecasting

Transportation analytics for trip insights

Smart city projects for mobility optimization

