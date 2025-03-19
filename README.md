# 🧠 Brain Tumor Detection App

This **Brain Tumor Detection App** is a machine-learning-powered web application that predicts whether a brain tumor is **Benign** or **Malignant**. The app is built using **Logistic Regression** for classification and **Streamlit** for the interactive web interface.

## 📌 Features

- 🧪 **Brain Tumor Classification:** Predicts whether the tumor is **Benign** or **Malignant**.
- 📊 **User Input Form:** Collects patient information like **Age**, **Tumor Size**, **Symptoms**, etc.
- 📈 **Real-time Prediction:** Provides real-time predictions based on user input.
- 📋 **Data Preprocessing:** Categorical data is encoded, and numerical features are scaled.
- 📁 **Model Serialization:** Uses `pickle` to save and load the trained model and encoders.

---

## 📂 Project Structure

```
Brain_Tumor_Detection_App/
├── app/
│    ├── main.py                 # Streamlit app code
│    ├── model.pkl               # Serialized Logistic Regression model
│    └── encoder.pkl             # Serialized OneHotEncoder
├── model/
│    ├── main.py                 # Model training and evaluation
│    └── brain_tumor_data.csv    # Dataset
├── README.md                    # Project documentation
└── requirements.txt             # Python dependencies
```

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Brain_Tumor_Detection_App.git
cd Brain_Tumor_Detection_App
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset
The model is trained on a **Brain Tumor Dataset** with the following key features:

- **Age**: Patient's age
- **Gender**: Male/Female
- **Tumor Size**: Size of the tumor in cm
- **Location**: Tumor location (Frontal, Temporal, Parietal, Occipital)
- **Histology**: Tumor type (Astrocytoma, Glioblastoma, etc.)
- **Symptoms**: Observed symptoms (Headache, Seizures, etc.)
- **Family History**: Whether the patient has a family history of brain tumors

---

## 📚 Model Training

The model uses **Logistic Regression** for classification:

1. **Preprocessing**:
   - OneHotEncode categorical features.
   - Standardize numerical features (Age, Tumor Size).
2. **Model**:
   - Trained a Logistic Regression model using `scikit-learn`.
3. **Saving the Model**:
   - `pickle` is used to serialize and save the trained model and encoder.

### Train the Model
```bash
python model/main.py
```

---

## 🌐 Running the Streamlit App

1. Ensure the model (`model.pkl`) and encoder (`encoder.pkl`) are present in the `app/` folder.

2. Run the Streamlit application:
```bash
streamlit run app/main.py
```

3. Open your browser and navigate to: [http://localhost:8501](http://localhost:8501)

---

## 🧰 Usage Instructions

1. Input patient details in the sidebar (e.g., Age, Gender, Tumor Size, etc.).
2. Click the **"Predict"** button.
3. View the tumor classification result (Benign or Malignant) on the main page.

---


## 📌 Future Improvements

- Integrate more advanced models (e.g., Random Forest, XGBoost).
- Enhance the UI with better visualizations.

---

## 🤝 Contributing

1. Fork the repository.
2. Create a new branch (`feature/new-feature`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a Pull Request.

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 📧 Contact

For any questions or suggestions:
- **Email:** yourname@example.com
- **GitHub:** [yourusername](https://github.com/yourusername)


 
