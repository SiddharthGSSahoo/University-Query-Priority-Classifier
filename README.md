# 🎓 University Query Priority Classifier
An end-to-end NLP pipeline that automates student support by classifying queries into High, Medium, or Low priority. Built with Scikit-Learn, it features a custom preprocessing engine (stemming, emoji handling, short-form expansion) and a multi-stage ColumnTransformer pipeline for seamless text and categorical data integration.


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: Scikit-Learn](https://img.shields.io/badge/Framework-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

## 📝 Project Description
This project implements an end-to-end NLP pipeline designed to automate student support desk operations. It classifies incoming student queries into **High**, **Medium**, or **Low** priority levels based on the query text and the target department. 

The system utilizes a custom preprocessing engine and a nested Scikit-Learn Pipeline architecture to handle text vectorization and categorical encoding simultaneously.

## 🚀 Key Features
* **Custom NLP Preprocessor**: Handles lowercasing, punctuation removal, short-form expansion (e.g., "asap" ➔ "as soon as possible"), emoji removal, and Porter Stemming.
* **Nested Pipeline Architecture**: Uses `ColumnTransformer` to manage text data (`TfidfVectorizer`) and categorical data (`OneHotEncoder`) in a single unified object.
* **Automated Model Selection**: Includes a benchmarking suite for Logistic Regression, Linear SVC, Random Forest, and Naive Bayes with Hyperparameter tuning via `GridSearchCV`.
* **Pickle-Ready**: Architecture designed for easy deployment via `joblib`.

## 📂 Project Structure
```text
├── data/
│   └── University_Query.csv    # Dataset
├── models/
│   ├── ModelPipeline.pkl  # Trained Pipeline object
│   └── Label_Map.pkl                # Numerical to Label mapping
├── notebooks/
│   └── Pipelining.ipynb           # Data analysis & Model training
|       TextPreprocessing.ipynb
├── src/
│   └── transformers.py              # Custom Preprocess & Flattener classes
├── app.py                           # Streamlit Web Application
├── requirements.txt                 # Dependencies
└── README.md
```

## 🛠️ How to Run

### 1. Clone the Repository
Open your terminal or command prompt and run:
```bash
git clone [https://github.com/your-username/university-query-priority.git](https://github.com/your-username/university-query-priority.git)
cd university-query-priority
```
### 2. Set Up a Virtual Environment
It is highly recommended to use a virtual environment to avoid dependency conflicts:
##### Create the environment
```bash
python -m venv venv
```
##### Activate the environment (Windows)
```bash
venv\Scripts\activate
```
##### Activate the environment (Mac/Linux)
```bash
source venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries and download the necessary NLTK data:
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### 4. Run the Web Application
This project uses Streamlit for the frontend. To launch the web interface, run:
```bash
streamlit run app.py
```

### 5. Training the Model (Optional)
If you wish to retrain the model or explore the data analysis, launch the Jupyter Notebook:
```bash
jupyter notebook notebooks/Training_EDA.ipynb
```
