# Medical-Abbreviation-Expander
A Smart NLP Tool to Decode and Analyze Medical Abbreviations

# Overview
The Medical Abbreviation Expander is an AI-powered Streamlit application that automatically detects and expands medical abbreviations in clinical text.
It supports both rule-based lookup and contextual embedding-based disambiguation for accurate expansion.

Additionally, a Jupyter Notebook is included to evaluate model Precision, Accuracy, and visualize results.

# Features
* Expand common medical abbreviations
* Contextual disambiguation using embeddings
* View annotated clinical notes with expanded text
* Auto-save all analysis results (analysis_data.csv)
* Precision & Accuracy evaluation using ground truth data
* Beautiful data visualization with charts and graphs
* Fully open-source and customizable
  
# Project Structure
```bash
Medical-Abbreviation-Expander/
│
├── app/
│   ├── main.py               # Streamlit main app
│   └── utils.py              # Helper functions (find, highlight)
│
├── models/
│   ├── resolver.py           # Rule-based abbreviation resolver
│   └── embed_disambiguator.py # Embedding model for contextual expansion
│
├── data/
│   ├── abbreviations.csv      # Base abbreviation dictionary
│   ├── analysis_data.csv      # Generated analysis data
│   ├── ground_truth.csv       # Ground truth for evaluation
│   └── evaluation_results.csv # Computed results (after running notebook)
│
├── evaluation.ipynb          # Jupyter notebook for metrics & graphs
├── requirements.txt          # Dependencies
├── .env                      # Environment variables (optional)
└── README.md                 # Project documentation
```

# Installation Guide

# Clone this repository
```bash
git clone https://github.com/<your-username>/Medical-Abbreviation-Expander.git
cd Medical-Abbreviation-Expander
```
# Create a virtual environment
```bash
python -m venv .venv
```
# Activate the environment
```bash
.venv\Scripts\activate
```

# Install dependencies
```bash
pip install -r requirements.txt
```

# Running the Streamlit App
```bash
streamlit run app/main.py
```
Then open your browser at:
```arduino
http://localhost:8501
```

# Running Evaluation Notebook
After you’ve used the app (so that analysis_data.csv is generated):
```bash
jupyter notebook
```

Then open evaluation.ipynb and run all cells to see:

* Precision & Accuracy values
* Correct vs Incorrect bar chart
* Confidence trend over time
* Confusion matrix of predictions

# Author
Developed by: Tarun
GitHub: @Tarunyl
Email: taruntelus123@gmail.com
