# Fake News Detection – Machine Learning Project

A beginner-friendly educational web application that uses Python, NLP, TF-IDF and Logistic Regression to classify a news text as **REAL-like** or **FAKE-like** based on patterns learned from the supplied sample dataset.

## Important limitation
The included CSV is a **small illustrative teaching dataset**, not a professionally verified news corpus. The model can make mistakes, learn wording shortcuts, and should never be used as proof that a story is true or false. Its confidence score is a model score, not a verified probability. For serious evaluation, replace the sample CSV with a large, balanced, properly sourced and labelled dataset, then test on unseen data.

## Requirements
- Python 3.10 or newer
- VS Code (recommended)

## Run on Windows
1. Extract the ZIP.
2. Open the extracted `FakeNewsDetection_ML` folder in VS Code.
3. Open Terminal → New Terminal.
4. Run:
   ```powershell
   .\run_project.bat
   ```
   The script creates a virtual environment, installs libraries, trains the model and starts Flask.
5. Open `http://127.0.0.1:5000` in your browser.
6. Keep the terminal open while using the website. Press `Ctrl + C` to stop it.

### Manual commands
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
python app.py
```
If PowerShell blocks activation, use:
```powershell
.\.venv\Scripts\activate.bat
```
or run the provided `run_project.bat`.

## Project structure
```text
FakeNewsDetection_ML/
├── app.py
├── train_model.py
├── requirements.txt
├── run_project.bat
├── README.md
├── ML_CONCEPTS_AND_VIVA.md
├── data/
│   └── news_dataset.csv
├── model/
│   └── (trained model and metrics are created here)
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── script.js
```

## How it works
1. The user enters a news passage in the browser.
2. JavaScript sends the text to Flask using a POST request.
3. The saved scikit-learn Pipeline converts text into TF-IDF features.
4. Logistic Regression predicts the class.
5. Flask returns the predicted class and model score as JSON.
6. The webpage displays the result and a fact-checking caution.

## Use your own dataset
Replace `data/news_dataset.csv` with a CSV containing exactly these columns:
- `text`: the news headline/article text
- `label`: `REAL` or `FAKE`

Keep both classes represented with enough examples. Remove duplicates, check labels, avoid leakage, and use a separate trustworthy test set. Then run `python train_model.py` again.

## Main files
- `train_model.py`: data cleaning, train/test split, TF-IDF, Logistic Regression, evaluation, model saving.
- `app.py`: Flask routes and prediction API.
- `model/`: generated model and metrics after training.
