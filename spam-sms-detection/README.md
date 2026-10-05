# Spam SMS Detection

Classifies SMS messages as **SPAM** or **NOT SPAM** using TF-IDF with Naive Bayes, Logistic Regression and Linear SVM. The best model is served through a Flask web app.

## Dataset (`data/spam.csv`)
- `v1` = label (`ham` / `spam`), `v2` = SMS text (5,572 rows; 4,825 ham, 747 spam)
- The three `Unnamed` columns are overflow fragments and are ignored
- 414 duplicate rows are removed before the train/test split (5,158 remain)
- File encoding is `latin-1`

## Results (20% stratified test set, spam = positive class)

| Model               | Accuracy | Precision | Recall | F1     |
|---------------------|----------|-----------|--------|--------|
| Naive Bayes         | 0.9855   | 0.9748    | 0.9062 | 0.9393 |
| Logistic Regression | 0.9835   | 0.9237    | 0.9453 | 0.9344 |
| Linear SVM          | 0.9864   | 0.9597    | 0.9297 | 0.9444 |

Confusion matrices `[[ham->ham, ham->spam], [spam->ham, spam->spam]]`:
- Naive Bayes: `[[901, 3], [12, 116]]`
- Logistic Regression: `[[894, 10], [7, 121]]`
- Linear SVM: `[[899, 5], [9, 119]]`

**Best model (highest F1): Linear SVM.** It is wrapped in `CalibratedClassifierCV` so it can output confidence scores.

## Setup and run
```bash
pip install -r requirements.txt
python training/train_model.py   # trains, compares, saves model/spam_model.pkl
python app.py                    # open http://127.0.0.1:5000
```

## API
`POST /predict` with JSON `{"message": "your sms text"}` returns:
```json
{"prediction": "SPAM", "confidence": 99.88}
```

## Project structure
```
spam-sms-detection/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/spam.csv
├── model/spam_model.pkl
├── training/train_model.py
├── templates/index.html
└── static/
    ├── css/style.css
    └── js/script.js
```
