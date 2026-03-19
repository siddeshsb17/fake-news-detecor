# Fake News Detector Using Machine Learning

A machine learning project to detect fake and real news headlines using Python.

## About
This project classifies news headlines as REAL or FAKE using:
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Logistic Regression

## Accuracy
- Test Accuracy: 90%
- F1-Score: 0.95

## Technologies Used
- Python 3
- Scikit-learn
- Pandas
- NLTK
- Porter Stemmer

## How to Run
1. Install requirements:
pip install pandas scikit-learn nltk

2. Run the program:
python fakenews1.py

3. Enter any news headline to check if it is REAL or FAKE

## Dataset
Custom dataset of 200 headlines:
- 100 Real News Headlines
- 100 Fake News Headlines

## Results
| Headline | Result | Confidence |
|---|---|---|
| NASA launched satellite to monitor climate | REAL | 60.7% |
| Earth is flat NASA lying | FAKE | 60.4% |

## Output Screenshot
![Output](Output%20for%20FAKE%20NEWS%20DETECTOR%20code.png)

## Author
siddeshsb
