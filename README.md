# Sentiment Analysis

- Naive Bayes
- SMOTE
- Cross Validaition
- Count Vectorizer
- Sastrawi
- Seaborn

## Installation

Export model trained

```python
# add this code in main.ipynb after defining pipeline
pipeline.fit(X, y)
joblib.dump(pipeline, 'sentiment_analysis_model.pkl')
```

Run flask web app (note: make sure to export  the model in advance) :

```bash
python flask_webapp.py
```
