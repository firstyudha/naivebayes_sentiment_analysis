import os
import io
import base64
import joblib
import re
from collections import Counter
from flask import Flask, render_template_string, request, redirect, url_for, send_file
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

def clean_text(text):
    """Basic text cleaning"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.strip()  # remove leading/trailing whitespace
    text = ''.join([c for c in text if c.isalpha() or c.isspace()])
    return text

def stopword_removal(text):
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory 
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover() 
    return stopword.remove(text) 

def stem_text(text):
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory 
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return stemmer.stem(text)

def normalize_text(text):

    normalization_dict = {
    'gw': 'saya', 
    'loe': 'kamu',
    'yg': 'yang',
    'gk': 'tidak',
    'ga': 'tidak',
    'gak': 'tidak',
    'aja': 'saja',
    'btw': 'by the way',
    'klo': 'kalau',
    'nih': 'ini',
    'tdk': 'tidak',
    'dgn': 'dengan',
    'sdh': 'sudah'
    } 

    words = text.split()
    normalized_words = [normalization_dict.get(word, word) for word in
    words]
    return ' '.join(normalized_words)   

def data_preprocessing(text):
    text = clean_text(text)
    text = stopword_removal(text)
    text = stem_text(text)
    text = normalize_text(text)
    return text

def generate_wordcloud(text, sentiment):
    """Generate word cloud image"""
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {sentiment} Sentiment')
    
    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_sentiment_segmentation(data, sentiment_counts,sentiment_percentages):
    """Generate sentiment segmentation"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Pie Chart
    axes[0].pie(
        sentiment_percentages, 
        labels=['positive', 'negative'], 
        autopct='%1.1f%%', 
        startangle=90, 
        colors=['#4CAF50', '#FF5722'],
        explode=(0.1, 0)
    )
    axes[0].set_title('Sentiment Percentages', fontsize=16)

    # Bar Plot
    sns.countplot(data=data, x='sentiment', palette='viridis', ax=axes[1])
    axes[1].set_title('Sentiment Distribution', fontsize=14)
    axes[1].set_xlabel('Sentiment Label', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)

    # Add counts on top of each bar
    for p in axes[1].patches:
        axes[1].annotate(
            f'{int(p.get_height())}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center',
            va='baseline',
            fontsize=12,
            color='black',
            xytext=(0, 5),
            textcoords='offset points'
        )
    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Sentiment Analysis</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet" />
  <style>
    /* Animated geometry background */
    body, html {
      height: 100%;
      margin: 0;
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      background: #f8fafc; /* light background */
      overflow-x: hidden;
      color: #1e293b; /* dark text */
      position: relative;
    }

    .animated-bg {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      z-index: 0;
      background: radial-gradient(circle at center, #cbd5e1 0%, #f8fafc 70%);
      overflow: hidden;
    }

    .animated-bg span {
      position: absolute;
      display: block;
      border-radius: 50%;
      background: rgba(59, 130, 246, 0.3); /* softer blue */
      animation: float 20s linear infinite;
      opacity: 0.5;
    }

    .animated-bg span:nth-child(1) {
      width: 120px;
      height: 120px;
      left: 10%;
      top: 20%;
      animation-duration: 25s;
      animation-delay: 0s;
    }

    .animated-bg span:nth-child(2) {
      width: 90px;
      height: 90px;
      left: 80%;
      top: 10%;
      animation-duration: 20s;
      animation-delay: 5s;
    }

    .animated-bg span:nth-child(3) {
      width: 150px;
      height: 150px;
      left: 40%;
      top: 70%;
      animation-duration: 30s;
      animation-delay: 10s;
    }

    @keyframes float {
      0% {
        transform: translateY(0) translateX(0);
        opacity: 0.5;
      }
      50% {
        opacity: 0.15;
      }
      100% {
        transform: translateY(-100px) translateX(100px);
        opacity: 0.5;
      }
    }

    /* Container styling */
    .container-main {
      position: relative;
      z-index: 10;
      max-width: 480px;
      margin: 100px auto;
      background: rgba(255, 255, 255, 0.85);
      padding: 30px;
      border-radius: 12px;
      box-shadow: 0 0 25px rgba(59, 130, 246, 0.4);
      color: #334155;
    }

    h1 {
      font-weight: 600;
      margin-bottom: 1.5rem;
      color: #2563eb;
      text-align: center;
    }

    label.form-label {
      font-weight: 600;
      color: #475569;
    }

    input.form-control {
      background: #f1f5f9;
      border: 1px solid #cbd5e1;
      color: #334155;
    }

    input.form-control:focus {
      border-color: #2563eb;
      box-shadow: 0 0 8px #2563eb;
      background: #ffffff;
      color: #334155;
    }

    .form-text {
      color: #64748b;
    }

    button.btn-primary {
      background-color: #2563eb;
      border: none;
      font-weight: 600;
      width: 100%;
      transition: background-color 0.3s ease;
      color: white;
    }

    button.btn-primary:hover:not(:disabled) {
      background-color: #1e40af;
    }

    button:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }

    /* Full screen loading overlay */
    .loading-screen {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      background: rgba(255, 255, 255, 0.85);
      z-index: 9999;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      pointer-events: all;
      user-select: none;
      color: #2563eb;
      font-weight: 600;
      font-size: 1.25rem;
    }
  </style>
</head>
<body>
  <!-- Animated geometry background container -->
  <div class="animated-bg" aria-hidden="true">
    <span></span>
    <span></span>
    <span></span>
  </div>

  <div class="container-main">
    <h1>Sentiment Analysis with Naive Bayes</h1>
    {% if error %}
    <div class="alert alert-danger" role="alert">{{ error }}</div>
    {% endif %}
    <!-- Loading screen overlay -->
    <div id="loadingScreen" class="loading-screen d-none" aria-hidden="true" role="alert" aria-live="assertive" aria-busy="true">
      <div class="spinner-border text-primary" role="status" aria-hidden="true"></div>
      <div class="mt-3">Analyzing, please wait...</div>
    </div>

    <form id="uploadForm" method="post" enctype="multipart/form-data" action="{{ url_for('analyze') }}">
    <div class="mb-3">
        <label for="file" class="form-label">Upload XLSX file</label>
        <input class="form-control" type="file" id="file" name="file" accept=".xlsx" required />
        <div class="form-text">XLSX must contain 'content' columns</div>
    </div>

    <div class="form-check form-switch mb-3">
        <input class="form-check-input" type="checkbox" role="switch" id="preprocessingToggle" name="preprocessing">
        <label class="form-check-label" for="preprocessingToggle">Data Preprocessing</label>
    </div>

    <button id="submitBtn" type="submit" class="btn btn-primary">Analyze</button>
    </form>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
  <script>
    document.getElementById('uploadForm').addEventListener('submit', function (e) {
      const loadingScreen = document.getElementById('loadingScreen');
      loadingScreen.classList.remove('d-none');
      loadingScreen.setAttribute('aria-hidden', 'false');
      document.getElementById('submitBtn').disabled = true;
    });
  </script>
</body>
</html>

    ''')


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return render_template_string('''
            <div class="alert alert-danger">No file uploaded</div>
            <a href="{{ url_for('index') }}" class="btn btn-primary">Back</a>
            ''', error="No file uploaded")
        
        file = request.files['file']
        if file.filename == '':
            return render_template_string('''
            <div class="alert alert-danger">No file selected</div>
            <a href="{{ url_for('index') }}" class="btn btn-primary">Back</a>
            ''', error="No file selected")
        
        if not file.filename.endswith('.xlsx'):
            return render_template_string('''
            <div class="alert alert-danger">Please upload a XLSX file</div>
            <a href="{{ url_for('index') }}" class="btn btn-primary">Back</a>
            ''', error="Please upload a XLSX file")
        
        # Read and validate XLSX
        df = pd.read_excel(file)
        if 'content' not in df.columns:
            return render_template_string('''
            <div class="alert alert-danger">XLSX must contain 'content' columns</div>
            <a href="{{ url_for('index') }}" class="btn btn-primary">Back</a>
            ''', error="XLSX must contain 'text' columns")
        
        # Preprocess data

        preprocessing_enabled = 'preprocessing' in request.form
        if preprocessing_enabled:
            df['cleaned_text'] = df['content'].apply(data_preprocessing)
            df['cleaned_text'].dropna(inplace=True)
        else:
            df['cleaned_text'] = df['content'].apply(clean_text)

        pipeline = joblib.load('sentiment_analysis_model.pkl')

        # Predict sentiment for the new data
        new_data = df['cleaned_text']
        predictions = pipeline.predict(new_data)
        # Create a DataFrame for visualization
        df_predictions = pd.DataFrame({'text': new_data, 'sentiment': predictions})

        sentiment_counts = df_predictions['sentiment'].value_counts().to_dict()
        sentiment_counts_series = pd.Series(sentiment_counts)
        sentiment_percentages = (sentiment_counts_series / len(df)) * 100

        # Generate word clouds for negative and positive sentiments
        positive_text = ' '.join(df_predictions[df_predictions['sentiment'] == 'positive']['text'])
        negative_text = ' '.join(df_predictions[df_predictions['sentiment'] == 'negative']['text'])

        sentiment_segmentation = generate_sentiment_segmentation(df_predictions,sentiment_counts,sentiment_percentages)

        positive_wc = generate_wordcloud(positive_text, 'Positive')
        negative_wc = generate_wordcloud(negative_text, 'Negative')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        cm_img = base64.b64encode(buf.read()).decode('utf-8')
        
        return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Analysis Results</title>
  <link
    href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
    rel="stylesheet"
  />
  <style>
    /* Base styles */
    body, html {
      margin: 0; padding: 0; height: 100%;
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      background: #f0f4f8;
      overflow-x: hidden;
      color: #222;
      position: relative;
      min-height: 100vh;
    }

    /* Animated background container */
    .animated-background {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      pointer-events: none;
      overflow: hidden;
      z-index: 0;
      background: #e9eef5;
    }

    /* Geometry shape base styles */
    .shape {
      position: absolute;
      opacity: 0.15;
      background: linear-gradient(135deg, #2575fc, #6a11cb);
      filter: drop-shadow(0 0 5px rgba(37, 117, 252, 0.6));
      will-change: transform;
      border-radius: 10px;
    }

    /* Circles */
    .circle {
      border-radius: 50%;
    }

    /* Triangles */
    .triangle {
      width: 0;
      height: 0;
      border-left: 30px solid transparent;
      border-right: 30px solid transparent;
      border-bottom: 50px solid #6a11cb;
      background: none;
      filter: drop-shadow(0 0 3px rgba(106, 17, 203, 0.6));
      border-radius: 0;
    }

    /* Hexagons */
    .hexagon {
      width: 60px;
      height: 34.64px;
      background: linear-gradient(135deg, #2575fc, #6a11cb);
      position: relative;
      margin: 17.32px 0;
    }
    .hexagon:before,
    .hexagon:after {
      content: "";
      position: absolute;
      width: 0;
      border-left: 30px solid transparent;
      border-right: 30px solid transparent;
    }
    .hexagon:before {
      bottom: 100%;
      border-bottom: 17.32px solid #6a11cb;
    }
    .hexagon:after {
      top: 100%;
      border-top: 17.32px solid #6a11cb;
    }

    /* Keyframe animations */
    @keyframes floatUpDown {
      0%, 100% {
        transform: translateY(0) rotate(0deg);
      }
      50% {
        transform: translateY(-20px) rotate(180deg);
      }
    }

    @keyframes floatLeftRight {
      0%, 100% {
        transform: translateX(0) rotate(0deg);
      }
      50% {
        transform: translateX(20px) rotate(180deg);
      }
    }

    @keyframes floatSlowRotate {
      0% {
        transform: rotate(0deg);
      }
      100% {
        transform: rotate(360deg);
      }
    }

    /* Individual shape positioning and animation */
    .shape1 {
      width: 80px;
      height: 80px;
      border-radius: 50%;
      top: 15%;
      left: 10%;
      animation: floatUpDown 10s ease-in-out infinite;
    }

    .shape2 {
      width: 60px;
      height: 60px;
      clip-path: polygon(50% 0%, 0% 100%, 100% 100%);
      background: #6a11cb;
      filter: drop-shadow(0 0 4px rgba(106, 17, 203, 0.5));
      top: 60%;
      left: 15%;
      animation: floatLeftRight 14s ease-in-out infinite;
    }

    .shape3 {
      width: 100px;
      height: 57.74px; /* For hexagon shape */
      top: 40%;
      left: 75%;
      animation: floatSlowRotate 25s linear infinite;
      opacity: 0.1;
    }

    .container-main {
      position: relative;
      z-index: 10;
      background: rgba(255, 255, 255, 0.85);
      padding: 30px 20px;
      border-radius: 15px;
      box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
      max-width: 960px;
      margin: 50px auto 60px;
    }

    /* Wordcloud margin */
    .wordcloud {
      margin: 20px 0;
    }

    /* Buttons */
    .btn-primary {
      background: #2575fc;
      border-color: #2575fc;
      transition: background-color 0.3s ease, border-color 0.3s ease;
    }
    .btn-primary:hover {
      background: #1855d1;
      border-color: #1855d1;
    }
  </style>
</head>
<body>
  <!-- Animated background shapes -->
  <div class="animated-background" aria-hidden="true">
    <div class="shape circle shape1"></div>
    <div class="shape triangle shape2"></div>
    <div class="shape hexagon shape3"></div>
  </div>

  <!-- Main content container -->
  <div class="container-main">
    <h1 class="mb-4 text-center">Analysis Results</h1>

    <div class="card mb-4 shadow-sm">
      <div class="card-header bg-primary text-white">
        <h2 class="mb-0">Sentiment Segmentation</h2>
      </div>
      <div class="card-body text-center">
        <img src="data:image/png;base64,{{ sentiment_segmentation }}" class="img-fluid" alt="Sentiment Segmentation" />
      </div>
    </div>

    <div class="row g-3">
      <div class="col-md-6">
        <div class="card shadow-sm">
          <div class="card-header bg-success text-white">
            <h3 class="mb-0">Positive Words</h3>
          </div>
          <div class="card-body text-center">
            <img src="data:image/png;base64,{{ positive_wc }}" class="img-fluid wordcloud" alt="Positive Words Wordcloud" />
          </div>
        </div>
      </div>
      <div class="col-md-6">
        <div class="card shadow-sm">
          <div class="card-header bg-danger text-white">
            <h3 class="mb-0">Negative Words</h3>
          </div>
          <div class="card-body text-center">
            <img src="data:image/png;base64,{{ negative_wc }}" class="img-fluid wordcloud" alt="Negative Words Wordcloud" />
          </div>
        </div>
      </div>
    </div>

    <div class="mt-4 text-center">
      <a href="{{ url_for('index') }}" class="btn btn-primary">Analyze Another File</a>
    </div>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>


        ''', positive_wc=positive_wc, negative_wc=negative_wc, cm_img=cm_img, sentiment_counts=sentiment_counts, sentiment_segmentation=sentiment_segmentation)
    
    except Exception as e:
        return render_template_string('''
        <div class="alert alert-danger">An error occurred: {{ error }}</div>
        <a href="{{ url_for('index') }}" class="btn btn-primary">Back</a>
        ''', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
    print("App running on http://127.0.0.1:5000")
