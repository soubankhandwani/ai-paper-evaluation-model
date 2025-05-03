# 🧠 PDF Answer Evaluation System (Flask + NLP + Vision AI)

This web application allows automatic evaluation of student answer sheets against teacher-provided answers using **OCR**, **text processing**, and **semantic similarity** scoring powered by **Google Cloud Vision** and **Sentence-BERT**.

## ✨ Features

- 📄 Extract text and images from uploaded PDFs
- 🔍 OCR for scanned student answers (via Google Cloud Vision API)
- 🚫 Detect and skip blank pages
- 🧹 Clean and preprocess answer texts (stopwords, lemmatization, etc.)
- 🧠 Semantic similarity scoring (conceptual & word-based)
- 📊 Question-wise and page-wise evaluation
- 🌐 Web interface with Flask (rendered HTML + JSON API)

---

## 📦 Installation

### Prerequisites
- Python 3.7+
- Google Cloud Vision credentials (`credentials.json`)
- Basic knowledge of Flask & Google Cloud setup

### 1. Clone the repository
```bash
https://github.com/soubankhandwani/ai-paper-evaluation-model.git
cd ai-paper-evaluation-model
