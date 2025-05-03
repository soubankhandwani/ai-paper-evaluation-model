from flask import Flask, request, render_template, session, jsonify
import pdfplumber
from google.cloud import vision
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
import re
from PIL import Image
import io
from nltk.stem import WordNetLemmatizer
import uuid

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Google Cloud Vision setup
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './credentials.json'
client = vision.ImageAnnotatorClient()

# Sentence Transformer model
model = SentenceTransformer('all-mpnet-base-v2')

# NLTK initialization
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# File paths configuration
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"PDF text extraction error: {e}")
    return text.strip()

# def extract_images_from_pdf(pdf_path, session_id):
#     images = []
#     try:
#         with pdfplumber.open(pdf_path) as pdf:
#             for i, page in enumerate(pdf.pages):
#                 for img in page.images:
#                     img_data = img['stream'].get_data()
#                     image = Image.open(io.BytesIO(img_data))
#                     img_path = os.path.join(UPLOAD_FOLDER, f'{session_id}_page_{i + 1}.png')
#                     image.save(img_path, format='PNG')
#                     images.append(img_path)
#     except Exception as e:
#         print(f"Image extraction error: {e}")
#     return images

def extract_images_from_pdf(pdf_path, session_id):
    images = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                for img in page.images:
                    img_data = img['stream'].get_data()
                    image = Image.open(io.BytesIO(img_data))
                    img_path = os.path.join(UPLOAD_FOLDER, f'{session_id}_page_{i + 1}.png')
                    
                    # Save image using a separate context manager
                    with open(img_path, 'wb') as f:
                        image.save(f, format='PNG')
                        f.flush()  # Flush before syncing
                        os.fsync(f.fileno())  # Sync before closing
                    
                    images.append(img_path)
    except Exception as e:
        print(f"Image extraction error: {e}")
    return images

def detect_blank_pages(pdf_path):
    blank_pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                has_text = False
                # Check for text directly
                if page.extract_text():
                    has_text = True
                else:
                    # Check images using OCR
                    for img in page.images:
                        img_data = img['stream'].get_data()
                        image = vision.Image(content=img_data)
                        response = client.text_detection(image=image)
                        if response.text_annotations:
                            has_text = True
                            break
                blank_pages.append(not has_text)
    except Exception as e:
        print(f"Blank page detection error: {e}")
    return blank_pages

def clean_scanned_text(text):
    try:
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\bPhoto\s*synthesis\b', 'Photosynthesis', text, flags=re.IGNORECASE)
        text = re.sub(r'\bSun\s*light\b', 'Sunlight', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.capitalize() for s in sentences if s]
        return ' '.join(sentences)
    except Exception as e:
        print(f"Text cleaning error: {e}")
        return text

def preprocess_text(text):
    try:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return " ".join(words)
    except Exception as e:
        print(f"Text preprocessing error: {e}")
        return text

def word_to_word_similarity(text1, text2):
    if not text1 or not text2:
        return 0
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return (len(intersection) / len(union)) * 100 if union else 0

def conceptual_similarity(text1, text2):
    if not text1 or not text2:
        return 0
    try:
        embeddings = model.encode([text1, text2], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item() * 100
        return max(0, similarity)
    except Exception as e:
        print(f"Concept similarity error: {e}")
        return 0

def parse_questions(text, is_teacher=False):
    questions = {}
    try:
        pattern = re.compile(r'(?i)(Q\s*\d+\s*[\):]?)\s*(.*?)(?=\s*Q\s*\d+\s*[\):]?|\Z)', re.DOTALL)
        matches = pattern.findall(text)
        for match in matches:
            q_label = re.search(r'\d+', match[0].lower()).group()
            answer = match[1].strip()
            if is_teacher:
                answer = re.sub(r'\n+', ' ', answer)
            questions[f'q{q_label}'] = answer
    except Exception as e:
        print(f"Question parsing error: {e}")
    return questions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id

        # Process teacher's PDF
        teacher_pdf = request.files['teacher_pdf']
        teacher_pdf_path = f"teacher_{session_id}.pdf"
        teacher_pdf.save(teacher_pdf_path)
        teacher_text = extract_text_from_pdf(teacher_pdf_path)
        teacher_questions = parse_questions(teacher_text, is_teacher=True)
        session['teacher_questions'] = teacher_questions

        # Process student's PDF
        student_pdf = request.files['student_pdf']
        student_pdf_path = f"student_{session_id}.pdf"
        student_pdf.save(student_pdf_path)
        
        # Extract images and detect blank pages
        student_images = extract_images_from_pdf(student_pdf_path, session_id)
        blank_pages = detect_blank_pages(student_pdf_path)
        
        # Create pages data structure
        student_pages = [{
            'image': img,
            'is_blank': blank_pages[i],
            'page_number': i+1
        } for i, img in enumerate(student_images)]
        
        session['student_pages'] = student_pages

        return render_template('result.html',
                            teacher_questions=teacher_questions,
                            student_pages=student_pages,
                            session_id=session_id)

    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({"error": "File processing failed"}), 500

@app.route('/evaluate/all')
def evaluate_all():
    try:
        session_id = session.get('session_id')
        teacher_questions = session.get('teacher_questions', {})
        student_pdf_path = f"student_{session_id}.pdf"
        
        with pdfplumber.open(student_pdf_path) as pdf:
            student_text = ""
            for page in pdf.pages:
                student_text += page.extract_text() + "\n"
        
        student_text = clean_scanned_text(student_text)
        student_questions = parse_questions(student_text)

        results = []
        total_marks = 0.0
        
        for q_label in teacher_questions:
            teacher_answer = teacher_questions.get(q_label, '')
            student_answer = student_questions.get(q_label, '')

            t_processed = preprocess_text(teacher_answer)
            s_processed = preprocess_text(student_answer)

            word_sim = word_to_word_similarity(t_processed, s_processed)
            concept_sim = conceptual_similarity(teacher_answer, student_answer)
            marks = (concept_sim / 100) * 5

            results.append({
                'question': q_label.upper(),
                'word_score': round(word_sim, 2),
                'concept_score': round(concept_sim, 2),
                'marks': round(marks, 2)
            })
            total_marks += marks

        return jsonify({
            'results': results,
            'total_marks': round(total_marks, 2)
        })

    except Exception as e:
        print(f"Evaluation error: {e}")
        return jsonify({"error": "Evaluation failed"}), 500

@app.route('/evaluate/page/<int:page_number>')
def evaluate_page(page_number):
    try:
        session_id = session.get('session_id')
        teacher_questions = session.get('teacher_questions', {})
        student_pages = session.get('student_pages', [])
        
        if page_number < 1 or page_number > len(student_pages):
            return jsonify({'error': 'Invalid page number'}), 400

        # OCR processing
        page_data = student_pages[page_number - 1]
        if page_data['is_blank']:
            return jsonify({'error': 'Cannot evaluate blank page'}), 400

        with open(page_data['image'], 'rb') as f:
            content = f.read()
        
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        student_text = response.text_annotations[0].description if response.text_annotations else ""
        student_text = clean_scanned_text(student_text)
        student_questions = parse_questions(student_text)

        # Question matching
        q_label = f'q{page_number}'
        teacher_answer = teacher_questions.get(q_label, '')
        student_answer = student_questions.get(q_label, '')

        # Similarity calculations
        t_processed = preprocess_text(teacher_answer)
        s_processed = preprocess_text(student_answer)
        
        word_sim = word_to_word_similarity(t_processed, s_processed)
        concept_sim = conceptual_similarity(teacher_answer, student_answer)
        marks = (concept_sim / 100) * 5

        result = {
            'question': q_label.upper(),
            'word_score': round(word_sim, 2),
            'concept_score': round(concept_sim, 2),
            'marks': round(marks, 2)
        }

        return jsonify({
            'result': result,
            'total_increment': round(marks, 2)
        })

    except Exception as e:
        print(f"Page evaluation error: {e}")
        return jsonify({"error": "Page evaluation failed"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)