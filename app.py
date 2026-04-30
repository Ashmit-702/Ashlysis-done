import os
import json
import re
import io
import time
from datetime import datetime, date
import pdfplumber
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from groq import Groq

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = '/tmp/ashlysis_uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ── DAILY QUOTA COUNTER ──
_quota = {'date': str(date.today()), 'count': 0}
DAILY_LIMIT = 100

def check_quota():
    today = str(date.today())
    if _quota['date'] != today:
        _quota['date'] = today
        _quota['count'] = 0
    if _quota['count'] >= DAILY_LIMIT:
        raise Exception(f'Daily analysis limit reached. Please try again tomorrow.')
    _quota['count'] += 1

# ── PDF TEXT EXTRACTION (pdfplumber) ──
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text and len(page_text.strip()) > 30:
                    text += page_text + "\n"
                    continue
                words = page.extract_words()
                if words:
                    line = " ".join(w['text'] for w in words)
                    if len(line.strip()) > 30:
                        text += line + "\n"
    except Exception as e:
        raise Exception(f"Could not read PDF: {str(e)}")
    return text.strip()

# ── TESSERACT OCR FOR SCANNED PDFs ──
def extract_text_via_ocr(pdf_path):
    try:
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image
        import tempfile

        doc = fitz.open(pdf_path)
        all_text = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            # Render at 300 DPI for good OCR accuracy
            mat = fitz.Matrix(300/72, 300/72)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
            img_bytes = pix.tobytes("png")

            # PIL image
            from PIL import Image as PILImage
            import io as _io
            img = PILImage.open(_io.BytesIO(img_bytes))

            # Tesseract OCR
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(img, config=custom_config)
            if text.strip():
                all_text.append(text)

        doc.close()
        return "\n".join(all_text).strip()
    except ImportError as e:
        raise Exception(f"OCR not available: {str(e)}")
    except Exception as e:
        raise Exception(f"OCR failed: {str(e)}")

# ── SMART QUESTION LINE EXTRACTION ──
def extract_questions_only(text):
    lines = text.split('\n')
    question_lines = []
    capture_next = False
    for line in lines:
        line = line.strip()
        if not line or len(line) < 10:
            capture_next = False
            continue
        is_question = bool(re.search(
            r'Q\.?\s*\d|question\s*\d|\b\d{1,2}\s*[.)]\s*[A-Z(]|'
            r'explain|describe|define|discuss|construct|design|compare|'
            r'differentiate|draw|write|state|list|what|how|why|derive|'
            r'calculate|implement|short\s*note|advantages|disadvantages|'
            r'elaborate|illustrate|justify|evaluate|analyze|generate|'
            r'compute|solve|find|prove|show|demonstrate',
            line, re.I
        ))
        if is_question:
            question_lines.append(line[:300])
            capture_next = True
        elif capture_next and len(line) > 20:
            if question_lines:
                question_lines[-1] = question_lines[-1] + ' ' + line[:100]
            capture_next = False
    return '\n'.join(question_lines[:80])

# ── PARSE PAPER NAME + YEAR ──
MONTH_MAP = {
    'jan': ('January', 1), 'feb': ('February', 2), 'mar': ('March', 3),
    'apr': ('April', 4), 'may': ('May', 5), 'jun': ('June', 6),
    'jul': ('July', 7), 'aug': ('August', 8), 'sep': ('September', 9),
    'oct': ('October', 10), 'nov': ('November', 11), 'dec': ('December', 12)
}

def parse_paper_name(filename):
    parts = filename.lower().replace('-', '_').split('_')
    year = next((int(p) for p in parts if re.match(r'20\d\d$', p)), None)
    month_key = next((p[:3] for p in parts if p[:3] in MONTH_MAP), None)
    month_name, month_num = MONTH_MAP.get(month_key, ('', 0))
    sort_key = year * 100 + month_num if year else 0
    if year and month_name:
        display = f"{month_name}_{year}"
    elif year:
        display = f"Paper_{year}"
    else:
        display = f"Paper_{filename[:10]}"
    return display, sort_key

# ── GROQ CALL WITH RETRY ──
def call_groq_with_retry(client, prompt, retries=2):
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an expert exam question analyzer. Always respond with valid JSON only. Never add markdown backticks or explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e).lower()
            if 'rate' in err or '429' in err:
                if attempt < retries:
                    time.sleep(30)
                    continue
                raise Exception('Rate limit reached. Please wait 1 minute and try again.')
            if 'timeout' in err or 'timed out' in err:
                if attempt < retries:
                    time.sleep(5)
                    continue
                raise Exception('Analysis timed out. Try uploading fewer papers.')
            raise e

# ── SAFE JSON PARSE ──
def safe_parse_json(text):
    text = re.sub(r'^```(?:json)?\n?', '', text.strip())
    text = re.sub(r'\n?```$', '', text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        match = re.search(r'"clusters"\s*:\s*\[', text)
        if match:
            start = match.end()
            depth = 1
            i = start
            last_complete = start
            while i < len(text) and depth > 0:
                if text[i] == '{': depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 1: last_complete = i + 1
                i += 1
            clusters_text = text[start:last_complete].rstrip(',')
            minimal = f'{{"clusters": [{clusters_text}], "predictions": [], "study_plan": {{"strategy": "Focus on HIGH priority topics.", "days": [], "golden_topics": [], "dont_skip": []}}, "paper_pattern": {{}}}}'
            return json.loads(minimal)
    except Exception:
        pass
    raise json.JSONDecodeError("Could not parse response", text, 0)

# ── MAIN ANALYSIS ──
def analyze_with_groq(all_papers_text, user_name):
    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

    sorted_papers = sorted(all_papers_text.items(), key=lambda x: x[1][1], reverse=True)

    papers_content = ""
    for paper_name, (text, _sort) in sorted_papers:
        if not text or len(text.strip()) < 50:
            continue
        questions_only = extract_questions_only(text)
        if len(questions_only.strip()) < 30:
            questions_only = text[:2500]
        else:
            questions_only = questions_only[:5000]
        papers_content += f"\n\n=== PAPER: {paper_name} ===\n{questions_only}"

    if not papers_content.strip():
        raise Exception("Could not read text from any of the papers")

    prompt = f"""You are an expert exam analyzer for engineering students in India (Mumbai University).
Student: {user_name} | Papers: {len(all_papers_text)}

Papers (sorted latest first):
{papers_content}

Analyze like a smart student:
1. Latest paper questions are the BASE
2. Find same/similar questions across papers
3. Note EXACT question number (Q1a, Q2B etc) and MARKS each time
4. Find patterns — always Q1? Always 10 marks?

Return ONLY this JSON:
{{
  "clusters": [
    {{
      "topic": "topic name max 6 words",
      "frequency": 3,
      "importance": "HIGH",
      "questions": ["question from paper1", "question from paper2"],
      "papers": ["May_2025", "Dec_2024"],
      "question_positions": ["Q1a", "Q2B"],
      "marks_each_time": [5, 10],
      "consistent_position": true,
      "consistent_marks": false,
      "pattern_note": "Always in Q1, marks vary 5-10",
      "tip": "practical exam tip",
      "keywords": ["word1", "word2"]
    }}
  ],
  "predictions": [
    {{
      "question": "predicted question text",
      "topic": "topic name",
      "confidence": "HIGH",
      "reason": "why likely",
      "likely_position": "Q1",
      "likely_marks": 10,
      "frequency": 3
    }}
  ],
  "paper_pattern": {{
    "compulsory_question": "Q1 always compulsory, 4 parts of 5 marks",
    "optional_questions": "Q2-Q6, attempt any 3, 20 marks each",
    "total_marks": 80,
    "duration": "3 hours",
    "key_insight": "key pattern observation"
  }},
  "study_plan": {{
    "strategy": "2-3 sentence strategy for {user_name}",
    "days": [
      {{"day": 1, "focus": "topic", "priority": "HIGH", "hours": 3, "tasks": ["task1", "task2", "task3"]}}
    ],
    "golden_topics": ["topic1", "topic2", "topic3"],
    "dont_skip": ["topic1", "topic2"]
  }}
}}

Rules: clusters 6-15 sorted by frequency, HIGH=3+papers, MEDIUM=2, LOW=1, predictions 8-10, exactly 7 days"""

    raw = call_groq_with_retry(client, prompt)
    return safe_parse_json(raw)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded.'}), 400

    files = request.files.getlist('files')
    if len(files) < 2:
        return jsonify({'error': 'Please upload at least 2 PYQ papers.'}), 400
    if len(files) > 10:
        return jsonify({'error': 'Maximum 10 papers allowed.'}), 400

    non_pdf = [f.filename for f in files if not f.filename.lower().endswith('.pdf')]
    if non_pdf:
        return jsonify({'error': f'Only PDF files supported. Remove: {", ".join(non_pdf)}'}), 400

    user_name = request.form.get('user_name', 'Student').strip() or 'Student'

    if not os.environ.get('GROQ_API_KEY'):
        return jsonify({'error': 'Service not configured. Contact admin.'}), 500

    try:
        check_quota()
    except Exception as e:
        return jsonify({'error': str(e)}), 429

    all_papers_text = {}
    paper_stats = {}
    ocr_used = []

    for file in files:
        if file and file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)
            paper_name, sort_key = parse_paper_name(filename)

            base_name = paper_name
            counter = 1
            while paper_name in all_papers_text:
                paper_name = f"{base_name}_{counter}"
                counter += 1

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Try normal text extraction first
                text = extract_text_from_pdf(filepath)

                # If too short → try OCR automatically
                if not text or len(text.strip()) < 100:
                    try:
                        text = extract_text_via_ocr(filepath)
                        if text and len(text.strip()) > 50:
                            ocr_used.append(paper_name)
                    except Exception:
                        pass

                if not text or len(text.strip()) < 50:
                    return jsonify({
                        'error': f'Could not read "{filename}". Please convert it at smallpdf.com using the OCR option and re-upload.'
                    }), 400

                all_papers_text[paper_name] = (text, sort_key)
                with pdfplumber.open(filepath) as pdf:
                    paper_stats[paper_name] = {
                        'questions': max(text.count('Q.'), text.count('?'), 5),
                        'pages': len(pdf.pages),
                        'ocr': paper_name in ocr_used
                    }
            except Exception as e:
                return jsonify({'error': f'Failed to read "{filename}": {str(e)}'}), 500
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)

    if len(all_papers_text) < 2:
        return jsonify({'error': 'Need at least 2 readable papers.'}), 400

    try:
        result = analyze_with_groq(all_papers_text, user_name)
        clusters = result.get('clusters', [])
        predictions = result.get('predictions', [])
        study_plan = result.get('study_plan', {})
        paper_pattern = result.get('paper_pattern', {})
    except json.JSONDecodeError:
        return jsonify({'error': 'Analysis returned unexpected format. Please try again.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({
        'clusters': clusters,
        'predictions': predictions,
        'study_plan': study_plan,
        'paper_pattern': paper_pattern,
        'user_name': user_name,
        'ocr_used': ocr_used,
        'stats': {
            'papers': len(all_papers_text),
            'total_questions': sum(s.get('questions', 0) for s in paper_stats.values()),
            'clusters': len(clusters),
            'high_priority': len([c for c in clusters if c.get('importance') == 'HIGH']),
            'paper_stats': paper_stats
        }
    })


@app.route('/export', methods=['POST'])
def export_results():
    data = request.json
    clusters = data.get('clusters', [])
    predictions = data.get('predictions', [])
    study_plan = data.get('study_plan', {})
    paper_pattern = data.get('paper_pattern', {})
    stats = data.get('stats', {})
    user_name = data.get('user_name', 'Student')
    today = datetime.now().strftime('%Y%m%d')
    export_filename = f'ashlysis_{user_name.replace(" ","_")}_{today}.txt'

    lines = [
        "╔══════════════════════════════════════════════════╗",
        "║           ASHLYSIS — EXAM INTELLIGENCE           ║",
        "╚══════════════════════════════════════════════════╝",
        "",
        f"Student         : {user_name}",
        f"Generated       : {datetime.now().strftime('%d %b %Y %I:%M %p')}",
        f"Papers analyzed : {stats.get('papers', 0)}",
        f"Repeat clusters : {stats.get('clusters', 0)}",
        f"HIGH priority   : {stats.get('high_priority', 0)}",
    ]

    if paper_pattern:
        lines += ["", "━"*52, "PAPER PATTERN", "━"*52,
            f"Compulsory : {paper_pattern.get('compulsory_question','')}",
            f"Optional   : {paper_pattern.get('optional_questions','')}",
            f"Marks      : {paper_pattern.get('total_marks',80)} | Duration: {paper_pattern.get('duration','3 hours')}",
            f"Insight    : {paper_pattern.get('key_insight','')}"]

    lines += ["", "━"*52, "REPEATING QUESTIONS (ranked)", "━"*52]
    for i, c in enumerate(clusters, 1):
        positions = c.get('question_positions', [])
        marks = c.get('marks_each_time', [])
        lines += [
            f"\n{i}. [{c.get('importance')}] {c.get('topic')} — {c.get('frequency')}x",
            f"   Papers   : {', '.join(c.get('papers', []))}",
            f"   Position : {', '.join(str(p) for p in positions)}{'  ← ALWAYS SAME ✓' if c.get('consistent_position') else ''}",
            f"   Marks    : {', '.join(str(m) for m in marks)}{'  ← ALWAYS SAME ✓' if c.get('consistent_marks') else ''}",
            f"   Pattern  : {c.get('pattern_note','')}",
            f"   Tip      : {c.get('tip','')}",
        ]
        for q in c.get('questions', [])[:3]:
            lines.append(f"   • {q[:200]}")

    lines += ["", "━"*52, "PREDICTED QUESTIONS", "━"*52]
    for i, p in enumerate(predictions, 1):
        lines += [
            f"\n{i}. [{p.get('confidence')}] {p.get('question','')[:200]}",
            f"   Position : likely {p.get('likely_position','?')} | Marks: ~{p.get('likely_marks','?')}",
            f"   Why      : {p.get('reason','')}"
        ]

    lines += ["", "━"*52, "7-DAY STUDY PLAN", "━"*52]
    if study_plan:
        lines.append(f"\nStrategy: {study_plan.get('strategy','')}")
        lines.append(f"Golden  : {', '.join(study_plan.get('golden_topics',[]))}")
        lines.append(f"Never skip: {', '.join(study_plan.get('dont_skip',[]))}")
        for day in study_plan.get('days', []):
            lines += [f"\nDay {day['day']} [{day['priority']}] — {day['focus']} ({day['hours']}hrs)"]
            for t in day.get('tasks', []):
                lines.append(f"  ✓ {t}")

    lines += ["", "━"*52, f"Generated by ASHLYSIS for {user_name}", "━"*52]
    buf = io.BytesIO("\n".join(lines).encode('utf-8'))
    buf.seek(0)
    return send_file(buf, mimetype='text/plain', as_attachment=True, download_name=export_filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
