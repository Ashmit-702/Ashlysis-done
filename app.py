import os
import json
import re
import io
import base64
import pdfplumber
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import google.generativeai as genai

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = '/tmp/ashlysis_uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ── CONFIGURE GEMINI ──
def get_gemini():
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise Exception('Service not configured. Contact admin.')
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

# ── PDF TEXT EXTRACTION ──
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

# ── READ SCANNED PDF VIA GEMINI VISION ──
def extract_text_via_gemini_vision(pdf_path, model):
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        all_text = []
        for page_num in range(min(len(doc), 4)):
            page = doc[page_num]
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            img_b64 = base64.standard_b64encode(img_bytes).decode()

            import PIL.Image
            import io as _io
            img = PIL.Image.open(_io.BytesIO(img_bytes))

            response = model.generate_content([
                img,
                "This is an exam question paper page. Extract ALL question text exactly as written. List each question on a new line with its number. Ignore instructions like 'attempt any 3' or marks allocation. Just output the questions."
            ])
            all_text.append(response.text)
        doc.close()
        return "\n".join(all_text)
    except Exception:
        return ""

# ── MAIN ANALYSIS WITH GEMINI ──
def analyze_with_gemini(all_papers_text, model):
    papers_content = ""
    for paper_name, text in all_papers_text.items():
        if not text or len(text.strip()) < 50:
            continue
        truncated = text[:3000] if len(text) > 3000 else text
        papers_content += f"\n\n=== PAPER: {paper_name} ===\n{truncated}"

    if not papers_content.strip():
        raise Exception("Could not read text from any of the papers")

    prompt = f"""You are an expert exam analyzer for engineering students in India (Mumbai University / similar universities).

Here are {len(all_papers_text)} previous year exam papers:
{papers_content}

Analyze these papers carefully and find:
1. Questions/topics that REPEAT across multiple papers (same concept, even if wording differs slightly)
2. Rank them by frequency
3. Predict likely questions for the NEXT exam
4. Create a 7-day study plan

Respond ONLY with a valid JSON object in this exact format:
{{
  "clusters": [
    {{
      "topic": "topic name (max 6 words)",
      "frequency": 3,
      "importance": "HIGH",
      "questions": ["exact question from paper 1", "same topic question from paper 2"],
      "papers": ["Paper_2023", "Paper_2024"],
      "tip": "one practical exam tip for this topic",
      "keywords": ["keyword1", "keyword2", "keyword3"]
    }}
  ],
  "predictions": [
    {{
      "question": "full predicted question text",
      "topic": "topic name",
      "confidence": "HIGH",
      "reason": "one line why this is likely",
      "frequency": 3
    }}
  ],
  "study_plan": {{
    "strategy": "2-3 sentence overall exam strategy",
    "days": [
      {{
        "day": 1,
        "focus": "topic name",
        "priority": "HIGH",
        "hours": 3,
        "tasks": ["specific task 1", "specific task 2", "specific task 3"]
      }}
    ],
    "golden_topics": ["most important topic 1", "topic 2", "topic 3"],
    "dont_skip": ["absolutely critical topic 1", "critical topic 2"]
  }}
}}

Important rules:
- clusters: 6-15 items sorted by frequency descending
- HIGH = appeared 3+ papers or very core topic
- MEDIUM = appeared in 2 papers  
- LOW = appeared once but important
- predictions: 8-10 items
- study_plan.days: exactly 7 days
- Return ONLY the JSON object, no markdown, no explanation"""

    response = model.generate_content(prompt)
    response_text = response.text.strip()
    response_text = re.sub(r'^```(?:json)?\n?', '', response_text)
    response_text = re.sub(r'\n?```$', '', response_text)
    return json.loads(response_text)


# ── ROUTES ──
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files = request.files.getlist('files')
    if len(files) < 2:
        return jsonify({'error': 'Please upload at least 2 PYQ papers'}), 400
    if len(files) > 10:
        return jsonify({'error': 'Maximum 10 papers allowed'}), 400

    if not os.environ.get('GEMINI_API_KEY'):
        return jsonify({'error': 'Service not configured. Contact admin.'}), 500

    try:
        model = get_gemini()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    all_papers_text = {}
    paper_stats = {}

    for file in files:
        if file and file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)

            # Clean up paper name - extract year
            paper_name = filename.rsplit('.', 1)[0]
            parts = paper_name.split('_')
            year = next((p for p in parts if re.match(r'20\d\d', p)), None)
            month = next((p for p in parts if p.lower() in ['may','nov','dec','jun','jan','feb','mar','apr']), None)
            subject = next((p for p in reversed(parts) if len(p) > 4 and not re.match(r'20\d\d|be|computer|engineering|semester|scheme|rev|cbcgs|cbsgs', p, re.I)), None)

            if year and month:
                paper_name = f"{month.title()}_{year}"
            elif year:
                paper_name = f"Paper_{year}"
            elif subject:
                paper_name = subject[:15]
            else:
                paper_name = f"Paper_{len(all_papers_text)+1}"

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Try text extraction first
                text = extract_text_from_pdf(filepath)

                # If too short, try vision
                if len(text.strip()) < 100:
                    text = extract_text_via_gemini_vision(filepath, model)

                if not text or len(text.strip()) < 50:
                    return jsonify({
                        'error': f'Could not extract text from {filename}. Please convert it using OCR at smallpdf.com and try again.'
                    }), 400

                all_papers_text[paper_name] = text
                with pdfplumber.open(filepath) as pdf:
                    paper_stats[paper_name] = {
                        'questions': max(text.count('Q.'), text.count('?'), 5),
                        'pages': len(pdf.pages)
                    }
            except Exception as e:
                return jsonify({'error': f'Failed to read {filename}: {str(e)}'}), 500
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)

    if len(all_papers_text) < 2:
        return jsonify({'error': 'Need at least 2 readable papers to find patterns'}), 400

    try:
        result = analyze_with_gemini(all_papers_text, model)
        clusters = result.get('clusters', [])
        predictions = result.get('predictions', [])
        study_plan = result.get('study_plan', {})
    except json.JSONDecodeError:
        return jsonify({'error': 'Analysis failed. Please try again.'}), 500
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

    return jsonify({
        'clusters': clusters,
        'predictions': predictions,
        'study_plan': study_plan,
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
    stats = data.get('stats', {})

    lines = [
        "╔══════════════════════════════════════════════════╗",
        "║           ASHLYSIS — EXAM INTELLIGENCE           ║",
        "╚══════════════════════════════════════════════════╝",
        "",
        f"Papers analyzed : {stats.get('papers', 0)}",
        f"Repeat clusters : {stats.get('clusters', 0)}",
        f"HIGH priority   : {stats.get('high_priority', 0)}",
        "", "━"*52,
        "SECTION 1 — REPEATING QUESTIONS (ranked)",
        "━"*52,
    ]
    for i, c in enumerate(clusters, 1):
        lines += [
            f"\n{i}. [{c.get('importance')}] {c.get('topic')} — {c.get('frequency')}x",
            f"   Papers: {', '.join(c.get('papers', []))}",
            f"   Tip: {c.get('tip', '')}",
            "   Questions:"
        ]
        for q in c.get('questions', [])[:3]:
            lines.append(f"   • {q[:200]}")

    lines += ["", "━"*52, "SECTION 2 — PREDICTED QUESTIONS", "━"*52]
    for i, p in enumerate(predictions, 1):
        lines += [
            f"\n{i}. [{p.get('confidence')}] {p.get('question','')[:200]}",
            f"   Topic: {p.get('topic')} | Why: {p.get('reason','')}"
        ]

    lines += ["", "━"*52, "SECTION 3 — 7-DAY STUDY PLAN", "━"*52]
    if study_plan:
        lines.append(f"\nStrategy: {study_plan.get('strategy','')}")
        lines.append(f"Golden Topics: {', '.join(study_plan.get('golden_topics',[]))}")
        lines.append(f"Never Skip: {', '.join(study_plan.get('dont_skip',[]))}")
        for day in study_plan.get('days', []):
            lines += [f"\nDay {day['day']} [{day['priority']}] — {day['focus']} ({day['hours']}hrs)"]
            for t in day.get('tasks', []):
                lines.append(f"  ✓ {t}")

    lines += ["", "━"*52, "Generated by ASHLYSIS — Powered by Google Gemini", "━"*52]
    buf = io.BytesIO("\n".join(lines).encode('utf-8'))
    buf.seek(0)
    return send_file(buf, mimetype='text/plain', as_attachment=True, download_name='ashlysis_report.txt')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
