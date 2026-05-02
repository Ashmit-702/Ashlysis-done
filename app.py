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

# ── DAILY QUOTA ──
_quota = {'date': str(date.today()), 'count': 0}
DAILY_LIMIT = 100

def check_quota():
    today = str(date.today())
    if _quota['date'] != today:
        _quota['date'] = today
        _quota['count'] = 0
    if _quota['count'] >= DAILY_LIMIT:
        raise Exception('Daily limit reached. Try again tomorrow.')
    _quota['count'] += 1

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

# ── ENHANCED TESSERACT OCR WITH IMAGE PRE-PROCESSING ──
def enhance_image_for_ocr(img):
    """Enhance image quality before OCR — removes watermarks, improves contrast"""
    from PIL import Image, ImageFilter, ImageEnhance, ImageOps
    import numpy as np

    # Convert to grayscale
    if img.mode != 'L':
        img = img.convert('L')

    # Convert to numpy for processing
    img_array = np.array(img)

    # Step 1: Remove light watermarks using threshold
    # Watermarks are usually light gray — make them white
    img_array[img_array > 200] = 255

    # Step 2: Increase contrast of dark text
    img_array[img_array < 100] = 0

    # Step 3: Convert back to PIL
    img = Image.fromarray(img_array)

    # Step 4: Sharpen
    img = img.filter(ImageFilter.SHARPEN)

    # Step 5: Enhance contrast further
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)

    return img

def extract_text_via_ocr(pdf_path):
    """Enhanced OCR with image pre-processing for scanned PDFs"""
    try:
        import fitz
        import pytesseract
        from PIL import Image as PILImage
        import io as _io

        doc = fitz.open(pdf_path)
        all_text = []

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Render at 300 DPI for good accuracy
            mat = fitz.Matrix(300/72, 300/72)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")

            # Load as PIL image
            img = PILImage.open(_io.BytesIO(img_bytes))

            # Apply enhancement
            img = enhance_image_for_ocr(img)

            # OCR with best settings for printed text
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()-+*/=[]{}\'\" \n'
            text = pytesseract.image_to_string(img, config=custom_config, lang='eng')

            if text.strip():
                all_text.append(text)

        doc.close()
        return "\n".join(all_text).strip()
    except ImportError:
        return ""
    except Exception as e:
        return ""

# ── SMART QUESTION EXTRACTION ──
def extract_questions_only(text):
    lines = text.split('\n')
    question_lines = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        is_q = bool(re.search(
            r'^\s*Q\.?\s*\d+|^\s*\d{1,2}\s*[.)]\s*[A-Za-z(]|'
            r'\bexplain\b|\bdescribe\b|\bdefine\b|\bdiscuss\b|\bconstruct\b|'
            r'\bdesign\b|\bcompare\b|\bdifferentiate\b|\bdraw\b|\bwrite\b|'
            r'\bstate\b|\blist\b|\bwhat\b|\bhow\b|\bwhy\b|\bderive\b|'
            r'\bcalculate\b|\bimplement\b|short\s*note|advantages|disadvantages|'
            r'\belaborate\b|\billustrate\b|\bjustify\b|\bevaluate\b|\banalyze\b|'
            r'\bgenerate\b|\bcompute\b|\bsolve\b|\bfind\b|\bprove\b|\bshow\b|'
            r'flowchart|difference between|types of|working of|phases of|'
            r'with example|with suitable|with neat|with diagram',
            line, re.I
        ))

        if is_q and len(line) > 12:
            combined = line
            if i + 1 < len(lines):
                next_line = lines[i+1].strip()
                if next_line and len(next_line) > 10 and not re.match(
                    r'^\s*Q\.?\s*\d+|\d{1,2}\s*[.)]', next_line):
                    combined += ' ' + next_line
                    i += 1
            question_lines.append(combined[:350])

        i += 1

    return '\n'.join(question_lines)

# ── PAPER NAME PARSER ──
MONTH_MAP = {
    'jan':('January',1),'feb':('February',2),'mar':('March',3),
    'apr':('April',4),'may':('May',5),'jun':('June',6),
    'jul':('July',7),'aug':('August',8),'sep':('September',9),
    'oct':('October',10),'nov':('November',11),'dec':('December',12)
}

def parse_paper_name(filename):
    parts = filename.lower().replace('-','_').split('_')
    year = next((int(p) for p in parts if re.match(r'20\d\d$', p)), None)
    month_key = next((p[:3] for p in parts if p[:3] in MONTH_MAP), None)
    month_name, month_num = MONTH_MAP.get(month_key, ('',0))
    sort_key = year * 100 + month_num if year else 0
    if year and month_name:
        display = f"{month_name}_{year}"
    elif year:
        display = f"Paper_{year}"
    else:
        display = f"Paper_{filename[:10]}"
    return display, sort_key

# ── GROQ WITH RETRY ──
def call_groq_with_retry(client, prompt, retries=2):
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an expert exam question analyzer for Mumbai University engineering papers. Always respond with valid JSON only. No markdown, no explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=4000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e).lower()
            if 'rate' in err or '429' in err:
                if attempt < retries:
                    time.sleep(30)
                    continue
                raise Exception('Rate limit hit. Wait 1 minute and try again.')
            if 'timeout' in err:
                if attempt < retries:
                    time.sleep(5)
                    continue
                raise Exception('Analysis timed out. Try fewer papers.')
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
            minimal = f'{{"clusters":[{clusters_text}],"predictions":[],"study_plan":{{"strategy":"Focus on HIGH priority topics.","days":[],"golden_topics":[],"dont_skip":[]}},"paper_pattern":{{}}}}'
            return json.loads(minimal)
    except Exception:
        pass
    raise json.JSONDecodeError("Could not parse", text, 0)

# ── MAP STEP: Extract questions from single paper ──
def map_extract_questions(client, paper_name, questions_text):
    """Step 1 of map-reduce: ask Groq to extract clean questions from one paper"""
    prompt = f"""Extract all exam questions from this paper. Return a JSON array of objects.

Paper: {paper_name}

Content:
{questions_text[:4000]}

Return ONLY a JSON array:
[
  {{
    "position": "Q1a",
    "question": "exact question text",
    "marks": 5,
    "topic_hint": "one word topic"
  }}
]

Rules:
- Include ALL questions Q1 through Q6
- position format: Q1a, Q1b, Q2A, Q2B, Q3, Q4a etc
- marks: integer, 0 if not visible
- Extract the actual question text exactly as written
- Return ONLY the JSON array"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Extract exam questions. Return valid JSON array only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'^```(?:json)?\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw).strip()
        questions = json.loads(raw)
        # Add paper name to each question
        for q in questions:
            q['paper'] = paper_name
        return questions
    except Exception:
        return []

# ── REDUCE STEP: Find patterns across all papers ──
def reduce_find_patterns(client, all_questions, user_name, num_papers):
    """Step 2 of map-reduce: find repeating patterns across all papers"""

    # Format questions by paper
    questions_by_paper = {}
    for q in all_questions:
        paper = q.get('paper', 'Unknown')
        if paper not in questions_by_paper:
            questions_by_paper[paper] = []
        questions_by_paper[paper].append(q)

    # Build compact representation
    papers_summary = ""
    for paper, questions in questions_by_paper.items():
        papers_summary += f"\n--- {paper} ---\n"
        for q in questions:
            papers_summary += f"  {q.get('position','?')} [{q.get('marks',0)}m]: {q.get('question','')[:150]}\n"

    prompt = f"""You are analyzing {num_papers} Mumbai University exam papers for student: {user_name}

Here are ALL extracted questions from ALL papers:
{papers_summary}

Find questions that repeat across papers. Same topic = same cluster even if wording differs.

SPECIFICALLY look for these known repeat topics:
- Two-pass assembler / Pass 1 flowchart
- Forward reference problem
- Direct Linking Loader / Dynamic Linking Loader
- Compiler phases
- Code optimization techniques (common subexpression, dead code, code motion, constant propagation)
- Macro processor (single-pass, two-pass, macro calls within macros)
- Intermediate code / Three address code / Basic blocks
- Parser (SLR, LL(1), operator precedence, predictive)
- Assembler statements / directives
- System software vs application software

Return ONLY this JSON:
{{
  "clusters": [
    {{
      "topic": "Two-pass Assembler Pass 1",
      "frequency": 4,
      "importance": "HIGH",
      "questions": ["Draw flowchart of pass1 of assembler...", "Explain Pass-I of two pass assembler...", "Draw and explain flowchart of Pass-I..."],
      "papers": ["Nov_2023", "May_2023", "Dec_2024"],
      "question_positions": ["Q2a", "Q6B", "Q2A"],
      "marks_each_time": [10, 10, 10],
      "consistent_position": false,
      "consistent_marks": true,
      "pattern_note": "Always 10 marks, appears in Q2 or Q6",
      "tip": "Draw the flowchart with all boxes clearly labeled — SYMTAB, LOCCTR, OPTAB",
      "keywords": ["assembler", "pass1", "flowchart", "symtab"]
    }}
  ],
  "predictions": [
    {{
      "question": "Draw the flowchart of Pass-I of two-pass assembler and explain its working with the databases used.",
      "topic": "Two-pass Assembler Pass 1",
      "confidence": "HIGH",
      "reason": "Appeared in all 4 papers, always 10 marks",
      "likely_position": "Q2",
      "likely_marks": 10,
      "frequency": 4
    }}
  ],
  "paper_pattern": {{
    "compulsory_question": "Q1 always compulsory — 4 sub-questions (a,b,c,d) of 5 marks each = 20 marks",
    "optional_questions": "Q2 to Q6 — attempt any 3, each has 2 parts (A and B) of 10 marks each = 20 marks",
    "total_marks": 80,
    "duration": "3 hours",
    "key_insight": "Q1 always tests 4 different short topics. For Q2-Q6, knowing 6 topics well enough to write 10-mark answers = guaranteed pass"
  }},
  "study_plan": {{
    "strategy": "personalized 2-3 sentence strategy for {user_name}",
    "days": [
      {{"day": 1, "focus": "topic name", "priority": "HIGH", "hours": 3, "tasks": ["task1", "task2", "task3"]}},
      {{"day": 2, "focus": "topic name", "priority": "HIGH", "hours": 3, "tasks": ["task1", "task2", "task3"]}},
      {{"day": 3, "focus": "topic name", "priority": "HIGH", "hours": 3, "tasks": ["task1", "task2", "task3"]}},
      {{"day": 4, "focus": "topic name", "priority": "MEDIUM", "hours": 2, "tasks": ["task1", "task2", "task3"]}},
      {{"day": 5, "focus": "topic name", "priority": "MEDIUM", "hours": 2, "tasks": ["task1", "task2", "task3"]}},
      {{"day": 6, "focus": "topic name", "priority": "LOW", "hours": 2, "tasks": ["task1", "task2", "task3"]}},
      {{"day": 7, "focus": "Full Revision + Mock", "priority": "HIGH", "hours": 4, "tasks": ["Revise all HIGH topics", "Attempt timed mock test", "Review weak areas", "Prepare cheat sheet"]}}
    ],
    "golden_topics": ["topic1", "topic2", "topic3"],
    "dont_skip": ["topic1", "topic2"]
  }}
}}

RULES:
- clusters: 8-15 sorted by frequency desc
- HIGH = 3+ papers, MEDIUM = 2, LOW = 1 but important
- Each cluster = ONE distinct topic only — never mix unrelated questions
- Do NOT group assembler + compiler + loader into one cluster
- predictions: exactly 10
- days: exactly 7
- Return ONLY JSON"""

    raw = call_groq_with_retry(client, prompt)
    return safe_parse_json(raw)

# ── MAIN ANALYSIS — MAP-REDUCE ──
def analyze_with_groq(all_papers_text, user_name):
    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

    sorted_papers = sorted(all_papers_text.items(), key=lambda x: x[1][1], reverse=True)

    # ── MAP: Extract questions from each paper ──
    all_questions = []
    for paper_name, (text, _sort) in sorted_papers:
        if not text or len(text.strip()) < 50:
            continue
        questions_only = extract_questions_only(text)
        if len(questions_only.strip()) < 30:
            questions_only = text[:4000]
        else:
            questions_only = questions_only[:6000]

        # Extract structured questions from this paper
        paper_questions = map_extract_questions(client, paper_name, questions_only)
        all_questions.extend(paper_questions)

        # Small delay to avoid rate limiting
        time.sleep(1)

    if not all_questions:
        raise Exception("Could not extract questions from papers")

    # ── REDUCE: Find patterns across all papers ──
    result = reduce_find_patterns(client, all_questions, user_name, len(sorted_papers))
    return result


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
                text = extract_text_from_pdf(filepath)

                if not text or len(text.strip()) < 100:
                    try:
                        ocr_text = extract_text_via_ocr(filepath)
                        if ocr_text and len(ocr_text.strip()) > 50:
                            text = ocr_text
                            ocr_used.append(paper_name)
                    except Exception:
                        pass

                if not text or len(text.strip()) < 50:
                    return jsonify({
                        'error': f'Could not read "{filename}". Convert at smallpdf.com using OCR option and re-upload.'
                    }), 400

                all_papers_text[paper_name] = (text, sort_key)
                with pdfplumber.open(filepath) as pdf:
                    paper_stats[paper_name] = {
                        'questions': max(text.count('Q.'), text.count('?'), 5),
                        'pages': len(pdf.pages),
                        'chars': len(text),
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
        for q in c.get('questions', [])[:4]:
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
