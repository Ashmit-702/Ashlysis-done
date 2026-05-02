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

# ── CLEAN TEXT — strip watermarks and fix merged words ──
def clean_pdf_text(text):
    """Remove watermark noise and fix common PDF extraction issues"""
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip lines that are pure watermark/hash garbage
        if re.match(r'^[A-F0-9]{20,}$', line):  # hex hashes
            continue
        if re.match(r'^[*_\-=]{5,}$', line):  # decoration lines
            continue
        if len(line) < 3:
            continue
        # Fix merged words like "Explainthe" -> "Explain the"
        line = re.sub(r'([a-z])([A-Z])', r'\1 \2', line)
        # Fix merged numbers like "Q1.Explain" -> "Q1. Explain"
        line = re.sub(r'(Q\d+)\.?([A-Z])', r'\1. \2', line)
        # Remove repeated chars like "......." or "------"
        line = re.sub(r'[.\-_]{4,}', ' ', line)
        clean_lines.append(line)
    return '\n'.join(clean_lines)

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
    return clean_pdf_text(text.strip())

# ── GROQ VISION FOR SCANNED PDFs ──
def extract_text_via_vision(pdf_path, groq_client):
    """Use Groq vision model to read scanned image PDFs"""
    try:
        import fitz
        import base64

        doc = fitz.open(pdf_path)
        all_text = []

        for page_num in range(min(len(doc), 3)):
            page = doc[page_num]
            # 150 DPI — small enough for API, clear enough for reading
            mat = fitz.Matrix(150/72, 150/72)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("jpeg")
            img_b64 = base64.b64encode(img_bytes).decode()

            try:
                response = groq_client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                            },
                            {
                                "type": "text",
                                "text": "This is an engineering exam paper. Extract every question exactly as written. Format: Q1a [5 marks]: question text. Include ALL questions from Q1 to Q6. Plain text only, no markdown."
                            }
                        ]
                    }],
                    max_tokens=1500,
                    temperature=0.1
                )
                page_text = response.choices[0].message.content
                if page_text and page_text.strip():
                    all_text.append(page_text.strip())
            except Exception:
                continue

        doc.close()
        return clean_pdf_text("\n".join(all_text))

    except Exception:
        return ""

# ── IS PDF SCANNED? ──
def is_scanned_pdf(text):
    """Detect if extracted text is from a scanned/image PDF"""
    if not text or len(text.strip()) < 150:
        return True
    alpha_ratio = len([c for c in text if c.isalpha()]) / max(len(text), 1)
    if alpha_ratio < 0.3:
        return True
    # Check if it has question markers
    has_questions = bool(re.search(r'Q\.?\s*\d|explain|describe|define|discuss', text, re.I))
    if not has_questions:
        return True
    return False

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

# ── GROQ CALL WITH RETRY ──
def call_groq(client, messages, max_tokens=2000, temperature=0.1, retries=2):
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=temperature,  # LOW = deterministic
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e).lower()
            if 'rate' in err or '429' in err:
                if attempt < retries:
                    time.sleep(30)
                    continue
                raise Exception('Rate limit. Wait 1 minute and try again.')
            if 'timeout' in err:
                if attempt < retries:
                    time.sleep(5)
                    continue
                raise Exception('Timed out. Try fewer papers.')
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

# ── MAP: Extract structured questions from one paper ──
def map_extract_questions(client, paper_name, questions_text):
    prompt = f"""Extract all exam questions from this Mumbai University paper.

Paper: {paper_name}
Content:
{questions_text[:4000]}

Return ONLY a JSON array:
[
  {{"position": "Q1a", "question": "exact question text", "marks": 5, "topic": "one word topic"}}
]

Include ALL questions Q1a through Q6B. Position format: Q1a, Q1b, Q2A, Q2B etc.
marks = integer (0 if not visible). Return ONLY the JSON array."""

    try:
        raw = call_groq(client, [
            {"role": "system", "content": "Extract exam questions. Return valid JSON array only. No markdown."},
            {"role": "user", "content": prompt}
        ], max_tokens=2000, temperature=0.1)

        raw = re.sub(r'^```(?:json)?\n?', '', raw.strip())
        raw = re.sub(r'\n?```$', '', raw).strip()
        questions = json.loads(raw)
        for q in questions:
            q['paper'] = paper_name
        return questions
    except Exception:
        return []

# ── REDUCE: Find patterns across papers ──
def reduce_find_patterns(client, all_questions, user_name, num_papers):
    questions_by_paper = {}
    for q in all_questions:
        paper = q.get('paper', 'Unknown')
        if paper not in questions_by_paper:
            questions_by_paper[paper] = []
        questions_by_paper[paper].append(q)

    papers_summary = ""
    for paper, questions in sorted(questions_by_paper.items()):
        papers_summary += f"\n--- {paper} ---\n"
        for q in questions:
            papers_summary += f"  {q.get('position','?')} [{q.get('marks',0)}m]: {q.get('question','')[:150]}\n"

    prompt = f"""Analyze {num_papers} Mumbai University engineering exam papers for student: {user_name}

ALL QUESTIONS FROM ALL PAPERS:
{papers_summary}

Find questions repeating across papers. Same concept = same cluster even if wording differs slightly.

MANDATORY — check every paper for these specific topics:
1. Two-pass assembler / Pass-1 flowchart / Pass-2
2. Forward reference problem
3. Direct Linking Loader / Absolute loader / Dynamic loader
4. Phases of compiler
5. Code optimization (dead code, common subexpression, code motion, constant propagation)
6. Macro processor (single-pass, two-pass, macro calls within macros)
7. Intermediate code / Three address code / Basic blocks / Flow graph
8. Parser (SLR, LL1, operator precedence, predictive parser)
9. System software vs application software
10. Assembler directives / statements

Return ONLY this JSON:
{{
  "clusters": [
    {{
      "topic": "Two-pass Assembler Pass 1",
      "frequency": 4,
      "importance": "HIGH",
      "questions": ["Draw flowchart of pass1...", "Explain Pass-I...", "Draw and explain..."],
      "papers": ["Nov_2023", "May_2023", "Dec_2024", "May_2025"],
      "question_positions": ["Q2a", "Q6B", "Q2A", "Q2A"],
      "marks_each_time": [10, 10, 10, 10],
      "consistent_position": false,
      "consistent_marks": true,
      "pattern_note": "Always 10 marks, appears in Q2 or Q6",
      "tip": "Draw full flowchart with SYMTAB, LOCCTR, OPTAB boxes clearly labeled",
      "keywords": ["assembler", "pass1", "flowchart"]
    }}
  ],
  "predictions": [
    {{
      "question": "Draw the flowchart of Pass-I of two-pass assembler and explain its working with databases.",
      "topic": "Two-pass Assembler Pass 1",
      "confidence": "HIGH",
      "reason": "Appeared in all 4 papers always 10 marks",
      "likely_position": "Q2",
      "likely_marks": 10,
      "frequency": 4
    }}
  ],
  "paper_pattern": {{
    "compulsory_question": "Q1 always compulsory 4 sub-questions a,b,c,d of 5 marks each = 20 marks",
    "optional_questions": "Q2-Q6 attempt any 3 each has 2 parts A and B of 10 marks = 20 marks per question",
    "total_marks": 80,
    "duration": "3 hours",
    "key_insight": "Q1 tests 4 short topics. Knowing 6 topics for 10-mark answers = guaranteed pass"
  }},
  "study_plan": {{
    "strategy": "2-3 sentence personalized strategy for {user_name}",
    "days": [
      {{"day": 1, "focus": "topic", "priority": "HIGH", "hours": 3, "tasks": ["task1", "task2", "task3"]}},
      {{"day": 2, "focus": "topic", "priority": "HIGH", "hours": 3, "tasks": ["task1", "task2", "task3"]}},
      {{"day": 3, "focus": "topic", "priority": "HIGH", "hours": 3, "tasks": ["task1", "task2", "task3"]}},
      {{"day": 4, "focus": "topic", "priority": "MEDIUM", "hours": 2, "tasks": ["task1", "task2", "task3"]}},
      {{"day": 5, "focus": "topic", "priority": "MEDIUM", "hours": 2, "tasks": ["task1", "task2", "task3"]}},
      {{"day": 6, "focus": "topic", "priority": "LOW", "hours": 2, "tasks": ["task1", "task2"]}},
      {{"day": 7, "focus": "Full Revision + Mock Test", "priority": "HIGH", "hours": 4, "tasks": ["Revise all HIGH topics", "Attempt timed mock", "Review weak areas", "Prepare quick reference sheet"]}}
    ],
    "golden_topics": ["topic1", "topic2", "topic3"],
    "dont_skip": ["topic1", "topic2"]
  }}
}}

STRICT RULES:
- clusters: 8-15 sorted by frequency desc
- HIGH = 3+ papers, MEDIUM = 2, LOW = 1 important topic
- ONE topic per cluster — never mix assembler + compiler + loader
- consistent_marks: true ONLY if ALL marks values are identical
- predictions: exactly 10
- days: exactly 7
- Return ONLY valid JSON — no markdown, no explanation"""

    raw = call_groq(client, [
        {"role": "system", "content": "You are an expert exam analyzer. Return valid JSON only. No markdown."},
        {"role": "user", "content": prompt}
    ], max_tokens=4000, temperature=0.1)

    return safe_parse_json(raw)

# ── MAIN ANALYSIS ──
def analyze_with_groq(all_papers_text, user_name):
    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
    sorted_papers = sorted(all_papers_text.items(), key=lambda x: x[1][1], reverse=True)

    all_questions = []
    for paper_name, (text, _sort) in sorted_papers:
        if not text or len(text.strip()) < 50:
            continue
        questions_only = extract_questions_only(text)
        if len(questions_only.strip()) < 30:
            questions_only = text[:4000]
        else:
            questions_only = questions_only[:6000]

        paper_questions = map_extract_questions(client, paper_name, questions_only)
        all_questions.extend(paper_questions)
        time.sleep(1)

    if not all_questions:
        raise Exception("Could not extract questions from papers")

    return reduce_find_patterns(client, all_questions, user_name, len(sorted_papers))


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

    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
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

                # If scanned, use Groq vision
                if is_scanned_pdf(text):
                    vision_text = extract_text_via_vision(filepath, client)
                    if vision_text and len(vision_text.strip()) > 50:
                        text = vision_text
                        ocr_used.append(paper_name)
                    elif not text or len(text.strip()) < 50:
                        return jsonify({
                            'error': f'Could not read "{filename}". It appears to be a scanned image PDF. Please convert it at smallpdf.com → select "OCR PDF" → download → re-upload.'
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
        return jsonify({'error': 'Analysis failed. Please try again.'}), 500
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
