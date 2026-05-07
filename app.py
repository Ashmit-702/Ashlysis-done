import os
import json
import re
import io
import time
from datetime import datetime, date
import gc
import threading
import pdfplumber
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from groq import Groq

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = '/tmp/ashlysis_uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ── CONCURRENT REQUEST GUARD ──
_analysis_lock = threading.Semaphore(1)

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

UNIVERSITY_DB = {
    "mumbai_be": {
        "name": "Mumbai University — BE",
        "structure": "Q1 compulsory (4 parts × 5 marks = 20 marks). Q2-Q6 attempt any 3 (2 parts × 10 marks = 20 marks each). Total: 80 marks, 3 hours.",
        "q1_marks": 5, "optional_marks": 10, "total_marks": 80, "duration": "3 hours",
        "pattern": "Q1a/b/c/d = 5 marks each. Q2A/B through Q6A/B = 10 marks each."
    }
}

SUBJECT_TOPICS = {
    "spcc": ["Two-pass Assembler Pass 1 flowchart","Forward reference problem","Direct Linking Loader / Absolute Loader / Dynamic Loader","Phases of Compiler","Code Optimization techniques","Macro Processor single-pass / two-pass","Intermediate Code / Three Address Code / Basic Blocks","Parser SLR / LL(1) / Operator Precedence / Predictive","System Software vs Application Software","Assembler directives and statements"],
    "dbms": ["Normalization 1NF 2NF 3NF BCNF","SQL queries joins","ER diagram","Transaction ACID properties","Concurrency control","Relational algebra","Indexing B-tree","Recovery techniques"],
    "os": ["Process scheduling algorithms","Deadlock detection prevention","Memory management paging","Semaphore mutex","Virtual memory","File system","Disk scheduling","Process synchronization"],
    "cn": ["OSI model layers","TCP IP protocol","Routing algorithms","Congestion control","Error detection correction","Medium access control","Socket programming","Network security"],
    "mc": ["GSM Architecture","Mobile IP Agent Discovery","Handover mechanism","Frequency Reuse","IEEE 802.11 WLAN","GPRS architecture","UMTS 3G","LTE 4G","Bluetooth","Mobile TCP Snooping","Hidden Exposed station problem","WAP architecture"],
    "dsa": ["Sorting algorithms time complexity","Tree traversal BST","Graph BFS DFS","Dynamic programming","Hashing","Stack queue linked list","Heap priority queue","Divide and conquer"],
    "general": ["Check every Q1-Q6 question in all papers for repeating topics"]
}

def clean_pdf_text(text):
    lines = text.split('\n')
    clean = []
    for line in lines:
        line = line.strip()
        if not line or len(line) < 3: continue
        # Skip solid hex watermarks e.g. "DB6A5D47F4D6..."
        if re.match(r'^[A-F0-9]{20,}$', line): continue
        # Skip spaced hex watermarks e.g. "DB 6A 5D 47 ..." or "D B 6 A 5 D..."
        if re.match(r'^([A-F0-9]{1,2}\s){6,}', line): continue
        # Skip lines that are >80% hex characters (watermark noise)
        hex_chars = len(re.findall(r'[A-F0-9]', line))
        if len(line) > 20 and hex_chars / max(len(line.replace(' ','')), 1) > 0.75: continue
        if re.match(r'^[*_\-=.]{5,}$', line): continue
        if len(line) < 12: continue  # skip very short noise lines
        line = re.sub(r'([a-z])([A-Z])', r'\1 \2', line)
        line = re.sub(r'(Q\.?\s*\d+)\.?([A-Za-z])', r'\1. \2', line)
        line = re.sub(r'[.\-_]{4,}', ' ', line)
        line = re.sub(r'\s+', ' ', line)
        clean.append(line)
    return '\n'.join(clean)

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

def is_scanned_pdf(text):
    if not text or len(text.strip()) < 150: return True
    # Clean first then check
    cleaned = clean_pdf_text(text)
    if not cleaned or len(cleaned.strip()) < 100: return True
    alpha = len([c for c in cleaned if c.isalpha()]) / max(len(cleaned), 1)
    if alpha < 0.35: return True
    if not re.search(r'Q\.?\s*\d|explain|describe|define|discuss|marks|attempt', cleaned, re.I): return True
    return False

def extract_text_via_vision(pdf_path, groq_client):
    """Stitch all pages into ONE image — single API call per paper"""
    try:
        import fitz, base64
        from PIL import Image as PILImage
        import io as _io

        doc = fitz.open(pdf_path)
        page_images = []
        for page_num in range(min(len(doc), 4)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(120/72, 120/72))
            img = PILImage.open(_io.BytesIO(pix.tobytes("png"))).convert('L')
            page_images.append(img)
        doc.close()

        if not page_images:
            return ""

        total_h = sum(img.height for img in page_images)
        max_w = max(img.width for img in page_images)
        stitched = PILImage.new('L', (max_w, total_h), 255)
        y = 0
        for img in page_images:
            stitched.paste(img, (0, y))
            y += img.height

        buf = _io.BytesIO()
        stitched.save(buf, format='JPEG', quality=65, optimize=True)
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        resp = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": "Engineering exam paper. Extract ALL questions Q1-Q6 with position and marks. Format exactly: Q1a [5m]: question text. One per line. Plain text only."}
            ]}],
            max_tokens=2000, temperature=0.1
        )
        text = resp.choices[0].message.content
        return clean_pdf_text(text) if text else ""
    except Exception:
        return ""

# ── PRE-EXTRACT STRUCTURED JSON (key accuracy fix) ──
def pre_extract_questions_json(paper_name, text, groq_client):
    """Extract questions as strict JSON — prevents one question being in two clusters"""
    prompt = f"""Extract all exam questions from this paper as JSON.

Paper: {paper_name}
Content:
{text[:5500]}

Return ONLY a JSON array. Each item must be unique — one question = one entry:
[
  {{"id": "{paper_name}_Q1a", "position": "Q1a", "marks": 5, "question": "exact question text here"}},
  {{"id": "{paper_name}_Q1b", "position": "Q1b", "marks": 5, "question": "exact question text here"}},
  {{"id": "{paper_name}_Q2A", "position": "Q2A", "marks": 10, "question": "exact question text here"}}
]

Rules:
- Include ALL questions from Q1 through Q6
- Each question gets a unique id like "{paper_name}_Q1a"
- position: Q1a, Q1b, Q1c, Q1d, Q2A, Q2B, Q3A, Q3B etc
- marks: integer (5 for Q1 parts, 10 for Q2-Q6 parts, 0 if unclear)
- question: exact text of the question
- One entry per question — never duplicate
- Return ONLY the JSON array"""

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Extract exam questions as JSON array. No markdown. No explanation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r'^```(?:json)?\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw).strip()
        questions = json.loads(raw)
        # Ensure all have paper field
        for q in questions:
            q['paper'] = paper_name
        return questions
    except Exception:
        return []

def validate_clusters(clusters, all_papers):
    valid = []
    paper_names = set(all_papers.keys())

    for c in clusters:
        valid_papers = [p for p in c.get('papers', []) if any(
            p.lower() in name.lower() or name.lower() in p.lower()
            for name in paper_names
        )]
        if not valid_papers:
            valid_papers = c.get('papers', [])

        freq = min(c.get('frequency', 1), len(all_papers))
        freq = max(freq, len(valid_papers))

        if freq >= 3: importance = 'HIGH'
        elif freq == 2: importance = 'MEDIUM'
        else: importance = c.get('importance', 'LOW')

        marks = [m for m in c.get('marks_each_time', []) if m and m > 0]
        consistent_marks = len(set(marks)) == 1 if len(marks) > 1 else False

        positions = [p for p in c.get('question_positions', []) if p]
        q_nums = [re.match(r'Q\d+', p, re.I).group() if re.match(r'Q\d+', p, re.I) else p for p in positions]
        consistent_position = len(set(q_nums)) == 1 if len(q_nums) > 1 else False

        topic = c.get('topic', '').strip()
        if not topic or len(topic) < 3: continue

        questions = c.get('questions', [])[:max(len(valid_papers), 1)]

        valid.append({
            **c,
            'frequency': freq, 'importance': importance,
            'papers': valid_papers if valid_papers else c.get('papers', []),
            'consistent_marks': consistent_marks,
            'consistent_position': consistent_position,
            'questions': questions,
            'marks_each_time': marks if marks else c.get('marks_each_time', []),
            'question_positions': positions
        })

    valid.sort(key=lambda x: (-x['frequency'], ['LOW','MEDIUM','HIGH'].index(x.get('importance','LOW'))))
    return valid

MONTH_MAP = {
    'jan':('January',1),'feb':('February',2),'mar':('March',3),
    'apr':('April',4),'may':('May',5),'jun':('June',6),
    'jul':('July',7),'aug':('August',8),'sep':('September',9),
    'oct':('October',10),'nov':('November',11),'dec':('December',12)
}

def parse_paper_name(filename):
    parts = filename.lower().replace('-','_').split('_')
    year = next((int(p) for p in parts if re.match(r'20\d\d$', p)), None)
    mk = next((p[:3] for p in parts if p[:3] in MONTH_MAP), None)
    mn, mnum = MONTH_MAP.get(mk, ('',0))
    sk = year*100+mnum if year else 0
    if year and mn: disp = f"{mn}_{year}"
    elif year: disp = f"Paper_{year}"
    else: disp = f"Paper_{filename[:10]}"
    return disp, sk

def safe_parse_json(text):
    text = re.sub(r'^```(?:json)?\n?','',text.strip())
    text = re.sub(r'\n?```$','',text).strip()
    try: return json.loads(text)
    except: pass
    try:
        m = re.search(r'"clusters"\s*:\s*\[', text)
        if m:
            s=m.end(); depth=1; i=s; lc=s
            while i<len(text) and depth>0:
                if text[i]=='{': depth+=1
                elif text[i]=='}':
                    depth-=1
                    if depth==1: lc=i+1
                i+=1
            ct=text[s:lc].rstrip(',')
            return json.loads(f'{{"clusters":[{ct}],"predictions":[],"study_plan":{{"strategy":"Focus on HIGH priority topics.","days":[],"golden_topics":[],"dont_skip":[]}},"paper_pattern":{{}}}}')
    except: pass
    raise json.JSONDecodeError("Cannot parse",text,0)

def analyze_with_groq(all_papers_text, all_structured_questions, user_name, university, subject):
    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
    sorted_papers = sorted(all_papers_text.items(), key=lambda x: x[1][1], reverse=True)

    uni_info = UNIVERSITY_DB.get(university, UNIVERSITY_DB['mumbai_be'])
    topic_list = SUBJECT_TOPICS.get(subject, SUBJECT_TOPICS['general'])

    # Build structured question table — prevents one Q being in two clusters
    # Ensure question_table includes questions from ALL papers
    # Group by paper first, then flatten - guarantees every paper is represented
    by_paper = {}
    for q in all_structured_questions:
        p = q.get('paper','unknown')
        if p not in by_paper:
            by_paper[p] = []
        by_paper[p].append(q)

    # Build balanced representation - each paper gets fair share
    balanced = []
    max_per_paper = max(20, 80 // max(len(by_paper), 1))
    for paper, qs in sorted(by_paper.items()):
        balanced.extend(qs[:max_per_paper])

    question_table = json.dumps(balanced, indent=2)[:12000]

    prompt = f"""You are an expert exam analyzer for {uni_info['name']} students.
Student: {user_name} | Subject: {subject.upper()} | Papers: {len(sorted_papers)}

PAPER STRUCTURE: {uni_info['structure']}

STRUCTURED QUESTIONS FROM ALL PAPERS (each question has a unique ID):
{question_table}

IMPORTANT RULES FOR CLUSTERING:
- Each question ID can appear in ONLY ONE cluster — never duplicate
- Two questions from the SAME paper CANNOT be in the same cluster
- A cluster means: same topic appeared in DIFFERENT papers
- Check every question against all others for topic similarity

KNOWN TOPICS FOR {subject.upper()} (find these specifically):
{chr(10).join(f"- {t}" for t in topic_list)}

Return ONLY valid JSON:
{{
  "clusters": [
    {{
      "topic": "exact topic name",
      "frequency": 3,
      "importance": "HIGH",
      "questions": ["exact Q text from paper1", "exact Q text from paper2", "exact Q text from paper3"],
      "papers": ["Nov_2023","May_2023","Dec_2024"],
      "question_positions": ["Q2A","Q3B","Q2A"],
      "marks_each_time": [10,10,10],
      "question_ids": ["Nov_2023_Q2A","May_2023_Q3B","Dec_2024_Q2A"],
      "consistent_position": false,
      "consistent_marks": true,
      "pattern_note": "Always 10 marks, appears in Q2 or Q3",
      "tip": "specific actionable tip",
      "keywords": ["keyword1","keyword2"]
    }}
  ],
  "predictions": [
    {{
      "question": "full predicted question text",
      "topic": "topic name",
      "confidence": "HIGH",
      "reason": "appeared in 3/4 papers always 10 marks",
      "likely_position": "Q2A",
      "likely_marks": 10,
      "frequency": 3
    }}
  ],
  "paper_pattern": {{
    "compulsory_question": "{uni_info['pattern']}",
    "optional_questions": "Q2-Q6 attempt any 3, Part A and B of 10 marks each",
    "total_marks": {uni_info['total_marks']},
    "duration": "{uni_info['duration']}",
    "key_insight": "specific insight about this subject paper pattern"
  }},
  "study_plan": {{
    "strategy": "2-3 sentence strategy for {user_name} for {subject.upper()} exam",
    "days": [
      {{"day":1,"focus":"topic","priority":"HIGH","hours":3,"tasks":["task1","task2","task3"]}},
      {{"day":2,"focus":"topic","priority":"HIGH","hours":3,"tasks":["task1","task2","task3"]}},
      {{"day":3,"focus":"topic","priority":"HIGH","hours":3,"tasks":["task1","task2","task3"]}},
      {{"day":4,"focus":"topic","priority":"MEDIUM","hours":2,"tasks":["task1","task2","task3"]}},
      {{"day":5,"focus":"topic","priority":"MEDIUM","hours":2,"tasks":["task1","task2","task3"]}},
      {{"day":6,"focus":"topic","priority":"LOW","hours":2,"tasks":["task1","task2"]}},
      {{"day":7,"focus":"Full Revision + Mock Test","priority":"HIGH","hours":4,"tasks":["Revise all HIGH topics","Timed mock test","Review weak areas","Quick reference sheet"]}}
    ],
    "golden_topics": ["topic1","topic2","topic3"],
    "dont_skip": ["topic1","topic2"]
  }}
}}

STRICT: clusters 10-15 sorted freq desc. HIGH=3+ MEDIUM=2 LOW=1.
ONE topic per cluster. Each question_id used ONCE only.
predictions exactly 10. days exactly 7. Return ONLY JSON."""

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role":"system","content":"Expert exam analyzer. Return valid JSON only. No markdown. Each question ID must appear in only one cluster."},
                    {"role":"user","content":prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )
            raw = resp.choices[0].message.content.strip()
            return safe_parse_json(raw)
        except json.JSONDecodeError:
            if attempt == 2: raise
            time.sleep(2)
        except Exception as e:
            err = str(e).lower()
            if 'rate' in err or '429' in err:
                if attempt < 2: time.sleep(30); continue
                raise Exception('Rate limit. Wait 1 minute and retry.')
            raise e


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/universities', methods=['GET'])
def get_universities():
    return jsonify({
        'universities': [{'id': 'mumbai_be', 'name': 'Mumbai University — BE/BTech'}],
        'subjects': {
            'mumbai_be': [
                {'id': 'spcc', 'name': 'System Programming & Compiler Construction'},
                {'id': 'dbms', 'name': 'Database Management Systems'},
                {'id': 'os', 'name': 'Operating Systems'},
                {'id': 'cn', 'name': 'Computer Networks'},
                {'id': 'mc', 'name': 'Mobile Computing'},
                {'id': 'dsa', 'name': 'Data Structures & Algorithms'},
                {'id': 'general', 'name': 'Other Subject'}
            ]
        }
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded.'}), 400
    files = request.files.getlist('files')
    if len(files) < 2:
        return jsonify({'error': 'Upload at least 2 PYQ papers.'}), 400
    if len(files) > 6:
        return jsonify({'error': 'Maximum 6 papers for best accuracy.'}), 400
    non_pdf = [f.filename for f in files if not f.filename.lower().endswith('.pdf')]
    if non_pdf:
        return jsonify({'error': f'Only PDFs supported. Remove: {", ".join(non_pdf)}'}), 400

    user_name = request.form.get('user_name', 'Student').strip() or 'Student'
    university = request.form.get('university', 'mumbai_be').strip()
    subject = request.form.get('subject', 'general').strip()

    if not os.environ.get('GROQ_API_KEY'):
        return jsonify({'error': 'Service not configured.'}), 500

    try: check_quota()
    except Exception as e: return jsonify({'error': str(e)}), 429

    # Only 1 analysis at a time — prevents RAM crash on simultaneous users
    acquired = _analysis_lock.acquire(blocking=False)
    if not acquired:
        return jsonify({'error': 'Server is busy analyzing another request. Please wait 30 seconds and try again.'}), 429

    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
    all_papers_text = {}
    paper_stats = {}
    ocr_used = []

    # Step 1: Extract text from all PDFs
    for file in files:
        if not (file and file.filename.lower().endswith('.pdf')): continue
        filename = secure_filename(file.filename)
        paper_name, sort_key = parse_paper_name(filename)
        base = paper_name; c = 1
        while paper_name in all_papers_text:
            paper_name = f"{base}_{c}"; c += 1

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            text = extract_text_from_pdf(filepath)
            if is_scanned_pdf(text):
                vt = extract_text_via_vision(filepath, client)
                if vt and len(vt.strip()) > 50:
                    text = vt
                    ocr_used.append(paper_name)
                elif not text or len(text.strip()) < 50:
                    return jsonify({'error': f'Cannot read "{filename}". Convert at smallpdf.com → OCR PDF → re-upload.'}), 400
            all_papers_text[paper_name] = (text, sort_key)
            with pdfplumber.open(filepath) as pdf:
                paper_stats[paper_name] = {'pages': len(pdf.pages), 'chars': len(text), 'ocr': paper_name in ocr_used}
        except Exception as e:
            return jsonify({'error': f'Failed: "{filename}": {str(e)}'}), 500
        finally:
            if os.path.exists(filepath): os.remove(filepath)
            gc.collect()  # release RAM immediately

    if len(all_papers_text) < 2:
        return jsonify({'error': 'Need at least 2 readable papers.'}), 400

    # Step 2: Pre-extract structured JSON questions from each paper
    all_structured_questions = []
    for paper_name, (text, _) in sorted(all_papers_text.items(), key=lambda x: x[1][1], reverse=True):
        # Clean text before pre-extraction to remove watermark noise
        clean_text = clean_pdf_text(text)
        questions = pre_extract_questions_json(paper_name, clean_text, client)
        if not questions:
            # Retry with raw text if cleaned version failed
            questions = pre_extract_questions_json(paper_name, text, client)
        all_structured_questions.extend(questions)
        time.sleep(0.5)

    # Fallback if structured extraction failed
    if len(all_structured_questions) < 4:
        all_structured_questions = []
        for paper_name, (text, _) in all_papers_text.items():
            all_structured_questions.append({
                "id": f"{paper_name}_raw",
                "paper": paper_name,
                "position": "unknown",
                "marks": 0,
                "question": text[:500]
            })

    try:
        result = analyze_with_groq(all_papers_text, all_structured_questions, user_name, university, subject)
        clusters = validate_clusters(result.get('clusters', []), all_papers_text)
        predictions = result.get('predictions', [])
        study_plan = result.get('study_plan', {})
        paper_pattern = result.get('paper_pattern', {})
    except json.JSONDecodeError:
        _analysis_lock.release()
        return jsonify({'error': 'Analysis failed. Please try again.'}), 500
    except Exception as e:
        _analysis_lock.release()
        return jsonify({'error': str(e)}), 500
    finally:
        gc.collect()  # clean up after full analysis

    _analysis_lock.release()
    return jsonify({
        'clusters': clusters, 'predictions': predictions,
        'study_plan': study_plan, 'paper_pattern': paper_pattern,
        'user_name': user_name, 'ocr_used': ocr_used,
        'university': UNIVERSITY_DB.get(university, {}).get('name', ''),
        'subject': subject.upper(),
        'stats': {
            'papers': len(all_papers_text),
            'total_questions': len(all_structured_questions),
            'clusters': len(clusters),
            'high_priority': len([c for c in clusters if c.get('importance')=='HIGH']),
            'paper_stats': paper_stats
        }
    })


@app.route('/export', methods=['POST'])
def export_results():
    data = request.json
    clusters = data.get('clusters',[])
    predictions = data.get('predictions',[])
    study_plan = data.get('study_plan',{})
    paper_pattern = data.get('paper_pattern',{})
    stats = data.get('stats',{})
    user_name = data.get('user_name','Student')
    university = data.get('university','')
    subject = data.get('subject','')
    today = datetime.now().strftime('%Y%m%d')

    lines = [
        "╔══════════════════════════════════════════════════╗",
        "║           ASHLYSIS — EXAM INTELLIGENCE           ║",
        "╚══════════════════════════════════════════════════╝","",
        f"Student    : {user_name}",
        f"University : {university}",
        f"Subject    : {subject}",
        f"Date       : {datetime.now().strftime('%d %b %Y %I:%M %p')}",
        f"Papers     : {stats.get('papers',0)}",
        f"Questions  : {stats.get('total_questions',0)}",
        f"Clusters   : {stats.get('clusters',0)}",
        f"HIGH       : {stats.get('high_priority',0)}",
    ]
    if paper_pattern:
        lines += ["","━"*52,"PAPER PATTERN","━"*52,
            f"Q1     : {paper_pattern.get('compulsory_question','')}",
            f"Q2-Q6  : {paper_pattern.get('optional_questions','')}",
            f"Marks  : {paper_pattern.get('total_marks',80)} | {paper_pattern.get('duration','3 hours')}",
            f"Insight: {paper_pattern.get('key_insight','')}"]
    lines += ["","━"*52,"REPEATING QUESTIONS","━"*52]
    for i,c in enumerate(clusters,1):
        pos=c.get('question_positions',[]); marks=c.get('marks_each_time',[])
        lines += [
            f"\n{i}. [{c.get('importance')}] {c.get('topic')} — {c.get('frequency')}x",
            f"   Papers : {', '.join(c.get('papers',[]))}",
            f"   Pos    : {', '.join(str(p) for p in pos)}{'  ✓ ALWAYS SAME' if c.get('consistent_position') else ''}",
            f"   Marks  : {', '.join(str(m) for m in marks)}{'  ✓ ALWAYS SAME' if c.get('consistent_marks') else ''}",
            f"   Pattern: {c.get('pattern_note','')}",
            f"   Tip    : {c.get('tip','')}",
        ]
        for q in c.get('questions',[])[:4]: lines.append(f"   • {q[:200]}")
    lines += ["","━"*52,"PREDICTED QUESTIONS","━"*52]
    for i,p in enumerate(predictions,1):
        lines += [f"\n{i}. [{p.get('confidence')}] {p.get('question','')[:200]}",
                  f"   Pos: {p.get('likely_position','?')} | Marks: {p.get('likely_marks','?')} | {p.get('reason','')}"]
    lines += ["","━"*52,"7-DAY STUDY PLAN","━"*52]
    if study_plan:
        lines += [f"\nStrategy: {study_plan.get('strategy','')}",
                  f"Golden: {', '.join(study_plan.get('golden_topics',[]))}",
                  f"Never skip: {', '.join(study_plan.get('dont_skip',[]))}"]
        for day in study_plan.get('days',[]):
            lines += [f"\nDay {day['day']} [{day['priority']}] — {day['focus']} ({day['hours']}hrs)"]
            for t in day.get('tasks',[]): lines.append(f"  ✓ {t}")
    lines += ["","━"*52,f"ASHLYSIS — generated for {user_name}","━"*52]
    buf = io.BytesIO("\n".join(lines).encode('utf-8'))
    buf.seek(0)
    return send_file(buf, mimetype='text/plain', as_attachment=True,
                     download_name=f'ashlysis_{user_name.replace(" ","_")}_{subject}_{today}.txt')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT',5000)), debug=False)
