import os
import json
import re
import io
import gc
import time
import threading
from datetime import datetime, date
import pdfplumber
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from groq import Groq

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = '/tmp/ashlysis_uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ── CONCURRENT GUARD — 1 analysis at a time ──
_analysis_lock = threading.Semaphore(1)

# ── DAILY QUOTA ──
_quota = {'date': str(date.today()), 'count': 0}
DAILY_LIMIT = 80

def check_quota():
    today = str(date.today())
    if _quota['date'] != today:
        _quota['date'] = today
        _quota['count'] = 0
    if _quota['count'] >= DAILY_LIMIT:
        raise Exception('Daily limit reached. Try again tomorrow.')
    _quota['count'] += 1

# ── UNIVERSITY DB ──
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

# ── CLEAN PDF TEXT ──
def clean_pdf_text(text):
    lines = text.split('\n')
    clean = []
    for line in lines:
        line = line.strip()
        if not line or len(line) < 12: continue
        if re.match(r'^[A-F0-9]{20,}$', line): continue
        if re.match(r'^([A-F0-9]{1,2}\s){6,}', line): continue
        hex_chars = len(re.findall(r'[A-F0-9]', line))
        if len(line) > 20 and hex_chars / max(len(line.replace(' ','')), 1) > 0.75: continue
        if re.match(r'^[*_\-=.]{5,}$', line): continue
        line = re.sub(r'([a-z])([A-Z])', r'\1 \2', line)
        line = re.sub(r'(Q\.?\s*\d+)\.?([A-Za-z])', r'\1. \2', line)
        line = re.sub(r'[.\-_]{4,}', ' ', line)
        line = re.sub(r'\s+', ' ', line)
        clean.append(line)
    return '\n'.join(clean)

# ── PDF EXTRACTION ──
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

# ── SCANNED PDF DETECTION ──
def is_scanned_pdf(text):
    if not text or len(text.strip()) < 150: return True
    cleaned = clean_pdf_text(text)
    if not cleaned or len(cleaned.strip()) < 100: return True
    alpha = len([c for c in cleaned if c.isalpha()]) / max(len(cleaned), 1)
    if alpha < 0.35: return True
    if not re.search(r'Q\.?\s*\d|explain|describe|define|discuss|marks|attempt', cleaned, re.I): return True
    return False

# ── GROQ VISION FOR SCANNED PDFs — single call per paper ──
def extract_text_via_vision(pdf_path, groq_client):
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
        gc.collect()

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
        del stitched, page_images
        gc.collect()

        resp = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": "Engineering exam paper. Extract ALL questions Q1-Q6 with position and marks. Format: Q1a [5m]: question text. One per line. Plain text only."}
            ]}],
            max_tokens=2000, temperature=0.1
        )
        text = resp.choices[0].message.content
        return clean_pdf_text(text) if text else ""
    except Exception:
        return ""

# ── QUESTION EXTRACTION ──
def extract_questions_only(text):
    lines = text.split('\n')
    out = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line and len(line) > 12 and re.search(
            r'^\s*Q\.?\s*\d+|^\s*\d{1,2}\s*[.)]\s*[A-Za-z(]|'
            r'\bexplain\b|\bdescribe\b|\bdefine\b|\bdiscuss\b|\bconstruct\b|'
            r'\bdesign\b|\bcompare\b|\bdifferentiate\b|\bdraw\b|\bwrite\b|'
            r'\bstate\b|\blist\b|\bwhat\b|\bhow\b|\bwhy\b|\bderive\b|'
            r'\bcalculate\b|\bimplement\b|short\s*note|advantages|flowchart|'
            r'difference between|types of|working of|phases of|with example',
            line, re.I):
            combined = line
            if i+1 < len(lines):
                nxt = lines[i+1].strip()
                if nxt and len(nxt) > 10 and not re.match(r'^\s*Q\.?\s*\d+|\d{1,2}\s*[.)]', nxt):
                    combined += ' ' + nxt
                    i += 1
            out.append(combined[:300])
        i += 1
    return '\n'.join(out)

# ── CLUSTER VALIDATION ──
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
            **c, 'frequency': freq, 'importance': importance,
            'papers': valid_papers if valid_papers else c.get('papers', []),
            'consistent_marks': consistent_marks, 'consistent_position': consistent_position,
            'questions': questions,
            'marks_each_time': marks if marks else c.get('marks_each_time', []),
            'question_positions': positions
        })
    valid.sort(key=lambda x: (-x['frequency'], ['LOW','MEDIUM','HIGH'].index(x.get('importance','LOW'))))
    return valid

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

# ── MAIN ANALYSIS — SINGLE GROQ CALL ──
def analyze_with_groq(all_papers_text, user_name, university, subject):
    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
    sorted_papers = sorted(all_papers_text.items(), key=lambda x: x[1][1], reverse=True)

    uni_info = UNIVERSITY_DB.get(university, UNIVERSITY_DB['mumbai_be'])
    topic_list = SUBJECT_TOPICS.get(subject, SUBJECT_TOPICS['general'])
    num_papers = len(sorted_papers)

    # Dynamic chars per paper — keeps total tokens under limit
    chars_per_paper = min(1200, max(600, 6000 // num_papers))

    papers_content = ""
    for paper_name, (text, _) in sorted_papers:
        if not text or len(text.strip()) < 50: continue
        # Clean again right before sending
        text = clean_pdf_text(text)
        qs = extract_questions_only(text)
        if len(qs.strip()) < 30: qs = text
        papers_content += f"\n\n=== {paper_name} ===\n{qs[:chars_per_paper]}"

    if not papers_content.strip():
        raise Exception("Could not extract questions from papers")

    prompt = f"""You are an expert exam analyzer for {uni_info['name']} students.
Student: {user_name} | Subject: {subject.upper()} | Papers: {num_papers}

PAPER STRUCTURE: {uni_info['structure']}

ALL EXAM PAPERS — read every single question in every paper:
{papers_content}

KNOWN TOPICS FOR {subject.upper()} — find each one across all papers:
{chr(10).join(f"- {t}" for t in topic_list)}

CRITICAL RULES:
- One question can only belong to ONE cluster
- Two questions from the SAME paper cannot be in the same cluster
- A cluster = same topic appearing in DIFFERENT papers
- Every paper listed above must appear in at least one cluster

Return ONLY valid JSON:
{{
  "clusters": [
    {{
      "topic": "exact topic name",
      "frequency": 3,
      "importance": "HIGH",
      "questions": ["Q text paper1", "Q text paper2", "Q text paper3"],
      "papers": ["Nov_2023","May_2023","Dec_2024"],
      "question_positions": ["Q2A","Q3B","Q2A"],
      "marks_each_time": [10,10,10],
      "consistent_position": false,
      "consistent_marks": true,
      "pattern_note": "Always 10 marks, appears in Q2 or Q3",
      "tip": "actionable exam tip",
      "keywords": ["keyword1","keyword2"]
    }}
  ],
  "predictions": [
    {{
      "question": "predicted question text",
      "topic": "topic name",
      "confidence": "HIGH",
      "reason": "why likely",
      "likely_position": "Q2A",
      "likely_marks": 10,
      "frequency": 3
    }}
  ],
  "paper_pattern": {{
    "compulsory_question": "{uni_info['pattern']}",
    "optional_questions": "Q2-Q6 attempt any 3, Part A and B 10 marks each",
    "total_marks": {uni_info['total_marks']},
    "duration": "{uni_info['duration']}",
    "key_insight": "key pattern insight for this subject"
  }},
  "study_plan": {{
    "strategy": "2-3 sentence strategy for {user_name}",
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

STRICT: clusters 10-15 freq desc. HIGH=3+ MEDIUM=2 LOW=1.
ONE topic per cluster. predictions 10. days 7. Return ONLY JSON."""

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role":"system","content":"Expert exam analyzer. Return valid JSON only. No markdown."},
                    {"role":"user","content":prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )
            return safe_parse_json(resp.choices[0].message.content.strip())
        except json.JSONDecodeError:
            if attempt == 2: raise
            time.sleep(2)
        except Exception as e:
            err = str(e).lower()
            if 'rate' in err or '429' in err:
                if attempt < 2:
                    time.sleep(35)
                    continue
                raise Exception('Rate limit reached. Please wait 1 minute and try again.')
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
        return jsonify({'error': 'Maximum 6 papers for best results.'}), 400
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

    acquired = _analysis_lock.acquire(blocking=False)
    if not acquired:
        return jsonify({'error': 'Server is busy. Please wait 30 seconds and try again.'}), 429

    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
    all_papers_text = {}
    paper_stats = {}
    ocr_used = []

    try:
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
                        _analysis_lock.release()
                        return jsonify({'error': f'Cannot read "{filename}". Convert at smallpdf.com → OCR PDF → re-upload.'}), 400
                all_papers_text[paper_name] = (text, sort_key)
                with pdfplumber.open(filepath) as pdf:
                    paper_stats[paper_name] = {'pages': len(pdf.pages), 'chars': len(text), 'ocr': paper_name in ocr_used}
            except Exception as e:
                _analysis_lock.release()
                return jsonify({'error': f'Failed: "{filename}": {str(e)}'}), 500
            finally:
                if os.path.exists(filepath): os.remove(filepath)
                gc.collect()

        if len(all_papers_text) < 2:
            _analysis_lock.release()
            return jsonify({'error': 'Need at least 2 readable papers.'}), 400

        result = analyze_with_groq(all_papers_text, user_name, university, subject)
        clusters = validate_clusters(result.get('clusters', []), all_papers_text)
        predictions = result.get('predictions', [])
        study_plan = result.get('study_plan', {})
        paper_pattern = result.get('paper_pattern', {})

    except json.JSONDecodeError:
        return jsonify({'error': 'Analysis failed. Please try again.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        gc.collect()
        _analysis_lock.release()

    return jsonify({
        'clusters': clusters, 'predictions': predictions,
        'study_plan': study_plan, 'paper_pattern': paper_pattern,
        'user_name': user_name, 'ocr_used': ocr_used,
        'university': UNIVERSITY_DB.get(university, {}).get('name', ''),
        'subject': subject.upper(),
        'stats': {
            'papers': len(all_papers_text),
            'total_questions': sum(s.get('chars',0)//80 for s in paper_stats.values()),
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
        f"Clusters   : {stats.get('clusters',0)}",
        f"HIGH       : {stats.get('high_priority',0)}",
    ]
    if paper_pattern:
        lines += ["","━"*52,"PAPER PATTERN","━"*52,
            f"Q1    : {paper_pattern.get('compulsory_question','')}",
            f"Q2-Q6 : {paper_pattern.get('optional_questions','')}",
            f"Marks : {paper_pattern.get('total_marks',80)} | {paper_pattern.get('duration','3 hours')}",
            f"Tip   : {paper_pattern.get('key_insight','')}"]
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
