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

def clean_pdf_text(text):
    lines = text.split('\n')
    clean = []
    for line in lines:
        line = line.strip()
        if not line or len(line) < 3:
            continue
        if re.match(r'^[A-F0-9]{20,}$', line): continue
        if re.match(r'^[*_\-=.]{5,}$', line): continue
        line = re.sub(r'([a-z])([A-Z])', r'\1 \2', line)
        line = re.sub(r'(Q\d+)\.?([A-Z])', r'\1. \2', line)
        line = re.sub(r'[.\-_]{4,}', ' ', line)
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
    alpha = len([c for c in text if c.isalpha()]) / max(len(text), 1)
    if alpha < 0.3: return True
    if not re.search(r'Q\.?\s*\d|explain|describe|define|discuss', text, re.I): return True
    return False

def extract_text_via_vision(pdf_path, groq_client):
    try:
        import fitz, base64
        doc = fitz.open(pdf_path)
        all_text = []
        for page_num in range(min(len(doc), 3)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72))
            img_b64 = base64.b64encode(pix.tobytes("jpeg")).decode()
            try:
                resp = groq_client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role":"user","content":[
                        {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img_b64}"}},
                        {"type":"text","text":"Extract every exam question with number and marks. Format: Q1a [5m]: question text. Include Q1-Q6. Plain text only."}
                    ]}],
                    max_tokens=1500, temperature=0.1
                )
                t = resp.choices[0].message.content
                if t and t.strip(): all_text.append(t.strip())
            except Exception: continue
        doc.close()
        return clean_pdf_text("\n".join(all_text))
    except Exception: return ""

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
                if nxt and len(nxt)>10 and not re.match(r'^\s*Q\.?\s*\d+|\d{1,2}\s*[.)]', nxt):
                    combined += ' ' + nxt
                    i += 1
            out.append(combined[:350])
        i += 1
    return '\n'.join(out)

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
            ct = text[s:lc].rstrip(',')
            return json.loads(f'{{"clusters":[{ct}],"predictions":[],"study_plan":{{"strategy":"Focus on HIGH priority topics.","days":[],"golden_topics":[],"dont_skip":[]}},"paper_pattern":{{}}}}')
    except: pass
    raise json.JSONDecodeError("Cannot parse",text,0)

def analyze_with_groq(all_papers_text, user_name):
    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
    sorted_papers = sorted(all_papers_text.items(), key=lambda x: x[1][1], reverse=True)

    # Build compact question summary for ALL papers
    papers_content = ""
    for paper_name, (text, _) in sorted_papers:
        if not text or len(text.strip()) < 50: continue
        qs = extract_questions_only(text)
        if len(qs.strip()) < 30: qs = text
        # 5000 chars per paper of pure question lines
        papers_content += f"\n\n=== {paper_name} ===\n{qs[:5000]}"

    if not papers_content.strip():
        raise Exception("Could not extract questions from papers")

    prompt = f"""You are an expert exam analyzer for Mumbai University engineering students.
Student: {user_name} | Papers: {len(sorted_papers)}

EXAM PAPERS (latest first — read ALL questions Q1 through Q6 in EVERY paper):
{papers_content}

YOUR JOB: Find every question that repeats across papers. Same topic = same cluster even if wording differs.

MANDATORY TOPICS TO CHECK IN EVERY PAPER:
- Two-pass assembler Pass 1 flowchart (Q2 or Q6 usually, 10 marks)
- Forward reference problem (Q1 usually, 5 marks)
- Direct/Absolute/Dynamic Linking Loader (Q3-Q5 usually, 10 marks)
- Phases of compiler (Q4-Q6 usually, 10 marks)
- Code optimization techniques (Q5-Q6 usually, 10 marks)
- Macro processor single/two-pass (Q3-Q4 usually, 10 marks)
- Intermediate code / Three address code / Basic blocks (Q2-Q3, 10 marks)
- Parser SLR/LL1/operator precedence/predictive (Q2-Q3, 10 marks)
- System software vs application software (Q1, 5 marks)
- Assembler directives and statements (Q4, 10 marks)

Return ONLY valid JSON (no markdown, no explanation):
{{
  "clusters": [
    {{
      "topic": "Two-pass Assembler Pass 1",
      "frequency": 4,
      "importance": "HIGH",
      "questions": ["Draw flowchart of pass1 of assembler Nov2023", "Explain Pass-I of two pass assembler May2023", "Draw and explain flowchart Pass-I Dec2024", "flowchart of Pass-I two pass assembler May2025"],
      "papers": ["Nov_2023","May_2023","Dec_2024","May_2025"],
      "question_positions": ["Q2a","Q6B","Q2A","Q2A"],
      "marks_each_time": [10,10,10,10],
      "consistent_position": false,
      "consistent_marks": true,
      "pattern_note": "Always 10 marks. Appears in Q2 or Q6 every year without exception.",
      "tip": "Draw complete flowchart with SYMTAB, LOCCTR, OPTAB. Label every box and arrow.",
      "keywords": ["assembler","pass1","flowchart","symtab"]
    }}
  ],
  "predictions": [
    {{
      "question": "Draw the flowchart of Pass-I of two-pass assembler and explain its working with the databases used.",
      "topic": "Two-pass Assembler Pass 1",
      "confidence": "HIGH",
      "reason": "Appeared in all 4 papers always 10 marks never missed",
      "likely_position": "Q2A",
      "likely_marks": 10,
      "frequency": 4
    }}
  ],
  "paper_pattern": {{
    "compulsory_question": "Q1 always compulsory — 4 parts (a,b,c,d) of 5 marks each = 20 marks total",
    "optional_questions": "Q2 to Q6 — attempt any 3 questions. Each question has Part A and Part B of 10 marks each = 20 marks per question",
    "total_marks": 80,
    "duration": "3 hours",
    "key_insight": "Strategy: nail Q1 (20 marks) + prepare 3 full questions from Q2-Q6 (60 marks) = 80 marks"
  }},
  "study_plan": {{
    "strategy": "2-3 sentence personalized plan for {user_name} based on these papers",
    "days": [
      {{"day":1,"focus":"topic1","priority":"HIGH","hours":3,"tasks":["task1","task2","task3"]}},
      {{"day":2,"focus":"topic2","priority":"HIGH","hours":3,"tasks":["task1","task2","task3"]}},
      {{"day":3,"focus":"topic3","priority":"HIGH","hours":3,"tasks":["task1","task2","task3"]}},
      {{"day":4,"focus":"topic4","priority":"MEDIUM","hours":2,"tasks":["task1","task2","task3"]}},
      {{"day":5,"focus":"topic5","priority":"MEDIUM","hours":2,"tasks":["task1","task2","task3"]}},
      {{"day":6,"focus":"topic6","priority":"LOW","hours":2,"tasks":["task1","task2"]}},
      {{"day":7,"focus":"Full Revision + Mock Test","priority":"HIGH","hours":4,"tasks":["Revise all HIGH priority topics","Attempt timed mock using past questions","Review weak areas","Prepare quick reference cheat sheet"]}}
    ],
    "golden_topics": ["most important topic","second","third"],
    "dont_skip": ["critical topic 1","critical topic 2"]
  }}
}}

STRICT RULES:
- clusters: minimum 10, maximum 15, sorted by frequency descending
- HIGH = appeared in 3+ papers, MEDIUM = 2 papers, LOW = 1 paper but important
- ONE distinct topic per cluster — never combine assembler + loader + compiler
- consistent_marks: true ONLY when every single marks value is identical
- predictions: exactly 10 items
- study_plan days: exactly 7 items
- Return ONLY the JSON object"""

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role":"system","content":"Expert exam question analyzer for Mumbai University. Return valid JSON only. No markdown. No explanation."},
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


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded.'}), 400
    files = request.files.getlist('files')
    if len(files) < 2:
        return jsonify({'error': 'Upload at least 2 PYQ papers.'}), 400
    if len(files) > 10:
        return jsonify({'error': 'Maximum 10 papers.'}), 400
    non_pdf = [f.filename for f in files if not f.filename.lower().endswith('.pdf')]
    if non_pdf:
        return jsonify({'error': f'Only PDFs supported. Remove: {", ".join(non_pdf)}'}), 400

    user_name = request.form.get('user_name','Student').strip() or 'Student'
    if not os.environ.get('GROQ_API_KEY'):
        return jsonify({'error': 'Service not configured.'}), 500

    try: check_quota()
    except Exception as e: return jsonify({'error': str(e)}), 429

    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
    all_papers_text = {}
    paper_stats = {}
    ocr_used = []

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
            return jsonify({'error': f'Failed to read "{filename}": {str(e)}'}), 500
        finally:
            if os.path.exists(filepath): os.remove(filepath)

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
        'clusters': clusters, 'predictions': predictions,
        'study_plan': study_plan, 'paper_pattern': paper_pattern,
        'user_name': user_name, 'ocr_used': ocr_used,
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
    today = datetime.now().strftime('%Y%m%d')

    lines = [
        "╔══════════════════════════════════════════════════╗",
        "║           ASHLYSIS — EXAM INTELLIGENCE           ║",
        "╚══════════════════════════════════════════════════╝","",
        f"Student  : {user_name}",
        f"Date     : {datetime.now().strftime('%d %b %Y %I:%M %p')}",
        f"Papers   : {stats.get('papers',0)}",
        f"Clusters : {stats.get('clusters',0)}",
        f"HIGH     : {stats.get('high_priority',0)}",
    ]
    if paper_pattern:
        lines += ["","━"*52,"PAPER PATTERN","━"*52,
            f"Q1 : {paper_pattern.get('compulsory_question','')}",
            f"Q2-Q6 : {paper_pattern.get('optional_questions','')}",
            f"Marks : {paper_pattern.get('total_marks',80)} | {paper_pattern.get('duration','3 hours')}",
            f"Tip : {paper_pattern.get('key_insight','')}"]
    lines += ["","━"*52,"REPEATING QUESTIONS","━"*52]
    for i,c in enumerate(clusters,1):
        pos = c.get('question_positions',[])
        marks = c.get('marks_each_time',[])
        lines += [
            f"\n{i}. [{c.get('importance')}] {c.get('topic')} — {c.get('frequency')}x",
            f"   Papers : {', '.join(c.get('papers',[]))}",
            f"   Pos    : {', '.join(str(p) for p in pos)}{'  ✓ ALWAYS SAME' if c.get('consistent_position') else ''}",
            f"   Marks  : {', '.join(str(m) for m in marks)}{'  ✓ ALWAYS SAME' if c.get('consistent_marks') else ''}",
            f"   Pattern: {c.get('pattern_note','')}",
            f"   Tip    : {c.get('tip','')}",
        ]
        for q in c.get('questions',[])[:4]:
            lines.append(f"   • {q[:200]}")
    lines += ["","━"*52,"PREDICTED QUESTIONS","━"*52]
    for i,p in enumerate(predictions,1):
        lines += [
            f"\n{i}. [{p.get('confidence')}] {p.get('question','')[:200]}",
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
                     download_name=f'ashlysis_{user_name.replace(" ","_")}_{today}.txt')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT',5000)), debug=False)
