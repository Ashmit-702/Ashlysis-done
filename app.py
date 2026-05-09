import os
import json
import re
import io
import gc
import time
import threading
import math
from collections import Counter, defaultdict
from datetime import datetime, date
import pdfplumber
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from groq import Groq

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = '/tmp/ashlysis_uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

_analysis_lock = threading.Semaphore(1)
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

UNIVERSITY_DB = {
    "mumbai_be": {
        "name": "Mumbai University — BE",
        "structure": "Q1 compulsory (4 parts × 5 marks = 20 marks). Q2-Q6 attempt any 3 (2 parts × 10 marks = 20 marks each). Total: 80 marks, 3 hours.",
        "q1_marks": 5, "optional_marks": 10, "total_marks": 80, "duration": "3 hours",
        "pattern": "Q1a/b/c/d = 5 marks each. Q2A/B through Q6A/B = 10 marks each."
    }
}

SUBJECT_TOPICS = {
    "spcc": [
        "Two-pass Assembler Pass 1 flowchart",
        "Forward reference problem",
        "Direct Linking Loader Absolute Loader Dynamic Loader",
        "Phases of Compiler",
        "Code Optimization techniques dead code common subexpression",
        "Macro Processor single-pass two-pass",
        "Intermediate Code Three Address Code Basic Blocks Flow Graph",
        "Parser SLR LL1 Operator Precedence Predictive",
        "System Software vs Application Software",
        "Assembler directives and statements"
    ],
    "mc": [
        "GSM System Architecture BSS NSS OSS",
        "GPRS Architecture SGSN GGSN",
        "Mobile Terminated Call Mobile Originated Call",
        "Handover mechanisms GSM hard soft inter-cell",
        "GSM Security A3 A5 A8 authentication",
        "Snooping TCP Mobile TCP",
        "Mobile IP agent discovery registration tunnelling",
        "IEEE 802.11 WLAN MAC protocol",
        "Hidden station Exposed station problem",
        "Frequency Reuse cell clustering co-channel",
        "LTE 4G UMTS 3G architecture",
        "Bluetooth protocol stack piconet",
        "WAP architecture",
        "Spread Spectrum FHSS DSSS"
    ],
    "os": [
        "Process scheduling FCFS SJF Round Robin Priority",
        "Deadlock detection prevention avoidance Bankers algorithm",
        "Memory management paging segmentation",
        "Semaphore mutex critical section producer consumer",
        "Virtual memory page replacement LRU FIFO Optimal",
        "File system allocation directory structure",
        "Disk scheduling SSTF SCAN C-SCAN",
        "Process synchronization monitors",
        "Thrashing working set",
        "Inter process communication IPC"
    ],
    "dbms": [
        "Normalization 1NF 2NF 3NF BCNF",
        "SQL queries joins",
        "ER diagram",
        "Transaction ACID properties",
        "Concurrency control",
        "Relational algebra",
        "Indexing B-tree",
        "Recovery techniques"
    ],
    "cn": [
        "OSI model layers",
        "TCP IP protocol",
        "Routing algorithms",
        "Congestion control",
        "Error detection correction",
        "Medium access control",
        "Socket programming",
        "Network security"
    ],
    "dsa": [
        "Sorting algorithms time complexity",
        "Tree traversal BST",
        "Graph BFS DFS",
        "Dynamic programming",
        "Hashing",
        "Stack queue linked list",
        "Heap priority queue",
        "Divide and conquer"
    ],
    "ai": [
        "Uninformed search BFS DFS Iterative Deepening",
        "Informed search A* Best First Heuristic",
        "Hill Climbing Simulated Annealing local search",
        "Minimax Alpha Beta Pruning game tree",
        "Constraint Satisfaction Problem CSP backtracking",
        "Propositional Logic First Order Logic resolution",
        "Bayesian Network probabilistic reasoning",
        "Machine Learning supervised unsupervised",
        "Neural Network perceptron backpropagation",
        "Natural Language Processing",
        "Planning STRIPS state space",
        "Knowledge Representation frames semantic net"
    ],
    "ml": [
        "Linear Regression Logistic Regression",
        "Decision Tree Random Forest",
        "Support Vector Machine SVM kernel",
        "K-Means clustering unsupervised",
        "Neural Network deep learning CNN RNN",
        "Overfitting underfitting regularization",
        "Cross validation train test split",
        "Naive Bayes classifier",
        "Principal Component Analysis PCA",
        "Reinforcement Learning Q-learning"
    ],
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

def is_scanned_pdf(text):
    if not text or len(text.strip()) < 150: return True
    cleaned = clean_pdf_text(text)
    if not cleaned or len(cleaned.strip()) < 100: return True
    alpha = len([c for c in cleaned if c.isalpha()]) / max(len(cleaned), 1)
    if alpha < 0.35: return True
    if not re.search(r'Q\.?\s*\d|explain|describe|define|discuss|marks|attempt', cleaned, re.I): return True
    return False

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

        if not page_images: return ""

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

# ── STEP 1: EXTRACT QUESTIONS AS JSON (per paper) ──
def extract_questions_as_json(paper_name, text, client):
    prompt = f"""Extract ALL exam questions from this paper as a JSON array.

Paper: {paper_name}
Text:
{text[:5000]}

Return ONLY a JSON array:
[
  {{"id":"{paper_name}_Q1a","paper":"{paper_name}","position":"Q1a","marks":5,"question":"exact question text"}},
  {{"id":"{paper_name}_Q2A","paper":"{paper_name}","position":"Q2A","marks":10,"question":"exact question text"}}
]

Rules:
- Include ALL questions from Q1a through Q6B
- Each question gets unique id: paper_position
- marks: 5 for Q1 parts, 10 for Q2-Q6, 0 if unclear
- Extract exact question text
- Return ONLY the JSON array, no markdown"""

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Extract exam questions as JSON array. No markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=2000
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r'^```(?:json)?\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw).strip()
        questions = json.loads(raw)
        for q in questions:
            q['paper'] = paper_name
        return questions
    except Exception:
        return []

# ── STEP 2: PYTHON TF-IDF CLUSTERING ──
STOPWORDS = {
    'a','an','the','and','or','but','in','on','at','to','for','of','with',
    'is','are','was','were','be','been','have','has','had','do','does','did',
    'will','would','could','should','may','might','can','what','how','why',
    'when','where','which','who','that','this','it','from','by','as','if',
    'explain','describe','define','discuss','write','short','note','answer',
    'question','marks','following','given','using','use','give','find',
    'state','list','show','prove','derive','calculate','determine','compare',
    'with','example','suitable','neat','diagram','detail','brief'
}

def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    return tokens

def cosine_sim(v1, v2):
    common = set(v1) & set(v2)
    if not common: return 0.0
    dot = sum(v1[k] * v2[k] for k in common)
    m1 = math.sqrt(sum(x**2 for x in v1.values()))
    m2 = math.sqrt(sum(x**2 for x in v2.values()))
    if m1 == 0 or m2 == 0: return 0.0
    return dot / (m1 * m2)

def build_tfidf(questions):
    N = len(questions)
    tokenized = [tokenize(q['question']) for q in questions]
    df = Counter()
    for tokens in tokenized:
        for t in set(tokens):
            df[t] += 1
    idf = {t: math.log((N+1)/(df[t]+1))+1 for t in df}
    vectors = []
    for tokens in tokenized:
        tf = Counter(tokens)
        total = max(len(tokens), 1)
        vec = {t: (tf[t]/total)*idf.get(t,1) for t in tf}
        vectors.append(vec)
    return vectors

TOPIC_NEGATIVE_KEYWORDS = {
    "Snooping TCP Mobile TCP": ["small cells", "heterogeneous", "femtocell", "picocell"],
    "Mobile IP agent discovery registration tunnelling": ["voip", "voice over", "ims"],
    "GPRS Architecture SGSN GGSN": ["handover", "hand off", "roaming process"],
}

def match_topic(question_text, topic_keywords):
    q_lower = question_text.lower()
    for topic_key, neg_words in TOPIC_NEGATIVE_KEYWORDS.items():
        if topic_key.lower() in topic_keywords.lower():
            if any(neg in q_lower for neg in neg_words):
                return False
    topic_words = tokenize(topic_keywords)
    if not topic_words: return False
    matches = sum(1 for w in topic_words if w in q_lower)
    threshold = max(2, math.ceil(len(topic_words) * 0.4))
    return matches >= threshold

def cluster_questions_python(all_questions, subject, num_papers):
    topic_list = SUBJECT_TOPICS.get(subject, SUBJECT_TOPICS['general'])
    if len(all_questions) < 2:
        return []

    vectors = build_tfidf(all_questions)
    topic_assignments = {}
    topic_clusters = defaultdict(list)

    for i, q in enumerate(all_questions):
        best_topic = None
        best_score = 0
        for topic in topic_list:
            if match_topic(q['question'], topic):
                topic_vec = {}
                for word in tokenize(topic):
                    topic_vec[word] = 1.0
                score = cosine_sim(vectors[i], topic_vec)
                if score > best_score:
                    best_score = score
                    best_topic = topic
        if best_topic and best_score > 0.12:
            topic_assignments[q['id']] = best_topic
            topic_clusters[best_topic].append(q)

    unmatched = [q for q in all_questions if q['id'] not in topic_assignments]
    if unmatched:
        unmatched_vectors = [vectors[all_questions.index(q)] for q in unmatched if q in all_questions]
        visited = [False] * len(unmatched)
        for i in range(len(unmatched)):
            if visited[i]: continue
            cluster = [unmatched[i]]
            visited[i] = True
            for j in range(i+1, len(unmatched)):
                if visited[j]: continue
                if i < len(unmatched_vectors) and j < len(unmatched_vectors):
                    sim = cosine_sim(unmatched_vectors[i], unmatched_vectors[j])
                    if sim > 0.4 and unmatched[i]['paper'] != unmatched[j]['paper']:
                        cluster.append(unmatched[j])
                        visited[j] = True
            if len(cluster) >= 2:
                papers_in = list(set(q['paper'] for q in cluster))
                if len(papers_in) >= 2:
                    all_words = tokenize(' '.join(q['question'] for q in cluster))
                    freq = Counter(all_words)
                    topic_name = ' '.join(w.title() for w, _ in freq.most_common(3))
                    topic_clusters[topic_name].extend(cluster)

    final_clusters = []
    for topic, questions in topic_clusters.items():
        paper_best = {}
        for q in questions:
            paper = q['paper']
            if paper not in paper_best:
                paper_best[paper] = q
            else:
                if len(q['question']) > len(paper_best[paper]['question']):
                    paper_best[paper] = q

        unique_questions = list(paper_best.values())
        papers = [q['paper'] for q in unique_questions]

        if len(papers) < 2:
            continue

        freq = len(papers)
        if freq >= 3: importance = 'HIGH'
        elif freq == 2: importance = 'MEDIUM'
        else: importance = 'LOW'

        marks = [q.get('marks', 0) for q in unique_questions]
        marks_nonzero = [m for m in marks if m > 0]
        consistent_marks = len(set(marks_nonzero)) == 1 if len(marks_nonzero) > 1 else False

        positions = [q.get('position', '') for q in unique_questions]
        q_nums = [re.match(r'Q\d+', p, re.I).group() if re.match(r'Q\d+', p, re.I) else p for p in positions]
        consistent_position = len(set(q_nums)) == 1 if len(q_nums) > 1 else False

        final_clusters.append({
            'topic': topic,
            'frequency': freq,
            'importance': importance,
            'questions': [q['question'] for q in unique_questions],
            'papers': papers,
            'question_positions': positions,
            'marks_each_time': marks_nonzero,
            'consistent_position': consistent_position,
            'consistent_marks': consistent_marks,
            'pattern_note': '',
            'tip': '',
            'keywords': []
        })

    final_clusters.sort(key=lambda x: (-x['frequency'], ['LOW','MEDIUM','HIGH'].index(x.get('importance','LOW'))))

    seen_qids = set()
    deduped = []
    for cluster in final_clusters:
        papers = cluster.get('papers', [])
        positions = cluster.get('question_positions', [])
        questions = cluster.get('questions', [])
        marks = cluster.get('marks_each_time', [])

        clean_idx = []
        for i, paper in enumerate(papers):
            qid = f"{paper}_{positions[i] if i < len(positions) else i}".lower()
            if qid not in seen_qids:
                seen_qids.add(qid)
                clean_idx.append(i)

        cluster['papers'] = [papers[i] for i in clean_idx if i < len(papers)]
        cluster['question_positions'] = [positions[i] for i in clean_idx if i < len(positions)]
        cluster['questions'] = [questions[i] for i in clean_idx if i < len(questions)]
        cluster['marks_each_time'] = [marks[i] for i in clean_idx if i < len(marks)]
        cluster['frequency'] = len(cluster['papers'])

        if cluster['frequency'] < 2:
            continue

        f = cluster['frequency']
        if f >= 3: cluster['importance'] = 'HIGH'
        elif f == 2: cluster['importance'] = 'MEDIUM'
        else: cluster['importance'] = 'LOW'

        deduped.append(cluster)

    return deduped[:15]

# ── STEP 3: GROQ ENRICHES CLUSTERS + GENERATES PREDICTIONS ──
def enrich_with_groq(clusters, all_questions, user_name, university, subject, client):
    """
    Groq job 2: adds tips + predictions + study plan.
    Now receives all_questions too so predictions have full context
    even when clustering found few repeats.
    """
    uni_info = UNIVERSITY_DB.get(university, UNIVERSITY_DB['mumbai_be'])

    # Build cluster summary for tips
    cluster_summary = json.dumps([{
        'topic': c['topic'],
        'frequency': c['frequency'],
        'importance': c['importance'],
        'papers': c['papers'],
        'sample_question': c['questions'][0][:150] if c['questions'] else ''
    } for c in clusters], indent=2)[:4000]

    # Build all-questions summary for better predictions
    # Use top repeated topics + sample questions from all papers
    all_q_summary = ""
    if all_questions:
        # Group by paper
        by_paper = defaultdict(list)
        for q in all_questions:
            by_paper[q['paper']].append(q)
        lines = []
        for paper, qs in list(by_paper.items())[:6]:
            lines.append(f"\n--- {paper} ---")
            for q in qs[:12]:
                lines.append(f"  {q.get('position','?')} [{q.get('marks',0)}m]: {q['question'][:120]}")
        all_q_summary = '\n'.join(lines)[:3000]

    prompt = f"""You are an expert exam analyst for {uni_info['name']} {subject.upper()} students.
Student: {user_name}

REPEAT TOPICS IDENTIFIED (DO NOT change these):
{cluster_summary}

ALL QUESTIONS FROM ALL PAPERS (use for prediction context):
{all_q_summary}

Your tasks:
1. For each cluster topic, write a short exam tip and pattern note.
2. Generate EXACTLY 10 high-quality predicted questions for the NEXT exam — base them on:
   - Topics that repeat most (HIGH importance)
   - Questions not asked recently (gap analysis)
   - Common exam patterns for {subject.upper()}
   Each prediction must be a FULL question (not just a topic name).
3. Paper pattern info for {uni_info['name']}.
4. 7-day study plan.

Return ONLY this JSON (no markdown, no extra text):
{{
  "tips": {{
    "TOPIC_NAME": {{"tip": "practical exam tip", "pattern_note": "when/where pattern"}}
  }},
  "predictions": [
    {{"question": "full question text here", "topic": "topic name", "confidence": "HIGH", "reason": "brief reason based on pattern", "likely_position": "Q2A", "likely_marks": 10, "frequency": 3}},
    {{"question": "full question text here", "topic": "topic name", "confidence": "HIGH", "reason": "brief reason", "likely_position": "Q3B", "likely_marks": 10, "frequency": 2}},
    {{"question": "full question text here", "topic": "topic name", "confidence": "MEDIUM", "reason": "brief reason", "likely_position": "Q1a", "likely_marks": 5, "frequency": 2}},
    {{"question": "full question text here", "topic": "topic name", "confidence": "MEDIUM", "reason": "brief reason", "likely_position": "Q4A", "likely_marks": 10, "frequency": 2}},
    {{"question": "full question text here", "topic": "topic name", "confidence": "MEDIUM", "reason": "brief reason", "likely_position": "Q5B", "likely_marks": 10, "frequency": 1}},
    {{"question": "full question text here", "topic": "topic name", "confidence": "MEDIUM", "reason": "brief reason", "likely_position": "Q2B", "likely_marks": 10, "frequency": 1}},
    {{"question": "full question text here", "topic": "topic name", "confidence": "LOW", "reason": "brief reason", "likely_position": "Q6A", "likely_marks": 10, "frequency": 1}},
    {{"question": "full question text here", "topic": "topic name", "confidence": "LOW", "reason": "brief reason", "likely_position": "Q3A", "likely_marks": 10, "frequency": 1}},
    {{"question": "full question text here", "topic": "topic name", "confidence": "LOW", "reason": "brief reason", "likely_position": "Q1b", "likely_marks": 5, "frequency": 1}},
    {{"question": "full question text here", "topic": "topic name", "confidence": "LOW", "reason": "brief reason", "likely_position": "Q5A", "likely_marks": 10, "frequency": 1}}
  ],
  "paper_pattern": {{
    "compulsory_question": "{uni_info['pattern']}",
    "optional_questions": "Q2-Q6 attempt any 3, Part A and B 10 marks each",
    "total_marks": {uni_info['total_marks']},
    "duration": "{uni_info['duration']}",
    "key_insight": "one key insight for {subject.upper()} exam"
  }},
  "study_plan": {{
    "strategy": "2-3 sentence personalized strategy for {user_name}",
    "days": [
      {{"day":1,"focus":"topic name","priority":"HIGH","hours":3,"tasks":["task1","task2","task3"]}},
      {{"day":2,"focus":"topic name","priority":"HIGH","hours":3,"tasks":["task1","task2","task3"]}},
      {{"day":3,"focus":"topic name","priority":"HIGH","hours":3,"tasks":["task1","task2","task3"]}},
      {{"day":4,"focus":"topic name","priority":"MEDIUM","hours":2,"tasks":["task1","task2","task3"]}},
      {{"day":5,"focus":"topic name","priority":"MEDIUM","hours":2,"tasks":["task1","task2","task3"]}},
      {{"day":6,"focus":"topic name","priority":"LOW","hours":2,"tasks":["task1","task2"]}},
      {{"day":7,"focus":"Full Revision + Mock Test","priority":"HIGH","hours":4,"tasks":["Revise all HIGH topics","Attempt timed mock","Review weak areas","Make quick reference sheet"]}}
    ],
    "golden_topics": ["topic1","topic2","topic3","topic4"],
    "dont_skip": ["topic1","topic2","topic3"]
  }}
}}

CRITICAL: predictions must be EXACTLY 10 items. Each "question" must be a full sentence question, not just a topic name. Return ONLY JSON."""

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Exam analyst. Return valid JSON only. No markdown fences. predictions array must have exactly 10 items."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=3500
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r'^```(?:json)?\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw).strip()
        enrichment = json.loads(raw)

        # Apply tips to clusters
        tips_map = enrichment.get('tips', {})
        for c in clusters:
            topic = c['topic']
            if topic in tips_map:
                c['tip'] = tips_map[topic].get('tip', '')
                c['pattern_note'] = tips_map[topic].get('pattern_note', '')
            elif not c.get('tip'):
                c['tip'] = f"Review all past questions on {topic} — appears consistently across papers."

        # Ensure we always have 10 predictions
        predictions = enrichment.get('predictions', [])
        if len(predictions) < 10:
            # Pad with generic predictions from HIGH clusters
            for c in clusters:
                if len(predictions) >= 10: break
                if c.get('questions'):
                    predictions.append({
                        'question': c['questions'][0],
                        'topic': c['topic'],
                        'confidence': c.get('importance', 'MEDIUM'),
                        'reason': f"Appeared {c['frequency']} times across papers",
                        'likely_position': c['question_positions'][0] if c.get('question_positions') else 'Q2A',
                        'likely_marks': c['marks_each_time'][0] if c.get('marks_each_time') else 10,
                        'frequency': c['frequency']
                    })

        enrichment['predictions'] = predictions[:10]
        return enrichment

    except Exception as e:
        # Fallback predictions from clusters
        fallback_predictions = []
        for c in clusters[:10]:
            if c.get('questions'):
                fallback_predictions.append({
                    'question': c['questions'][0],
                    'topic': c['topic'],
                    'confidence': c.get('importance', 'MEDIUM'),
                    'reason': f"Repeated {c['frequency']} times — high chance of repeat",
                    'likely_position': c['question_positions'][0] if c.get('question_positions') else 'Q2A',
                    'likely_marks': c['marks_each_time'][0] if c.get('marks_each_time') else 10,
                    'frequency': c['frequency']
                })
        return {
            'predictions': fallback_predictions,
            'paper_pattern': {},
            'study_plan': {
                'strategy': f'Focus on HIGH priority topics for {subject.upper()}. Prioritize topics that appeared 3+ times.',
                'days': [], 'golden_topics': [], 'dont_skip': []
            }
        }

def parse_paper_name(filename):
    MONTH_MAP = {
        'jan':('January',1),'feb':('February',2),'mar':('March',3),
        'apr':('April',4),'may':('May',5),'jun':('June',6),
        'jul':('July',7),'aug':('August',8),'sep':('September',9),
        'oct':('October',10),'nov':('November',11),'dec':('December',12)
    }
    parts = filename.lower().replace('-','_').split('_')
    year = next((int(p) for p in parts if re.match(r'20\d\d$', p)), None)
    mk = next((p[:3] for p in parts if p[:3] in MONTH_MAP), None)
    mn, mnum = MONTH_MAP.get(mk, ('',0))
    sk = year*100+mnum if year else 0
    if year and mn: disp = f"{mn}_{year}"
    elif year: disp = f"Paper_{year}"
    else: disp = f"Paper_{filename[:10]}"
    return disp, sk


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
                {'id': 'ai', 'name': 'Artificial Intelligence'},
                {'id': 'ml', 'name': 'Machine Learning'},
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

                SUBJECT_SIGNALS = {
                    'mc': ['mobile','gsm','gprs','handover','bluetooth','cellular','wireless'],
                    'os': ['process','scheduling','deadlock','semaphore','paging','memory management'],
                    'spcc': ['assembler','compiler','macro','parser','loader','lexical'],
                    'dbms': ['normalization','sql','transaction','relational','query'],
                    'ai': ['search','heuristic','minimax','planning','inference','knowledge'],
                    'ml': ['regression','classification','clustering','neural','learning rate'],
                }
                signals = SUBJECT_SIGNALS.get(subject, [])
                if signals and len(text) > 200:
                    text_lower = text.lower()
                    matches = sum(1 for s in signals if s in text_lower)
                    if matches < 2:
                        ocr_used.append(f"⚠ {paper_name} may be wrong subject")
                        continue
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
                gc.collect()

        if len(all_papers_text) < 2:
            return jsonify({'error': 'Need at least 2 readable papers.'}), 400

        # Step 2: Extract structured questions from each paper (1 Groq call per paper)
        all_questions = []
        sorted_papers = sorted(all_papers_text.items(), key=lambda x: x[1][1], reverse=True)
        for paper_name, (text, _) in sorted_papers:
            questions = extract_questions_as_json(paper_name, clean_pdf_text(text), client)
            if questions:
                all_questions.extend(questions)
            else:
                lines = clean_pdf_text(text).split('\n')
                for line in lines:
                    if re.search(r'Q\d+|explain|describe|define', line, re.I) and len(line) > 20:
                        all_questions.append({'id': f"{paper_name}_{len(all_questions)}", 'paper': paper_name, 'position': '', 'marks': 0, 'question': line[:300]})
            time.sleep(1)
            gc.collect()

        # Step 3: Python clustering — deterministic
        clusters = cluster_questions_python(all_questions, subject, len(all_papers_text))

        # Step 3b: Seed missing known topics
        topic_list = SUBJECT_TOPICS.get(subject, SUBJECT_TOPICS['general'])
        existing_topics = {c['topic'] for c in clusters}
        for topic in topic_list:
            found_questions = []
            for q in all_questions:
                if match_topic(q['question'], topic):
                    found_questions.append(q)
            if found_questions and topic not in existing_topics:
                papers = list({q['paper'] for q in found_questions})
                if len(papers) >= 1:
                    clusters.append({
                        'topic': topic,
                        'frequency': len(papers),
                        'importance': 'HIGH' if len(papers) >= 3 else 'MEDIUM' if len(papers) >= 2 else 'LOW',
                        'questions': [q['question'] for q in found_questions[:4]],
                        'papers': papers[:4],
                        'question_positions': [q.get('position','') for q in found_questions[:4]],
                        'marks_each_time': [q.get('marks',0) for q in found_questions[:4]],
                        'consistent_position': False,
                        'consistent_marks': False,
                        'pattern_note': '',
                        'tip': '',
                        'keywords': []
                    })
        clusters.sort(key=lambda x: (-x['frequency'], ['LOW','MEDIUM','HIGH'].index(x.get('importance','LOW'))))
        clusters = clusters[:15]

        # Step 4: Enrich — now passes all_questions for better predictions
        enrichment = enrich_with_groq(clusters, all_questions, user_name, university, subject, client)
        predictions = enrichment.get('predictions', [])
        study_plan = enrichment.get('study_plan', {})
        paper_pattern = enrichment.get('paper_pattern', {})

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
            'total_questions': len(all_questions),
            'clusters': len(clusters),
            'high_priority': len([c for c in clusters if c.get('importance')=='HIGH']),
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
    university = data.get('university', '')
    subject = data.get('subject', '')
    today = datetime.now().strftime('%Y%m%d')

    W = 60  # width
    DIV  = "═" * W
    DIV2 = "─" * W

    def header(title):
        pad = W - len(title) - 2
        left = pad // 2
        right = pad - left
        return [f"╔{DIV}╗", f"║{' ' * left}{title}{' ' * right}║", f"╚{DIV}╝"]

    def section(title):
        return ["", DIV2, f"  {title}", DIV2]

    def box(title):
        return [f"┌─ {title} {'─' * max(0, W - len(title) - 4)}┐"]

    lines = header("ASHLYSIS — EXAM INTELLIGENCE REPORT")
    lines += [
        "",
        f"  Student    : {user_name}",
        f"  University : {university}",
        f"  Subject    : {subject}",
        f"  Generated  : {datetime.now().strftime('%d %b %Y  %I:%M %p')}",
        "",
    ]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 1 — SUMMARY
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    lines += section("SUMMARY")
    lines += [
        f"  Papers Analysed    : {stats.get('papers', 0)}",
        f"  Questions Scanned  : {stats.get('total_questions', 0)}",
        f"  Repeat Clusters    : {stats.get('clusters', 0)}",
        f"  HIGH Priority      : {stats.get('high_priority', 0)}",
        f"  Predictions        : {len(predictions)}",
    ]
    if paper_pattern:
        lines += [
            "",
            f"  Paper Format  : {paper_pattern.get('total_marks', 80)} marks  |  {paper_pattern.get('duration', '3 hours')}",
            f"  Q1 Structure  : {paper_pattern.get('compulsory_question', '')}",
            f"  Q2-Q6         : {paper_pattern.get('optional_questions', '')}",
        ]
        if paper_pattern.get('key_insight'):
            lines += [f"  Key Insight   : {paper_pattern['key_insight']}"]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 2 — MOST REPEATED QUESTIONS
    # (clean numbered list — ready to paste into ChatGPT/Claude)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    lines += section("MOST REPEATED QUESTIONS")
    lines += [
        "  [Paste this block into ChatGPT / Claude for answers]",
        "",
        f"  I am studying {subject} for {university}.",
        f"  Please answer the following frequently-repeated exam questions:",
        "",
    ]

    q_num = 1
    for c in clusters:
        if c.get('frequency', 0) < 2:
            continue
        best_q = c.get('questions', [''])[0]
        marks_list = c.get('marks_each_time', [])
        pos_list   = c.get('question_positions', [])
        m_str = f"[{marks_list[0]}m]" if marks_list else ""
        p_str = f"[{pos_list[0]}]" if pos_list else ""
        lines.append(f"  {q_num}. {best_q.strip()} {m_str} {p_str}".rstrip())
        q_num += 1

    lines += ["", f"  (Total: {q_num - 1} repeated questions found across {stats.get('papers', 0)} papers)"]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 3 — PATTERN ANALYSIS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    lines += section("PATTERN ANALYSIS")

    for i, c in enumerate(clusters, 1):
        freq     = c.get('frequency', 0)
        imp      = c.get('importance', '')
        topic    = c.get('topic', '')
        papers   = c.get('papers', [])
        pos      = c.get('question_positions', [])
        marks    = c.get('marks_each_time', [])
        qs       = c.get('questions', [])
        tip      = c.get('tip', '')
        pattern  = c.get('pattern_note', '')
        con_pos  = "✓ ALWAYS SAME POSITION" if c.get('consistent_position') else ""
        con_mrk  = "✓ ALWAYS SAME MARKS"    if c.get('consistent_marks')    else ""

        lines += [
            "",
            f"  {i}. [{imp}] {topic}",
            f"     Frequency : appeared {freq}× across papers",
            f"     Papers    : {', '.join(papers)}",
        ]
        if pos:
            lines.append(f"     Positions : {', '.join(str(p) for p in pos)}  {con_pos}".rstrip())
        if marks:
            lines.append(f"     Marks     : {', '.join(str(m) for m in marks)}  {con_mrk}".rstrip())
        if pattern:
            lines.append(f"     Pattern   : {pattern}")
        if tip:
            lines.append(f"     Tip       : {tip}")

        # All versions of the question
        if qs:
            lines.append(f"     Versions seen:")
            for qi, (q, paper) in enumerate(zip(qs, papers), 1):
                lines.append(f"       {qi}. [{paper}] {q[:180]}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 4 — PREDICTED QUESTIONS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    lines += section("PREDICTED QUESTIONS FOR NEXT EXAM")
    lines += [
        "  [Based on repeat patterns — high probability questions]",
        "",
    ]
    for i, p in enumerate(predictions, 1):
        conf  = p.get('confidence', '')
        q     = p.get('question', '')
        pos   = p.get('likely_position', '?')
        mrks  = p.get('likely_marks', '?')
        topic = p.get('topic', '')
        freq  = p.get('frequency', 0)
        why   = p.get('reason', '')
        lines += [
            f"  {i}. [{conf}] {q}",
            f"     → Position: {pos}  |  Marks: {mrks}m  |  Topic: {topic}  |  Seen: {freq}×",
            f"     → Why: {why}",
            "",
        ]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SECTION 5 — 7-DAY STUDY PLAN
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    lines += section("7-DAY STUDY PLAN")
    if study_plan:
        lines += [
            f"  Strategy    : {study_plan.get('strategy', '')}",
            f"  Golden Topics : {', '.join(study_plan.get('golden_topics', []))}",
            f"  Never Skip    : {', '.join(study_plan.get('dont_skip', []))}",
        ]
        for day in study_plan.get('days', []):
            lines += [
                "",
                f"  DAY {day['day']} [{day['priority']}] — {day['focus']}  ({day['hours']} hrs)",
            ]
            for t in day.get('tasks', []):
                lines.append(f"    ✓ {t}")

    lines += [
        "",
        DIV,
        f"  ASHLYSIS — Report for {user_name}  |  {subject}  |  {datetime.now().strftime('%d %b %Y')}",
        "  AN ASHMIT SINGH PRODUCTION",
        DIV,
    ]

    buf = io.BytesIO("\n".join(lines).encode('utf-8'))
    buf.seek(0)
    return send_file(
        buf, mimetype='text/plain', as_attachment=True,
        download_name=f'ashlysis_{user_name.replace(" ", "_")}_{subject}_{today}.txt'
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
