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

PAPER_FORMAT = "Q1a/b/c/d = 5 marks each. Q2A/B through Q6A/B = 10 marks each. Total 80 marks, 3 hours."

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
        "Five state process model suspended states",
        "Resource Allocation Graph deadlock"
    ],
    "dbms": [
        "Normalization 1NF 2NF 3NF BCNF",
        "SQL queries joins",
        "ER diagram entity relationship",
        "Transaction ACID properties",
        "Concurrency control",
        "Relational algebra",
        "Indexing B-tree",
        "Recovery techniques"
    ],
    "cn": [
        "OSI model seven layers",
        "TCP IP protocol suite",
        "Routing algorithms distance vector link state",
        "Congestion control",
        "Error detection correction CRC",
        "Medium access control CSMA",
        "Socket programming",
        "Network security cryptography"
    ],
    "dsa": [
        "Sorting algorithms time complexity quicksort mergesort",
        "Tree traversal BST binary",
        "Graph BFS DFS traversal",
        "Dynamic programming",
        "Hashing collision",
        "Stack queue linked list",
        "Heap priority queue",
        "Divide and conquer"
    ],
    "ai": [
        "Hill Climbing algorithm local search problems",
        "A star search heuristic informed uninformed",
        "Alpha Beta pruning minimax game tree",
        "Partial order planning hierarchical STRIPS",
        "PEAS descriptors task environment agent",
        "Forward chaining backward chaining inference",
        "First Order Logic FOL resolution CNF",
        "Genetic algorithms evolutionary computation",
        "Reinforcement learning reward policy",
        "Wumpus world environment",
        "Learning agent architecture",
        "Bayesian belief network probabilistic"
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
        if len(line) > 20 and hex_chars / max(len(line.replace(' ', '')), 1) > 0.75: continue
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
            pix = page.get_pixmap(matrix=fitz.Matrix(120 / 72, 120 / 72))
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

# ── STEP 1: EXTRACT QUESTIONS AS JSON ──
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

# ── STEP 2: TF-IDF CLUSTERING ──
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
    return [t for t in text.split() if t not in STOPWORDS and len(t) > 2]

def cosine_sim(v1, v2):
    common = set(v1) & set(v2)
    if not common: return 0.0
    dot = sum(v1[k] * v2[k] for k in common)
    m1 = math.sqrt(sum(x ** 2 for x in v1.values()))
    m2 = math.sqrt(sum(x ** 2 for x in v2.values()))
    if m1 == 0 or m2 == 0: return 0.0
    return dot / (m1 * m2)

def build_tfidf(questions):
    N = len(questions)
    tokenized = [tokenize(q['question']) for q in questions]
    df = Counter()
    for tokens in tokenized:
        for t in set(tokens): df[t] += 1
    idf = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in df}
    vectors = []
    for tokens in tokenized:
        tf = Counter(tokens)
        total = max(len(tokens), 1)
        vectors.append({t: (tf[t] / total) * idf.get(t, 1) for t in tf})
    return vectors

def match_topic(question_text, topic_keywords):
    q_lower = question_text.lower()
    topic_words = tokenize(topic_keywords)
    if not topic_words: return False
    matches = sum(1 for w in topic_words if w in q_lower)
    threshold = max(2, math.ceil(len(topic_words) * 0.4))
    return matches >= threshold

def cluster_questions_python(all_questions, subject, num_papers):
    topic_list = SUBJECT_TOPICS.get(subject, SUBJECT_TOPICS['general'])
    if len(all_questions) < 2: return []

    vectors = build_tfidf(all_questions)
    topic_assignments = {}
    topic_clusters = defaultdict(list)

    for i, q in enumerate(all_questions):
        best_topic, best_score = None, 0
        for topic in topic_list:
            if match_topic(q['question'], topic):
                topic_vec = {word: 1.0 for word in tokenize(topic)}
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
            for j in range(i + 1, len(unmatched)):
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
            if paper not in paper_best or len(q['question']) > len(paper_best[paper]['question']):
                paper_best[paper] = q
        unique_questions = list(paper_best.values())
        papers = [q['paper'] for q in unique_questions]
        if len(papers) < 2: continue
        freq = len(papers)
        importance = 'HIGH' if freq >= 3 else 'MEDIUM' if freq == 2 else 'LOW'
        marks = [q.get('marks', 0) for q in unique_questions]
        marks_nonzero = [m for m in marks if m > 0]
        positions = [q.get('position', '') for q in unique_questions]
        q_nums = [re.match(r'Q\d+', p, re.I).group() if re.match(r'Q\d+', p, re.I) else p for p in positions]
        final_clusters.append({
            'topic': topic, 'frequency': freq, 'importance': importance,
            'questions': [q['question'] for q in unique_questions],
            'papers': papers, 'question_positions': positions,
            'marks_each_time': marks_nonzero,
            'consistent_position': len(set(q_nums)) == 1 if len(q_nums) > 1 else False,
            'consistent_marks': len(set(marks_nonzero)) == 1 if len(marks_nonzero) > 1 else False,
            'pattern_note': '', 'tip': '', 'keywords': []
        })

    final_clusters.sort(key=lambda x: (-x['frequency'], ['LOW','MEDIUM','HIGH'].index(x.get('importance','LOW'))))

    # Global dedup — same question position only in ONE cluster
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
        if cluster['frequency'] < 2: continue
        f = cluster['frequency']
        cluster['importance'] = 'HIGH' if f >= 3 else 'MEDIUM' if f == 2 else 'LOW'
        deduped.append(cluster)

    return deduped[:15]

# ── PAPER STRATEGY — pure Python, zero API calls ──

NUMERICAL_SIGNALS = [
    r'\d{3,}',                          # 3+ digit numbers (cylinder 345, track 80)
    r'\d+\s*(kb|mb|gb|ns|ms|ms)',       # units
    r'calculate|compute|find|determine|solve|evaluate',
    r'gantt chart|turnaround|waiting time|hit ratio|miss ratio',
    r'page fault|frame size|time quantum',
    r'cylinder|track|seek|head',         # disk scheduling
    r'arrive|burst|priority|quantum',    # process scheduling
    r'lru|fifo|optimal.*page',           # page replacement
    r'fcfs|sjf|round robin|rr\b',
    r'first fit|best fit|worst fit',
    r'bandwidth|throughput|utilization',
    r'\b\d+\s*[×x\*]\s*\d+',           # multiplication
    r'checksum|crc|hamming',
]

def classify_question(question_text):
    q = question_text.lower()
    for pattern in NUMERICAL_SIGNALS:
        if re.search(pattern, q, re.I):
            return 'numerical'
    return 'theory'

# Subject-specific exam hall strategy
SUBJECT_STRATEGY = {
    "os": {
        "approach": "Start with Q1 (all parts — process states, scheduling criteria are Q1 staples). Then pick Disk Scheduling (numerical — solve first, guaranteed marks). Follow with Process Scheduling (Gantt chart). Leave Deadlock for last — most explanation heavy.",
        "time_split": {"Q1": 30, "Q2-Q6 each": 25, "buffer": 5},
        "first_attempt": "Disk Scheduling → Process Scheduling → Semaphore/Deadlock",
        "avoid_first": "Memory management (paging calculations take time)",
        "quick_win": "Q1 short notes — 20 marks in 30 minutes if you know the topics"
    },
    "mc": {
        "approach": "MC is almost entirely theory + diagrams. No numerical. Start Q1 (architecture diagrams are Q1 staples). Then pick Mobile Packet Delivery (diagram-heavy but predictable). GSM/GPRS architecture diagrams are the fastest marks — draw once, explain labels.",
        "time_split": {"Q1": 25, "Q2-Q6 each": 27, "buffer": 3},
        "first_attempt": "GPRS/GSM Architecture → Mobile Terminated Call → Packet Delivery",
        "avoid_first": "Snooping TCP (requires comparison table — takes time)",
        "quick_win": "Draw the architecture diagram first, then write labels — saves 5 minutes per question"
    },
    "spcc": {
        "approach": "Mix of theory and numerical (assembler pass tables, parse tables). Start Q1 (definitions, short notes). Then Two-pass Assembler (draw flowchart — fast marks). Parser questions are numerical — save for when you're warmed up.",
        "time_split": {"Q1": 25, "Q2-Q6 each": 27, "buffer": 3},
        "first_attempt": "Two-pass Assembler → Loader → Compiler Phases",
        "avoid_first": "SLR/LL(1) parse tables (calculation-heavy, time consuming)",
        "quick_win": "Assembler directives list + Pass 1 vs Pass 2 table — 10 marks in 10 minutes"
    },
    "dbms": {
        "approach": "Normalization is the king topic — always appears, always 10 marks. Start Q1 (ER diagram, short notes). Normalization first from Q2-Q6. SQL queries are fast if you know them.",
        "time_split": {"Q1": 25, "Q2-Q6 each": 27, "buffer": 3},
        "first_attempt": "Normalization → ER Diagram → SQL Queries",
        "avoid_first": "Relational algebra (notation-heavy, easy to make errors)",
        "quick_win": "Write 1NF → 2NF → 3NF → BCNF with example table — most marks in normalization"
    },
    "cn": {
        "approach": "OSI model is the backbone — always in Q1. Start Q1 (OSI layers, TCP/IP). Then routing algorithms (Dijkstra is numerical but structured). Error detection is mix of theory + CRC calculation.",
        "time_split": {"Q1": 25, "Q2-Q6 each": 27, "buffer": 3},
        "first_attempt": "OSI Model → TCP/IP → Routing Algorithms",
        "avoid_first": "Congestion control (requires understanding of multiple mechanisms)",
        "quick_win": "OSI 7 layers with functions table — 5 marks in 5 minutes"
    },
    "dsa": {
        "approach": "Almost all numerical — tracing algorithms, drawing trees. Start with sorting (draw array, show passes). Graph BFS/DFS is fast to trace. Dynamic programming is hardest — save for last.",
        "time_split": {"Q1": 20, "Q2-Q6 each": 28, "buffer": 4},
        "first_attempt": "Sorting trace → Graph BFS/DFS → Tree operations",
        "avoid_first": "Dynamic programming (time-consuming for complex problems)",
        "quick_win": "Quick sort partition trace — show pivot selection and array state after each pass"
    },
    "ai": {
        "approach": "Mix of theory and algorithm tracing (A*, minimax). Start Q1 (PEAS, agent types — pure theory). Then Hill Climbing (algorithm trace is structured). Alpha-Beta pruning is fast to draw on paper.",
        "time_split": {"Q1": 25, "Q2-Q6 each": 27, "buffer": 3},
        "first_attempt": "Hill Climbing → Planning → Alpha-Beta Pruning",
        "avoid_first": "FOL/Resolution (notation-heavy, easy to make errors)",
        "quick_win": "Alpha-Beta pruning on a given tree — draw tree, mark α β values"
    },
    "ml": {
        "approach": "Theory-heavy with some calculations (regression, confusion matrix). Start Q1 (definitions). Linear/Logistic Regression next — structured derivation. SVM is diagram-heavy.",
        "time_split": {"Q1": 25, "Q2-Q6 each": 27, "buffer": 3},
        "first_attempt": "Regression → Decision Tree → Neural Network",
        "avoid_first": "SVM (kernel math is complex under time pressure)",
        "quick_win": "Decision tree with example dataset — draw tree, show splits"
    },
    "general": {
        "approach": "Attempt Q1 first (compulsory, 20 marks). Then pick the 3 questions from Q2-Q6 where you have most knowledge. Always attempt questions where you can draw a diagram — diagrams get marks even if explanation is incomplete.",
        "time_split": {"Q1": 25, "Q2-Q6 each": 27, "buffer": 3},
        "first_attempt": "Topics you know best → diagram-heavy questions → theory questions",
        "avoid_first": "Questions requiring long calculations unless you are very confident",
        "quick_win": "Any question with a diagram — draw first, label it, then explain in 4-5 points"
    }
}

def classify_all_questions(all_questions):
    """Classify every extracted question as theory or numerical"""
    theory_count = 0
    numerical_count = 0
    for q in all_questions:
        if classify_question(q.get('question', '')) == 'numerical':
            numerical_count += 1
        else:
            theory_count += 1
    total = max(theory_count + numerical_count, 1)
    return {
        'theory': theory_count,
        'numerical': numerical_count,
        'theory_pct': round(theory_count / total * 100),
        'numerical_pct': round(numerical_count / total * 100),
        'total': total
    }

def build_paper_strategy(clusters, all_questions, subject):
    """Build complete paper strategy — pure Python, zero API calls"""
    qt = classify_all_questions(all_questions)
    strategy = SUBJECT_STRATEGY.get(subject, SUBJECT_STRATEGY['general'])

    # Which clusters have numerical questions
    numerical_clusters = []
    theory_clusters = []
    for c in clusters:
        is_num = any(classify_question(q) == 'numerical' for q in c.get('questions', []))
        if is_num:
            numerical_clusters.append(c['topic'])
        else:
            theory_clusters.append(c['topic'])

    # Attempt order based on importance + type
    high_theory = [c['topic'] for c in clusters if c['importance'] == 'HIGH' and c['topic'] in theory_clusters]
    high_numerical = [c['topic'] for c in clusters if c['importance'] == 'HIGH' and c['topic'] in numerical_clusters]

    # Time allocation
    time_split = strategy['time_split']

    return {
        'question_types': qt,
        'numerical_clusters': numerical_clusters,
        'theory_clusters': theory_clusters,
        'approach': strategy['approach'],
        'time_split': time_split,
        'first_attempt': strategy['first_attempt'],
        'avoid_first': strategy['avoid_first'],
        'quick_win': strategy['quick_win'],
        'high_theory_topics': high_theory[:5],
        'high_numerical_topics': high_numerical[:5],
        'paper_format': {
            'total_marks': 80,
            'duration_mins': 180,
            'q1_marks': 20,
            'q1_parts': 4,
            'q1_marks_each': 5,
            'optional_questions': 5,
            'attempt_optional': 3,
            'optional_marks_each': 20,
            'marks_per_minute': round(80 / 180, 2)
        }
    }

# ── STEP 3: GROQ ENRICHMENT ──
# Returns: tips (applied to clusters) + predictions (10) + last_hour_prep (5 must-do)
def enrich_with_groq(clusters, user_name, subject, client):

    # Build fallback in case Groq fails — always works from clusters
    def build_fallback():
        preds = []
        for c in clusters[:10]:
            q = c['questions'][0] if c['questions'] else f"Explain {c['topic']} with suitable example."
            preds.append({
                "question": q,
                "topic": c['topic'],
                "confidence": c['importance'],
                "reason": f"Appeared in {c['frequency']} papers",
                "likely_position": c['question_positions'][0] if c['question_positions'] else "Q2A",
                "likely_marks": c['marks_each_time'][0] if c['marks_each_time'] else 10,
                "frequency": c['frequency']
            })
        must_do = []
        for i, c in enumerate(clusters[:5]):
            must_do.append({
                "rank": i + 1,
                "topic": c['topic'],
                "question": c['questions'][0][:200] if c['questions'] else f"Explain {c['topic']}.",
                "marks": c['marks_each_time'][0] if c['marks_each_time'] else 10,
                "position": c['question_positions'][0] if c['question_positions'] else "",
                "why": f"Appeared in {c['frequency']} papers",
                "quick_tip": "Draw a neat labeled diagram. Cover all key components. Give one real example."
            })
        return {
            'predictions': preds,
            'last_hour_prep': {
                'message': f"Stay calm and focused, {user_name}. You have prepared for this.",
                'passing_strategy': "Attempt all of Q1 (20 marks) + any 2 questions from Q2-Q6 (40 marks) = 60 marks. That's passing.",
                'must_do': must_do
            }
        }

    cluster_summary = json.dumps([{
        'topic': c['topic'],
        'frequency': c['frequency'],
        'importance': c['importance'],
        'papers': c['papers'],
        'sample_question': c['questions'][0][:200] if c['questions'] else '',
        'typical_marks': c['marks_each_time'][0] if c['marks_each_time'] else 10,
        'position': c['question_positions'][0] if c['question_positions'] else ''
    } for c in clusters], indent=2)[:6000]

    prompt = f"""You are an exam intelligence engine for {subject.upper()} engineering students.
Student: {user_name}

These are CONFIRMED repeat topics from past papers (do NOT change frequencies):
{cluster_summary}

Return ONLY this JSON — start with {{ end with }} — no markdown no explanation:
{{
  "tips": {{
    "EXACT_TOPIC_NAME_FROM_ABOVE": {{
      "tip": "specific practical tip for answering this in the exam",
      "pattern_note": "when and where this appears in the paper"
    }}
  }},
  "predictions": [
    {{
      "question": "Complete specific exam question text — write it like an actual exam question not a topic name",
      "topic": "topic name from clusters",
      "confidence": "HIGH",
      "reason": "Appeared in X of Y papers, always Z marks",
      "likely_position": "Q2A",
      "likely_marks": 10,
      "frequency": 3
    }}
  ],
  "last_hour_prep": {{
    "message": "One short genuine motivational line for {user_name} — not generic",
    "passing_strategy": "Attempt all of Q1 (20 marks) + any 2 from Q2-Q6 (40 marks) = 60 marks minimum passing",
    "must_do": [
      {{
        "rank": 1,
        "topic": "topic name",
        "question": "exact most likely question — copy or rephrase from sample_question",
        "marks": 10,
        "position": "Q2A",
        "why": "Appeared in X/Y papers always same marks",
        "quick_tip": "Name exactly what to include: draw diagram with X Y Z labels, mention A B C concepts"
      }}
    ]
  }}
}}

RULES — strictly follow:
- predictions: EXACTLY 10 items. Each must be a real exam question, not just a topic name.
- last_hour_prep.must_do: EXACTLY 5 items — top 5 highest frequency topics only.
- quick_tip: be specific — name actual concepts, diagram labels, algorithms to mention.
- Return ONLY valid JSON. Nothing before {{. Nothing after }}."""

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "Return ONLY valid JSON. Start with { end with }. No markdown. No text before or after."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=3500
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            start = raw.find('{')
            end = raw.rfind('}')
            if start == -1 or end == -1 or end <= start:
                raise ValueError("No JSON object found")
            raw = raw[start:end + 1]
            enrichment = json.loads(raw)

            # Apply tips to clusters
            tips_map = enrichment.get('tips', {})
            for c in clusters:
                if c['topic'] in tips_map:
                    c['tip'] = tips_map[c['topic']].get('tip', '')
                    c['pattern_note'] = tips_map[c['topic']].get('pattern_note', '')
                if not c.get('tip'):
                    c['tip'] = f"Review all past questions on this topic — appears consistently across papers."

            # ── FIX: validate predictions — never return empty ──
            predictions = enrichment.get('predictions', [])
            if not isinstance(predictions, list) or len(predictions) == 0:
                predictions = build_fallback()['predictions']
            # Ensure each prediction has all required fields
            cleaned_preds = []
            for p in predictions:
                if not isinstance(p, dict): continue
                q = p.get('question', '')
                if not q or len(q) < 10:
                    q = p.get('topic', 'Review this topic') + ' — explain with suitable example and diagram.'
                cleaned_preds.append({
                    'question': q,
                    'topic': str(p.get('topic', '')),
                    'confidence': str(p.get('confidence', 'MEDIUM')),
                    'reason': str(p.get('reason', f"Appeared in {p.get('frequency', 2)} papers")),
                    'likely_position': str(p.get('likely_position', 'Q2A')),
                    'likely_marks': int(p.get('likely_marks', 10)),
                    'frequency': int(p.get('frequency', 2))
                })
            enrichment['predictions'] = cleaned_preds if cleaned_preds else build_fallback()['predictions']

            # ── FIX: validate last_hour_prep — never return empty ──
            lhp = enrichment.get('last_hour_prep', {})
            if not isinstance(lhp, dict) or not lhp.get('must_do'):
                lhp = build_fallback()['last_hour_prep']
            else:
                cleaned_must_do = []
                for i, q in enumerate(lhp.get('must_do', [])):
                    if not isinstance(q, dict): continue
                    question = q.get('question', '')
                    if not question or len(question) < 10:
                        question = clusters[i]['questions'][0][:200] if i < len(clusters) and clusters[i]['questions'] else f"Explain {q.get('topic', '')}."
                    cleaned_must_do.append({
                        'rank': int(q.get('rank', i + 1)),
                        'topic': str(q.get('topic', '')),
                        'question': question,
                        'marks': int(q.get('marks', 10)),
                        'position': str(q.get('position', '')),
                        'why': str(q.get('why', 'Appears frequently in past papers')),
                        'quick_tip': str(q.get('quick_tip', 'Draw a neat diagram, cover all key components, give one example.'))
                    })
                lhp['must_do'] = cleaned_must_do if cleaned_must_do else build_fallback()['last_hour_prep']['must_do']
                if not lhp.get('message'):
                    lhp['message'] = f"Stay calm, {user_name}. You have prepared for this."
                if not lhp.get('passing_strategy'):
                    lhp['passing_strategy'] = "Attempt all of Q1 (20 marks) + any 2 from Q2-Q6 (40 marks) = 60 marks minimum."
            enrichment['last_hour_prep'] = lhp

            return enrichment

        except (json.JSONDecodeError, ValueError):
            if attempt < 2:
                time.sleep(5)
                continue
            break
        except Exception as e:
            err = str(e).lower()
            if 'rate' in err or '429' in err:
                if attempt < 2:
                    time.sleep(20)
                    continue
                raise Exception('AI rate limit. Wait 1 minute and try again.')
            raise e

    # All attempts failed — full Python fallback
    for c in clusters:
        if not c.get('tip'):
            c['tip'] = "Review all past questions on this topic — appears consistently across papers."
    return build_fallback()


def parse_paper_name(filename):
    MONTH_MAP = {
        'jan':('January',1),'feb':('February',2),'mar':('March',3),
        'apr':('April',4),'may':('May',5),'jun':('June',6),
        'jul':('July',7),'aug':('August',8),'sep':('September',9),
        'oct':('October',10),'nov':('November',11),'dec':('December',12)
    }
    parts = filename.lower().replace('-', '_').split('_')
    year = next((int(p) for p in parts if re.match(r'20\d\d$', p)), None)
    mk = next((p[:3] for p in parts if p[:3] in MONTH_MAP), None)
    mn, mnum = MONTH_MAP.get(mk, ('', 0))
    sk = year * 100 + mnum if year else 0
    if year and mn: disp = f"{mn}_{year}"
    elif year: disp = f"Paper_{year}"
    else: disp = f"Paper_{filename[:10]}"
    return disp, sk


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
    if len(files) > 6:
        return jsonify({'error': 'Maximum 6 papers for best results.'}), 400
    non_pdf = [f.filename for f in files if not f.filename.lower().endswith('.pdf')]
    if non_pdf:
        return jsonify({'error': f'Only PDFs supported. Remove: {", ".join(non_pdf)}'}), 400

    user_name = request.form.get('user_name', 'Student').strip() or 'Student'
    user_name = re.sub(r'[^a-zA-Z0-9 ]', '', user_name)[:30]
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
                        return jsonify({'error': f'Cannot read "{filename}". Convert at smallpdf.com and re-upload.'}), 400
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

        clusters = cluster_questions_python(all_questions, subject, len(all_papers_text))

        # Build paper strategy — pure Python, zero API calls
        paper_strategy = build_paper_strategy(clusters, all_questions, subject)

        enrichment = enrich_with_groq(clusters, user_name, subject, client)

        # ── GUARANTEED: predictions and last_hour_prep always present ──
        predictions = enrichment.get('predictions', [])
        last_hour_prep = enrichment.get('last_hour_prep', {})

        # Final safety net
        if not predictions:
            predictions = [{'question': c['questions'][0] if c['questions'] else c['topic'], 'topic': c['topic'], 'confidence': c['importance'], 'reason': f"Appeared {c['frequency']} times", 'likely_position': c['question_positions'][0] if c['question_positions'] else 'Q2A', 'likely_marks': c['marks_each_time'][0] if c['marks_each_time'] else 10, 'frequency': c['frequency']} for c in clusters[:10]]
        if not last_hour_prep.get('must_do'):
            last_hour_prep = {'message': f"Stay focused, {user_name}.", 'passing_strategy': "Attempt Q1 (20m) + any 2 from Q2-Q6 (40m) = 60m minimum.", 'must_do': [{'rank': i+1, 'topic': c['topic'], 'question': c['questions'][0][:200] if c['questions'] else c['topic'], 'marks': c['marks_each_time'][0] if c['marks_each_time'] else 10, 'position': c['question_positions'][0] if c['question_positions'] else '', 'why': f"Appeared {c['frequency']} times", 'quick_tip': 'Draw diagram, cover all components, give example.'} for i, c in enumerate(clusters[:5])]}

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        gc.collect()
        _analysis_lock.release()

    return jsonify({
        'clusters': clusters,
        'predictions': predictions,
        'last_hour_prep': last_hour_prep,
        'paper_strategy': paper_strategy,
        'user_name': user_name,
        'subject': subject.upper(),
        'ocr_used': ocr_used,
        'stats': {
            'papers': len(all_papers_text),
            'total_questions': len(all_questions),
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
    last_hour_prep = data.get('last_hour_prep', {})
    stats = data.get('stats', {})
    user_name = data.get('user_name', 'Student')
    subject = data.get('subject', '')
    today = datetime.now().strftime('%Y%m%d')
    D = "━" * 52

    lines = [
        "╔══════════════════════════════════════════════════╗",
        "║           ASHLYSIS — EXAM INTELLIGENCE           ║",
        "╚══════════════════════════════════════════════════╝", "",
        f"Student  : {user_name}",
        f"Subject  : {subject}",
        f"Date     : {datetime.now().strftime('%d %b %Y %I:%M %p')}",
        f"Papers   : {stats.get('papers', 0)}",
        f"Clusters : {stats.get('clusters', 0)}",
        f"HIGH     : {stats.get('high_priority', 0)}",
    ]
    
    lines += ["", D, "MOST REPEATED QUESTIONS", D]

    lines += [
       "Paste into ChatGPT / Claude for full answers:",
       "",
       f"I am a Mumbai University student studying {subject}.",
       "Please answer these frequently-repeated exam questions:",
       ""
    ]

    seen_questions = set()

    for i, c in enumerate(clusters[:15], 1):

        questions = c.get('questions', [])

        if not questions:
        continue

        main_q = questions[0].strip()

        if main_q in seen_questions:
        continue

        seen_questions.add(main_q)

        marks = ""

        if c.get('marks_each_time'):
        marks = f" [{c.get('marks_each_time')[0]}m]"

        pos = ""

        if c.get('question_positions'):
        pos = f" [{c.get('question_positions')[0]}]"

        lines.append(
        f"{i}. {main_q}{marks}{pos}"
       )

    lines.append("")
    lines.append(f"({min(len(clusters),15)} repeated questions from {stats.get('papers',0)} papers)")

    lines += ["", D, "PREDICTED QUESTIONS", D]
    for i, p in enumerate(predictions, 1):
        lines += [
            f"\n{i}. [{p.get('confidence')}] {p.get('question', '')[:200]}",
            f"   Pos: {p.get('likely_position', '?')} | Marks: {p.get('likely_marks', '?')} | {p.get('reason', '')}"
        ]

    lines += ["", D, "LAST HOUR PREP — MINIMUM PASSING (40%)", D]
    if last_hour_prep:
        lines += [
            f"\n{last_hour_prep.get('message', '')}",
            f"\nStrategy: {last_hour_prep.get('passing_strategy', '')}",
            "\nMUST DO — 5 questions:"
        ]
        for q in last_hour_prep.get('must_do', []):
            lines += [
                f"\n{q.get('rank')}. {q.get('topic')} [{q.get('marks', 10)}m] — {q.get('position', '')}",
                f"   Q: {q.get('question', '')[:200]}",
                f"   Why: {q.get('why', '')}",
                f"   Include: {q.get('quick_tip', '')}"
            ]

    lines += ["", D, f"ASHLYSIS — generated for {user_name}", D]
    buf = io.BytesIO("\n".join(lines).encode('utf-8'))
    buf.seek(0)
    return send_file(buf, mimetype='text/plain', as_attachment=True,
                     download_name=f'ashlysis_{user_name.replace(" ", "_")}_{subject}_{today}.txt')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
