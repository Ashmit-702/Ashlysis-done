"""Pure-Python helpers for ASHLYSIS (no network, no Flask) so they can be unit-tested."""
import hashlib
import re
from collections import Counter

SUBJECT_NAMES = {
    'spcc': 'System Programming & Compiler Construction', 'dbms': 'Database Management Systems',
    'os': 'Operating Systems', 'cn': 'Computer Networks', 'mc': 'Mobile Computing',
    'dsa': 'Data Structures & Algorithms', 'ai': 'Artificial Intelligence',
    'ml': 'Machine Learning', 'general': 'General / Unidentified Subject',
}

# title phrases (strong signal) and body keywords (weak signal)
_TITLES = {
    'ml': ['machine learning'], 'ai': ['artificial intelligence'],
    'spcc': ['system programming', 'compiler construction'], 'dbms': ['database management', 'advanced database'],
    'os': ['operating system'], 'cn': ['computer network'], 'mc': ['mobile computing'],
    'dsa': ['data structure'],
}
_KEYWORDS = {
    'ml': ['supervised', 'unsupervised', 'regression', 'classification', 'svm', 'support vector', 'decision tree',
           'clustering', 'k-means', 'bias', 'variance', 'neural network', 'overfitting', 'gradient descent',
           'pca', 'naive bayes', 'ensemble', 'random forest', 'reinforcement', 'regularization', 'kernel'],
    'ai': ['heuristic', 'a* algorithm', 'knowledge representation', 'expert system', 'wumpus', 'peas',
           'intelligent agent', 'first order logic', 'propositional', 'minimax', 'alpha-beta', 'fuzzy'],
    'spcc': ['assembler', 'macro', 'loader', 'linker', 'lexical analy', 'parser', 'yacc', 'intermediate code',
             'code generation', 'compiler', 'two pass', 'symbol table'],
    'dbms': ['sql', 'normalization', 'transaction', 'er diagram', 'relational', 'concurrency', 'acid', 'schema',
             'functional dependency', 'query'],
    'os': ['scheduling', 'deadlock', 'paging', 'semaphore', 'memory management', 'file system', 'thread',
           'page replacement', 'process'],
    'cn': ['osi', 'tcp', 'routing', 'subnet', 'ip address', 'data link', 'congestion', 'ethernet'],
    'mc': ['gsm', 'cellular', 'handoff', 'wireless', 'mobile ip', 'manet', 'cdma'],
    'dsa': ['linked list', 'stack', 'queue', 'binary tree', 'graph', 'sorting', 'hashing', 'complexity'],
}
_MONTHS = {'jan': 'January', 'feb': 'February', 'mar': 'March', 'apr': 'April', 'may': 'May', 'jun': 'June',
           'jul': 'July', 'aug': 'August', 'sep': 'September', 'oct': 'October', 'nov': 'November', 'dec': 'December'}
_MONTH_NUM = {v: i + 1 for i, v in enumerate(_MONTHS.values())}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _norm_words(text):
    return re.findall(r'[a-z0-9]+', (text or '').lower())


def shingles(text, n=4):
    w = _norm_words(text)
    return {' '.join(w[i:i + n]) for i in range(max(len(w) - n + 1, 0))}


def content_similarity(a, b):
    """Jaccard similarity of 4-word shingles (0..1)."""
    sa, sb = shingles(a), shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def detect_metadata(text, filename=''):
    """Returns dict: subject (key|None), subject_confidence, subject_code, year, month, period, session."""
    head = (text or '')[:1200].lower()
    low = (text or '').lower()
    scores = {}
    for key, titles in _TITLES.items():
        s = 0
        for t in titles:
            if t in head:
                s += 20
            elif t in low:
                s += 6
        for kw in _KEYWORDS.get(key, []):
            s += min(low.count(kw), 3)
        scores[key] = s
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    best, second = ranked[0], ranked[1]
    subject, conf = None, 'none'
    if best[1] >= 6:
        subject = best[0]
        conf = 'high' if best[1] >= 15 and best[1] >= 1.8 * max(second[1], 1) else 'medium'
        if best[1] < 10 or best[1] < 1.3 * max(second[1], 1):
            conf = 'low'

    code = None
    m = re.search(r'(?:subject|course)\s*code\s*[:\-]?\s*([A-Z]{2,6}\s?\d{3,6}[A-Z]?)', text or '', re.I)
    if m:
        code = re.sub(r'\s+', '', m.group(1)).upper()
    else:
        m = re.search(r'\b((?:CSC|ITC|CSDLO|ITDLO|CSL|ITL|AIC|AIDLO|CSM)\d{3,5}[A-Z]?)\b', text or '')
        if m:
            code = m.group(1)

    month, year = None, None
    src = (text or '')[:1500] + ' ' + filename.replace('_', ' ').replace('-', ' ')
    mon = r'(?<![a-z])(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*'
    m = re.search(mon + r'\.?[\s,/_-]*(?:[a-z]*[\s,/_-]*)?(20\d\d)(?!\d)', src, re.I)
    m_rev = re.search(r'(?<!\d)(20\d\d)[\s,/_-]*' + mon + r'(?![a-z])', src, re.I)
    if m:
        month, year = _MONTHS[m.group(1).lower()[:3]], int(m.group(2))
    elif m_rev:
        month, year = _MONTHS[m_rev.group(2).lower()[:3]], int(m_rev.group(1))
    else:
        m = re.search(r'\b(20\d\d)\b', src)
        if m:
            year = int(m.group(1))
        m2 = re.search(mon + r'(?![a-z])', filename, re.I)
        if m2:
            month = _MONTHS[m2.group(1).lower()[:3]]
    period = f"{month} {year}" if month and year else (str(year) if year else None)
    session = None
    if month in ('May', 'June'):
        session = 'May/June'
    elif month in ('November', 'December'):
        session = 'Nov/Dec'
    elif month:
        session = month
    sort_key = year * 100 + _MONTH_NUM.get(month, 0) if year else 0
    return {'subject': subject, 'subject_confidence': conf, 'subject_scores': {k: v for k, v in ranked[:3]},
            'subject_code': code, 'year': year, 'month': month, 'period': period,
            'session': session, 'sort_key': sort_key}


# ───────── local question parser (used when no AI provider is usable) ─────────
_INSTR = re.compile(r'^(attempt|solve|answer|note|n\.?b|total marks|time\s*:|duration|assume|figures to the right|'
                    r'question no|all questions|any (?:one|two|three|four))', re.I)
_MARKS = re.compile(r'(?:\[\s*(\d{1,2})\s*(?:marks?|m)?\s*\]|\(\s*(\d{1,2})\s*(?:marks?|m)\s*\)|'
                    r'\b(\d{1,2})\s*marks?\b)', re.I)
_QSTART = re.compile(r'^\s*(?:Q\.?\s*|Question\s*)(\d{1,2})\s*[.):\-]?\s*(?:\(?([A-Da-d])[.)]\s+)?(.*)$', re.I)
_SUB = re.compile(r'^\s*\(?([A-Da-d])[.)]\s+(.*)$')


def local_parse_questions(text, paper):
    """Rule-based fallback parser: Q1 a) / Q2 A) / Q3 (B) style papers. Never invents text."""
    qs, cur, main = [], None, None

    def flush():
        nonlocal cur
        if cur and len(cur['question'].strip()) >= 12:
            mk = _MARKS.search(cur['question'])
            if mk:
                cur['marks'] = int(next(g for g in mk.groups() if g))
                cur['question'] = _MARKS.sub('', cur['question']).strip()
            cur['question'] = re.sub(r'\s+', ' ', cur['question']).strip(' -–:')
            qs.append(cur)
        cur = None

    for line in (text or '').split('\n'):
        line = line.strip()
        if not line:
            continue
        m = _QSTART.match(line)
        if m and not (m.group(3) and m.group(3)[:1].islower() and not m.group(2)):
            flush()
            main = int(m.group(1))
            sub, rest = m.group(2), (m.group(3) or '').strip()
            if not sub and _INSTR.match(rest):
                rest = ''
            pos = f"Q{main}{(sub.lower() if main == 1 else sub.upper()) if sub else ''}"
            cur = {'position': pos, 'marks': 0, 'question': rest, 'source': 'local'}
            continue
        s = _SUB.match(line)
        if s and main is not None:
            flush()
            sub = s.group(1)
            cur = {'position': f"Q{main}{sub.lower() if main == 1 else sub.upper()}", 'marks': 0,
                   'question': s.group(2), 'source': 'local'}
            continue
        if cur is not None and not _INSTR.match(line):
            cur['question'] += ' ' + line
    flush()
    seen, out = set(), []
    for q in qs:
        qid = f"{paper}_{q['position']}"
        if qid in seen:
            continue
        seen.add(qid)
        q.update({'id': qid, 'paper': paper})
        if q['position'].upper().startswith('Q1') and not q['marks']:
            q['marks'] = 0
        out.append(q)
    return out


def question_supported_by_text(question, source_text, threshold=0.6):
    """Guard against model-invented questions: most content words must exist in the source text."""
    src = set(_norm_words(source_text))
    words = [w for w in _norm_words(question) if len(w) > 3]
    if not words:
        return False
    return sum(w in src for w in words) / len(words) >= threshold


def importance_for(freq, total_unique):
    ratio = freq / max(total_unique, 1)
    if freq >= 3 and ratio >= 0.5:
        return 'HIGH'
    if freq >= 2 and ratio >= 0.33:
        return 'MEDIUM'
    return 'LOW'


def marks_distribution(questions):
    c = Counter(q.get('marks', 0) for q in questions if q.get('marks', 0) > 0)
    return dict(sorted(c.items()))
