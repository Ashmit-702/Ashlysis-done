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
from ashlysis_core import (sha256_file, content_similarity, detect_metadata, local_parse_questions,
                           question_supported_by_text, importance_for, marks_distribution, SUBJECT_NAMES)

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

# ── AI PROVIDERS (both optional): GROQ_API_KEY, GEMINI_API_KEY ──
# Groq : GROQ_API_KEY (+ GROQ_API_KEY_2..20 or GROQ_API_KEYS=a,b) — text + vision, same key.
# Gemini: GEMINI_API_KEY (+ GEMINI_API_KEY_2..20 or GEMINI_API_KEYS=a,b) — PDF understanding + text.
# Optional model overrides: GROQ_TEXT_MODEL, GROQ_VISION_MODEL, GEMINI_MODEL
import logging
log = logging.getLogger('ashlysis')

def _model_chain(env_name, defaults):
    chain = []
    for m in [os.environ.get(env_name, '').strip()] + defaults:
        if m and m not in chain:
            chain.append(m)
    return chain

# Retired by providers in 2026: llama-4-scout, llama-3.3-70b, gemini-2.5-*. Chains skip dead IDs automatically.
TEXT_MODELS = _model_chain('GROQ_TEXT_MODEL',
    ['openai/gpt-oss-120b', 'qwen/qwen3.6-27b', 'llama-3.3-70b-versatile'])
VISION_MODELS = _model_chain('GROQ_VISION_MODEL',
    ['qwen/qwen3.6-27b', 'qwen/qwen3.8-27b', 'meta-llama/llama-4-scout-17b-16e-instruct'])
GEMINI_MODELS = _model_chain('GEMINI_MODEL',
    ['gemini-3.8-flash', 'gemini-3-flash-preview', 'gemini-3.5-flash-lite'])

_key_index = {'i': 0}
_key_lock = threading.Lock()
_dead_keys = {}        # key -> reason (401/403): skipped for process lifetime
_cooldowns = {}        # (key, model) -> unix time when usable again (429)
_bad_models = set()    # models reported retired / unknown / no access
_no_reasoning = set()
_clients = {}

class GroqUnavailable(Exception):
    """An AI provider could not serve the request. code/status feed err_response()."""
    def __init__(self, code, message, details='', status=503, retryable=True):
        self.code, self.message, self.details = code, message, details
        self.status, self.retryable = status, retryable
        super().__init__(message)
ProviderError = GroqUnavailable

def _collect_keys(base, plural):
    raw = [os.environ.get(base, ''), os.environ.get(plural, '')]
    raw += [os.environ.get(f'{base}_{n}', '') for n in range(1, 21)]
    keys = []
    for item in raw:
        for k in item.split(','):
            k = k.strip().strip('"\'')
            if k and k not in keys:
                keys.append(k)
    return keys

def get_groq_keys():   return _collect_keys('GROQ_API_KEY', 'GROQ_API_KEYS')
def get_gemini_keys(): return _collect_keys('GEMINI_API_KEY', 'GEMINI_API_KEYS')

def _redact(s):
    s = str(s)
    for k in get_groq_keys() + get_gemini_keys():
        s = s.replace(k, '[redacted]')
    return re.sub(r'(gsk_|AIza)[A-Za-z0-9_\-]{8,}', '[redacted]', s)

def provider_status():
    """Safe to show in the UI: counts and states only, never key material."""
    gk, mk = get_groq_keys(), get_gemini_keys()
    return {
        'groq': {'configured': bool(gk), 'keys': len(gk),
                 'invalid_keys': len([k for k in gk if k in _dead_keys]), 'vision_models': VISION_MODELS[:1]},
        'gemini': {'configured': bool(mk), 'keys': len(mk),
                   'invalid_keys': len([k for k in mk if k in _dead_keys]), 'models': GEMINI_MODELS[:1]},
        'local_text_extraction': True,
    }

def _client_for(key):
    if key not in _clients:
        _clients[key] = Groq(api_key=key, max_retries=0, timeout=90.0)
    return _clients[key]

def _gemini_client_for(key):
    if ('gem', key) not in _clients:
        from google import genai
        _clients[('gem', key)] = genai.Client(api_key=key)
    return _clients[('gem', key)]

def get_groq_client():
    keys = get_groq_keys()
    if not keys:
        raise GroqUnavailable('NOT_CONFIGURED', 'Groq is not configured.', 'No GROQ_API_KEY is set.', 500, False)
    with _key_lock:
        live = [k for k in keys if k not in _dead_keys] or keys
        return _client_for(live[_key_index['i'] % len(live)]), len(keys)

def rotate_groq_key():
    with _key_lock:
        _key_index['i'] += 1
    return len(get_groq_keys()) > 1

def _classify_error(e):
    """-> (kind, retry_after_seconds). kind: rate|auth|model|reasoning|transient|other"""
    status = getattr(e, 'status_code', None) or getattr(e, 'code', None)
    if not isinstance(status, int):
        status = None
    msg = str(e).lower()
    name = type(e).__name__
    if status == 400 and 'reasoning' in msg:
        return 'reasoning', 0
    if 'api key not valid' in msg or 'api_key_invalid' in msg or 'invalid api key' in msg:
        return 'auth', 0
    if status in (400, 403, 404) and ('model' in msg) and any(
            w in msg for w in ('decommission', 'not found', 'does not exist', 'no longer',
                               'not supported', 'permission', 'access', 'deprecated', 'blocked')):
        return 'model', 0
    if status in (401, 403):
        return 'auth', 0
    if status == 429 or name == 'RateLimitError' or 'resource_exhausted' in msg:
        ra = 0.0
        try:
            ra = float(e.response.headers.get('retry-after', 0))
        except Exception:
            m = re.search(r'(?:try again|retry) in\s+(?:(\d+)m)?\s*([\d.]+)s', msg)
            if m:
                ra = int(m.group(1) or 0) * 60 + float(m.group(2))
        if not ra:
            ra = 3600 if 'per day' in msg or 'daily' in msg else 20
        return 'rate', min(ra, 3600)
    if (status and status >= 500) or name in ('APIConnectionError', 'APITimeoutError', 'InternalServerError',
                                              'ServerError', 'ConnectError', 'ReadTimeout'):
        return 'transient', 0
    return 'other', 0

def _run_chain(provider, keys, models, invoke, max_tokens):
    """Walk models; per model rotate keys with cooldowns. invoke(key, model, reasoning, max_tokens)->text."""
    P = provider.upper()
    last, saw_rate, saw_net = '', False, False
    for model in [m for m in models if m not in _bad_models]:
        transient_left, guard = 2, 0
        while guard < len(keys) * 3 + 4:
            guard += 1
            now = time.time()
            with _key_lock:
                usable = [k for k in keys if k not in _dead_keys]
                if not usable:
                    raise GroqUnavailable(f'{P}_AUTH', f'{provider.title()} rejected all configured API keys.',
                        f'Every {P}_API_KEY value was refused (invalid or revoked). Create a fresh key and update the environment.',
                        502, False)
                ready = [k for k in usable if _cooldowns.get((k, model), 0) <= now]
                if ready:
                    key = ready[_key_index['i'] % len(ready)]
                    _key_index['i'] += 1
                    wait = 0
                else:
                    key = None
                    wait = min(_cooldowns.get((k, model), 0) for k in usable) - now
            if key is None:
                if wait <= 25:
                    time.sleep(max(wait, 0.5)); continue
                saw_rate = True
                break
            try:
                text = invoke(key, model, model not in _no_reasoning, max_tokens)
                text = re.sub(r'<think>.*?</think>', '', text or '', flags=re.S).strip()
                if text:
                    return text
                last = f'{model}: empty reply'
                break
            except Exception as e:
                ekind, ra = _classify_error(e)
                last = f'{model}: {type(e).__name__} {getattr(e, "status_code", "") or getattr(e, "code", "")}'
                log.warning('%s call failed (key #%d, %s)', provider, keys.index(key) + 1, last)
                if ekind == 'reasoning':
                    _no_reasoning.add(model); continue
                if ekind == 'model':
                    _bad_models.add(model); break
                if ekind == 'auth':
                    with _key_lock: _dead_keys[key] = last
                    continue
                if ekind == 'rate':
                    saw_rate = True
                    with _key_lock: _cooldowns[(key, model)] = time.time() + ra
                    continue
                if ekind == 'transient':
                    saw_net = True
                    if transient_left > 0:
                        transient_left -= 1; time.sleep(2); continue
                break
    if saw_rate:
        raise GroqUnavailable('RATE_LIMITED', f'{provider.title()} is rate-limited right now.',
                              'Every key hit its quota. Wait a minute and retry, or add another key.', 429, True)
    if saw_net:
        raise GroqUnavailable('NETWORK_ERROR', f'Could not reach {provider.title()}.',
                              'Network failure or provider outage. Try again shortly.', 503, True)
    raise GroqUnavailable('AI_UNAVAILABLE', f'{provider.title()} could not process the request.',
                          f'Last error: {last or "no usable model"}. Models may be retired; set '
                          f'{"GEMINI_MODEL" if provider == "gemini" else "GROQ_TEXT_MODEL / GROQ_VISION_MODEL"}.',
                          503, True)

def groq_chat(kind, messages, max_tokens=2000, temperature=0.0):
    keys = get_groq_keys()
    if not keys:
        raise GroqUnavailable('NOT_CONFIGURED', 'Groq is not configured.', 'No GROQ_API_KEY is set.', 500, False)
    def invoke(key, model, reasoning, mt):
        kw = dict(model=model, messages=messages, temperature=temperature,
                  max_tokens=mt + (1500 if model.startswith('openai/gpt-oss') else 0))
        if reasoning:
            kw['extra_body'] = {'reasoning_effort': 'low' if model.startswith('openai/gpt-oss') else 'none'}
        r = _client_for(key).chat.completions.create(**kw)
        return r.choices[0].message.content or ''
    return _run_chain('groq', keys, VISION_MODELS if kind == 'vision' else TEXT_MODELS, invoke, max_tokens)

def gemini_generate(prompt, pdf_bytes=None, max_tokens=8000):
    """Gemini text, or native PDF understanding when pdf_bytes is given (inline, <=18 MB)."""
    keys = get_gemini_keys()
    if not keys:
        raise GroqUnavailable('NOT_CONFIGURED', 'Gemini is not configured.', 'No GEMINI_API_KEY is set.', 500, False)
    def invoke(key, model, reasoning, mt):
        from google.genai import types
        parts = []
        if pdf_bytes is not None:
            parts.append(types.Part.from_bytes(data=pdf_bytes, mime_type='application/pdf'))
        parts.append(prompt)
        r = _gemini_client_for(key).models.generate_content(
            model=model, contents=parts,
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=mt))
        return r.text or ''
    return _run_chain('gemini', keys, GEMINI_MODELS, invoke, max_tokens)

def ai_text(system, user, max_tokens=2500):
    """Text generation: Groq first, Gemini as fallback; either may be absent."""
    errs = []
    if get_groq_keys():
        try:
            return groq_chat('text', [{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens)
        except GroqUnavailable as e:
            errs.append(e)
    if get_gemini_keys():
        try:
            return gemini_generate(system + "\n\n" + user, max_tokens=max_tokens + 1500)
        except GroqUnavailable as e:
            errs.append(e)
    if errs:
        raise errs[0]
    raise GroqUnavailable('NOT_CONFIGURED', 'No AI provider configured.',
                          'Set GROQ_API_KEY and/or GEMINI_API_KEY.', 500, False)

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

# ── STRUCTURED ERRORS ──
class PDFProcessingError(Exception):
    """Raised for PDF issues that should surface as a structured, user-facing
    error instead of a generic 500 or a guessable plain string."""
    def __init__(self, code, message, details=''):
        self.code = code
        self.message = message
        self.details = details
        super().__init__(message)

def err_response(code, message, details='', status=400, retryable=False):
    return jsonify({
        'success': False,
        'error': {
            'code': code,
            'message': message,
            'details': _redact(details),
            'retryable': retryable
        }
    }), status

# ── PDF EXTRACTION ──
# Two independent, cheap (non-OCR) text-layer extractors are tried before
# ever falling back to the expensive vision-model OCR path. pdfplumber and
# PyMuPDF use different underlying decoders — a real academic PDF with an
# unusual/subsetted font encoding can come back empty or garbled from one
# library while the other reads it fine. Trying both costs a few ms and
# avoids OCR (and its Groq rate limits / cold-start time) for PDFs that
# actually have a perfectly good text layer.

def _open_pymupdf_validated(pdf_path):
    """Open with PyMuPDF first: this is also our validation step, since it
    reliably distinguishes 'not a real PDF' / 'encrypted' / 'zero pages'
    from a genuine extraction difficulty."""
    import pymupdf
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        raise PDFProcessingError(
            'CORRUPTED_PDF',
            "This file doesn't look like a valid PDF.",
            "It may be corrupted, truncated, or not actually a PDF. "
            "Try re-downloading or re-exporting it."
        ) from e
    if doc.is_encrypted:
        # Some university PDFs are "protected" (no-copy/no-print) but have
        # no real open password — that case unlocks with an empty password.
        if not doc.authenticate(""):
            doc.close()
            raise PDFProcessingError(
                'ENCRYPTED_PDF',
                "This PDF is password-protected.",
                "Remove the password (open it and use 'Print to PDF' or "
                "'Save As' to create an unprotected copy) and re-upload."
            )
    if doc.page_count == 0:
        doc.close()
        raise PDFProcessingError('EMPTY_PDF', "This PDF has no pages.", "")
    return doc

def _extract_text_pdfplumber(pdf_path):
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
    except Exception:
        # Don't hard-fail here — PyMuPDF gets a chance next. A real
        # corrupted/encrypted file was already caught by the PyMuPDF
        # validation step above, so a pdfplumber-only failure usually just
        # means pdfplumber's decoder specifically choked on this file.
        return ""
    return text

def _extract_text_pymupdf(doc):
    text = ""
    try:
        for page in doc:
            text += (page.get_text() or "") + "\n"
    except Exception:
        return ""
    return text

def extract_text_from_pdf(pdf_path):
    """Returns (text, method, page_count). Raises PDFProcessingError for
    corrupted / encrypted / zero-page files."""
    doc = _open_pymupdf_validated(pdf_path)
    try:
        page_count = doc.page_count
        plumber_text = clean_pdf_text(_extract_text_pdfplumber(pdf_path))
        if plumber_text and len(plumber_text) >= 150:
            return plumber_text, 'pdfplumber', page_count
        # pdfplumber came back thin/empty — try PyMuPDF's own text layer
        # (cheap, no rendering/OCR involved) before assuming it's scanned.
        mupdf_text = clean_pdf_text(_extract_text_pymupdf(doc))
        if len(mupdf_text) > len(plumber_text):
            return mupdf_text, 'pymupdf', page_count
        return plumber_text, 'pdfplumber', page_count
    finally:
        doc.close()

def is_scanned_pdf(text):
    if not text or len(text.strip()) < 150: return True
    cleaned = clean_pdf_text(text)
    if not cleaned or len(cleaned.strip()) < 100: return True
    alpha = len([c for c in cleaned if c.isalpha()]) / max(len(cleaned), 1)
    if alpha < 0.35: return True
    if not re.search(r'Q\.?\s*\d|explain|describe|define|discuss|marks|attempt', cleaned, re.I): return True
    return False

OCR_MAX_PAGES = 6
OCR_PAGES_PER_REQUEST = 2
GEMINI_PDF_MAX_BYTES = 18 * 1024 * 1024
_ocr_cache = {}   # sha256 -> (text, provider, warnings); a file is never OCR'd twice

OCR_PROMPT = ("Transcribe ALL printed text of this university exam question paper exactly, in reading order. "
              "Keep question numbers (Q1, Q2, a), b), (A), (B)), marks (e.g. [5], 10 marks), headings, and render "
              "tables as plain text rows. If a page is rotated, read it upright. If any word or number is unreadable, "
              "write [illegible] instead of guessing. Do not add commentary and do not invent missing text. Plain text only.")

def _render_pages_b64(pdf_path):
    """Render pages (<=OCR_MAX_PAGES) as contrast-enhanced grayscale JPEGs (long edge <=1800px)."""
    import pymupdf, base64
    from PIL import Image as PILImage, ImageOps, ImageFilter
    out, total = [], 0
    doc = pymupdf.open(pdf_path)
    try:
        total = len(doc)
        for i in range(min(total, OCR_MAX_PAGES)):
            pix = doc[i].get_pixmap(matrix=pymupdf.Matrix(150 / 72, 150 / 72), alpha=False)
            img = PILImage.open(io.BytesIO(pix.tobytes("png"))).convert('L')
            img = ImageOps.autocontrast(img, cutoff=1).filter(ImageFilter.SHARPEN)
            if max(img.size) > 1800:
                img.thumbnail((1800, 1800))
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=80, optimize=True)
            out.append(base64.b64encode(buf.getvalue()).decode())
    finally:
        doc.close()
        gc.collect()
    return out, total

def _groq_vision_ocr(pdf_path):
    pages, total = _render_pages_b64(pdf_path)
    if not pages:
        return "", []
    parts = []
    for i in range(0, len(pages), OCR_PAGES_PER_REQUEST):
        content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
                   for b in pages[i:i + OCR_PAGES_PER_REQUEST]]
        content.append({"type": "text", "text": OCR_PROMPT})
        parts.append(groq_chat('vision', [{"role": "user", "content": content}], max_tokens=3500))
    warns = [f"Only the first {OCR_MAX_PAGES} of {total} pages were read."] if total > OCR_MAX_PAGES else []
    return '\n'.join(parts), warns

def extract_text_via_vision(pdf_path, groq_client=None):
    """Back-compat wrapper: Groq vision OCR of a PDF."""
    return clean_pdf_text(_groq_vision_ocr(pdf_path)[0])

def ocr_pdf(pdf_path, file_hash=None):
    """OCR a scanned PDF. Gemini native PDF understanding first (if configured), Groq vision second.
    Returns (text, provider, warnings). Raises GroqUnavailable only if every configured provider failed."""
    if file_hash and file_hash in _ocr_cache:
        t, p, w = _ocr_cache[file_hash]
        return t, p, list(w) + ['Result reused from an earlier OCR of the identical file.']
    errs, text, provider, warns = [], '', None, []
    if get_gemini_keys() and os.path.getsize(pdf_path) <= GEMINI_PDF_MAX_BYTES:
        try:
            with open(pdf_path, 'rb') as f:
                text = gemini_generate(OCR_PROMPT, pdf_bytes=f.read(), max_tokens=8000)
            provider = 'gemini'
        except GroqUnavailable as e:
            errs.append(e)
    if not text and get_groq_keys():
        try:
            text, warns = _groq_vision_ocr(pdf_path)
            provider = 'groq'
        except GroqUnavailable as e:
            errs.append(e)
    if not text:
        if errs:
            raise errs[-1] if len(errs) == 1 else errs[0]
        raise GroqUnavailable('NO_OCR_PROVIDER', 'This looks like a scanned PDF but no OCR provider is configured.',
                              'Set GEMINI_API_KEY and/or GROQ_API_KEY to read scanned papers.', 422, False)
    ill = len(re.findall(r'\[illegible\]', text, re.I))
    if ill:
        warns.append(f"{ill} unreadable fragment(s) marked [illegible] — those questions may be incomplete.")
    text = clean_pdf_text(text)
    if len(text) < 200:
        warns.append("OCR returned very little text; the scan may be blurry, cropped or low-contrast.")
    if len(_ocr_cache) > 32:
        _ocr_cache.clear()
    _ocr_cache[file_hash or ''] = (text, provider, warns)
    return text, provider, warns

# ── STEP 1: EXTRACT QUESTIONS AS JSON ──
def extract_questions_as_json(paper_name, text, client):
    prompt = f"""Extract ALL exam questions from this paper as a JSON array.

Paper: {paper_name}
Text:
{text[:9000]}

Return ONLY a JSON array:
[
  {{"id":"{paper_name}_Q1a","paper":"{paper_name}","position":"Q1a","marks":5,"question":"exact question text"}},
  {{"id":"{paper_name}_Q2A","paper":"{paper_name}","position":"Q2A","marks":10,"question":"exact question text"}}
]

Rules:
- Include ALL questions from Q1a through Q6B
- Each question gets unique id: paper_position
- marks: exactly as printed in the paper; 0 if not printed or unclear (never guess)
- Extract exact question text copied from the paper; NEVER invent or complete missing text
- Omit a question entirely if its text is not present
- Return ONLY the JSON array, no markdown"""
    for _attempt in range(2):
        try:
            raw = ai_text("Extract exam questions as JSON array. No markdown.", prompt, max_tokens=3500)
            raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
            raw = re.sub(r'\s*```$', '', raw).strip()
            i, j = raw.find('['), raw.rfind(']')
            questions = json.loads(raw[i:j + 1])
            for q in questions:
                q['paper'] = paper_name
            return questions
        except GroqUnavailable:
            raise
        except Exception:
            continue
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

_SYN = {r'\bsvm\b': 'support vector machine', r'\bpca\b': 'principal component analysis',
        r'\bknn\b': 'k nearest neighbor', r'k[\s-]?means': 'kmeans', r'\bnn\b': 'neural network',
        r'\bnaive bayes(?:ian)?\b': 'naivebayes', r'\bover[\s-]?fit(?:ting)?\b': 'overfitting',
        r'\bunder[\s-]?fit(?:ting)?\b': 'underfitting', r'\bbias[\s-]?variance\b': 'bias variance'}

def _stem(t):
    for suf in ('ing', 'ed', 'es', 's', 'ly'):
        if len(t) > len(suf) + 3 and t.endswith(suf):
            return t[:-len(suf)]
    return t

def _expand(text):
    text = text.lower()
    for pat, rep in _SYN.items():
        text = re.sub(pat, rep, text)
    return text

def tokenize(text):
    text = _expand(text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return [_stem(t) for t in text.split() if t not in STOPWORDS and len(t) > 2]

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
    q_lower = _expand(question_text)
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
                    if sim > 0.33 and unmatched[i]['paper'] != unmatched[j]['paper']:
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
        importance = importance_for(freq, num_papers)
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
        cluster['importance'] = importance_for(f, num_papers)
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

    # Sort clusters deterministically so same input = same prompt = same output
    sorted_clusters = sorted(clusters, key=lambda c: (-c['frequency'], c['topic']))
    cluster_summary = json.dumps([{
        'topic': c['topic'],
        'frequency': c['frequency'],
        'importance': c['importance'],
        'papers': sorted(c['papers']),
        'sample_question': c['questions'][0][:200] if c['questions'] else '',
        'typical_marks': c['marks_each_time'][0] if c['marks_each_time'] else 10,
        'position': c['question_positions'][0] if c['question_positions'] else ''
    } for c in sorted_clusters], indent=2)[:6000]

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
            raw = ai_text("Return ONLY valid JSON. Start with { end with }. No markdown. No text before or after.",
                          prompt, max_tokens=3500)
            raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
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
        except GroqUnavailable:
            break  # key/model/rate problems already retried inside groq_chat -> Python fallback
        except Exception:
            break

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
    name_no_ext = re.sub(r'\.[a-z0-9]{1,5}$', '', filename.lower())
    parts = name_no_ext.replace('-', '_').split('_')
    year = next((int(p) for p in parts if re.match(r'^20\d\d$', p)), None)
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

MAX_FILES = 8
MAX_FILE_BYTES = 15 * 1024 * 1024
DUP_SIMILARITY = 0.85
PREDICTION_DISCLAIMER = ("Predictions are estimates based on how often topics appeared in the uploaded papers. "
                         "They are not guaranteed questions and are not a measured accuracy.")

def _tok_set(t): return set(tokenize(t))

def annotate_clusters(clusters, total_unique, papers, qlookup):
    for c in clusters:
        refs, uncertain = [], []
        aligned_marks = []
        for i, p in enumerate(c.get('papers', [])):
            pos = c['question_positions'][i] if i < len(c['question_positions']) else ''
            q = qlookup.get((p, pos), {})
            m = q.get('marks', 0) or 0
            aligned_marks.append(m)
            info = papers.get(p, {})
            refs.append({'paper': p, 'position': pos, 'marks': m, 'question': c['questions'][i] if i < len(c['questions']) else '',
                         'extraction': info.get('method', '')})
            if q.get('uncertain'):
                uncertain.append(f"{p} {pos}: question text may be incomplete")
            if str(info.get('method', '')).startswith('ocr'):
                uncertain.append(f"{p}: read via OCR")
        c['marks_each_time'] = aligned_marks
        nz = [m for m in aligned_marks if m > 0]
        c['consistent_marks'] = len(set(nz)) == 1 if len(nz) > 1 else False
        c['references'] = refs
        c['unique_papers_with_topic'] = c['frequency']
        c['total_unique_papers'] = total_unique
        c['frequency_label'] = f"{c['frequency']} of {total_unique} unique papers"
        c['uncertainty'] = sorted(set(uncertain))
        c['confidence'] = 'low' if any('incomplete' in u for u in uncertain) else ('medium' if uncertain else 'high')
    return clusters

def build_year_trends(papers, clusters):
    rows = []
    for name, p in sorted(papers.items(), key=lambda x: x[1]['sort_key']):
        qs = p.get('questions', [])
        rows.append({'paper': name, 'period': p['meta'].get('period') or '', 'year': p['meta'].get('year'),
                     'session': p['meta'].get('session') or '', 'questions': len(qs),
                     'marks_total': sum(q.get('marks', 0) or 0 for q in qs),
                     'marks_distribution': marks_distribution(qs),
                     'repeated_topics': [c['topic'] for c in clusters if name in c.get('papers', [])]})
    return rows

def run_backtest(all_questions, papers, subject):
    """Hold out the newest dated paper; check how many of its questions fall in topics that earlier
    papers already flagged as repeating. A genuine out-of-sample check — only with >=3 unique papers."""
    dated = {n: p for n, p in papers.items() if p['sort_key'] > 0}
    if len(papers) < 3 or len(dated) < 2:
        return {'available': False, 'reason': 'Needs at least 3 unique papers with identifiable dates; '
                'predictions are therefore unvalidated estimates.'}
    held = max(dated, key=lambda n: dated[n]['sort_key'])
    train_q = [q for q in all_questions if q['paper'] != held]
    held_q = [q for q in all_questions if q['paper'] == held]
    if len(held_q) < 3:
        return {'available': False, 'reason': f'Too few questions were extracted from {held} to validate.'}
    clusters = cluster_questions_python(train_q, subject, len(papers) - 1)
    if not clusters:
        return {'available': False, 'reason': 'No repeating topics were found in the earlier papers to test.'}
    hits = 0
    for q in held_q:
        qt = _tok_set(q['question'])
        ok = False
        for c in clusters:
            if match_topic(q['question'], c['topic']):
                ok = True; break
            for cq in c['questions']:
                s = _tok_set(cq)
                if qt and s and len(qt & s) / len(qt | s) >= 0.3:
                    ok = True; break
            if ok: break
        hits += ok
    return {'available': True, 'held_out_paper': held, 'train_papers': len(papers) - 1,
            'held_out_questions': len(held_q), 'matched_questions': hits,
            'hit_rate_pct': round(100 * hits / len(held_q), 1),
            'note': (f"Back-test: {hits} of {len(held_q)} questions in {held} (unseen) belonged to topics already "
                     f"repeating in the {len(papers) - 1} earlier papers. Small sample — indicative only.")}

def attach_prediction_evidence(predictions, clusters, total_unique):
    for p in predictions:
        pt = _tok_set(str(p.get('topic', '')) + ' ' + str(p.get('question', '')))
        best, bs = None, 0.0
        for c in clusters:
            ct = _tok_set(c['topic'] + ' ' + ' '.join(c['questions']))
            s = len(pt & ct) / max(len(pt | ct), 1)
            if s > bs: best, bs = c, s
        if best is not None and bs >= 0.12:
            p['basis'] = 'repeated-topic'
            p['historical_frequency'] = best['frequency_label']
            p['evidence'] = best['references']
            p['frequency'] = best['frequency']
            p['priority'] = best['importance']
        else:
            p['basis'] = 'model-suggestion'
            p['historical_frequency'] = 'No matching repeated topic found in the uploaded papers'
            p['evidence'] = []
            p['priority'] = 'LOW'
        p['label'] = 'Estimate'
    return predictions

@app.route('/api/providers')
def api_providers():
    return jsonify(provider_status())

@app.route('/analyze', methods=['POST'])
def analyze():
    import uuid
    if 'files' not in request.files:
        return err_response('NO_FILES', 'No files uploaded.', '', 400)
    files = request.files.getlist('files')
    if len(files) < 2:
        return err_response('MIN_FILES', 'Upload at least 2 PYQ papers.',
                             'Ashlysis needs at least 2 different papers to detect repeat patterns.', 400)
    if len(files) > MAX_FILES:
        return err_response('MAX_FILES', f'Maximum {MAX_FILES} papers per analysis.', '', 400)
    non_pdf = [f.filename for f in files if not (f.filename or '').lower().endswith('.pdf')]
    if non_pdf:
        return err_response('INVALID_FILE_TYPE', 'Only PDF files are supported.',
                             f'Remove: {", ".join(non_pdf)}', 400)

    user_name = request.form.get('user_name', 'Student').strip() or 'Student'
    user_name = re.sub(r'[^a-zA-Z0-9 ]', '', user_name)[:30] or 'Student'
    subject_choice = request.form.get('subject', 'auto').strip().lower()
    if subject_choice not in SUBJECT_NAMES:
        subject_choice = 'auto'

    try:
        check_quota()
    except Exception as e:
        return err_response('DAILY_LIMIT', str(e), 'Please try again tomorrow.', 429)
    if not _analysis_lock.acquire(blocking=False):
        return err_response('SERVER_BUSY', 'Server is busy. Please wait 30 seconds and try again.',
                             '', 429, retryable=True)

    papers, seen_hashes, duplicates, ocr_used = {}, {}, [], []
    run_warnings = []
    try:
        for file in files:
            filename = secure_filename(file.filename) or 'paper.pdf'
            tmp = os.path.join(app.config['UPLOAD_FOLDER'], f'{uuid.uuid4().hex}.pdf')
            file.save(tmp)
            try:
                if os.path.getsize(tmp) > MAX_FILE_BYTES:
                    return err_response('FILE_TOO_LARGE', f'"{filename}" is larger than {MAX_FILE_BYTES // 1048576} MB.',
                                         'Compress the PDF or split it.', 413)
                with open(tmp, 'rb') as fh:
                    if fh.read(5) != b'%PDF-':
                        return err_response('CORRUPTED_PDF', f'"{filename}" is not a valid PDF.', '', 400)
                h = sha256_file(tmp)
                if h in seen_hashes:
                    duplicates.append({'file': filename, 'duplicate_of': seen_hashes[h],
                                       'reason': 'Identical file (same SHA-256 hash)', 'similarity': 1.0})
                    continue
                seen_hashes[h] = filename
                try:
                    text, method, page_count = extract_text_from_pdf(tmp)
                except PDFProcessingError as pe:
                    return err_response(pe.code, f'"{filename}": {pe.message}', pe.details, 400,
                                         retryable=(pe.code not in ('ENCRYPTED_PDF', 'CORRUPTED_PDF', 'EMPTY_PDF')))
                warnings = []
                if is_scanned_pdf(text):
                    vt, prov, w = '', None, []
                    try:
                        vt, prov, w = ocr_pdf(tmp, h)
                    except GroqUnavailable as ge:
                        if not text or len(text.strip()) < 50:
                            return err_response(ge.code, f'"{filename}": {ge.message}', ge.details,
                                                 ge.status, retryable=ge.retryable)
                        warnings.append(f"OCR unavailable ({ge.message}); using the partial text layer — extraction may be incomplete.")
                    except Exception as oe:
                        log.warning('OCR failed: %s', type(oe).__name__)
                    if vt and len(vt.strip()) > 50:
                        text, method = vt, f'ocr_{prov}'
                        warnings += w
                    elif not text or len(text.strip()) < 50:
                        return err_response('SCANNED_NO_TEXT', f'"{filename}" doesn\'t contain readable text.',
                            "Neither the text layer nor OCR could read this file. It may be a very blurry scan, "
                            "a photo of a paper, or use an unusual font. Try a clearer scan.", 400, retryable=True)
                    else:
                        warnings.append("OCR returned too little text; using the partial text layer — extraction may be incomplete.")
                meta = detect_metadata(text, filename)
                dup_of = None
                for n, p in papers.items():
                    sim = content_similarity(text, p['text'])
                    pa, pb = meta['period'], p['meta']['period']
                    if sim >= DUP_SIMILARITY and (not (pa and pb) or pa == pb):
                        dup_of = (n, sim); break
                if dup_of:
                    duplicates.append({'file': filename, 'duplicate_of': papers[dup_of[0]]['file'],
                                       'reason': 'Same content (text similarity)', 'similarity': round(dup_of[1], 3)})
                    continue
                paper_name, sort_key = parse_paper_name(filename)
                if not re.search(r'20\d\d', filename) and meta['period']:
                    paper_name = meta['period'].replace(' ', '_'); sort_key = meta['sort_key']
                elif meta['year'] and re.search(r'20\d\d', filename):
                    fy = int(re.search(r'20\d\d', filename).group())
                    if fy != meta['year']:
                        warnings.append(f"Filename says {fy} but the paper text says {meta['year']}; filename used.")
                base, c = paper_name, 1
                while paper_name in papers:
                    paper_name = f"{base}_{c}"; c += 1
                if method.startswith('ocr'):
                    ocr_used.append(paper_name)
                papers[paper_name] = {'text': text, 'sort_key': sort_key, 'meta': meta, 'method': method,
                                      'pages': page_count, 'file': filename, 'warnings': warnings}
            except Exception as e:
                if isinstance(e, GroqUnavailable):
                    raise
                return err_response('SERVER_ERROR', f'Something went wrong processing "{filename}".',
                                     _redact(e), 500, retryable=True)
            finally:
                if os.path.exists(tmp): os.remove(tmp)
                gc.collect()

        N = len(papers)
        if N < 2:
            dtxt = f" {len(duplicates)} duplicate(s) were removed." if duplicates else ''
            return err_response('NOT_ENOUGH_UNIQUE', f'Only {N} unique paper found.',
                                 f'Ashlysis needs at least 2 different papers.{dtxt}', 400)

        # ── subject / code detection ──
        votes = Counter()
        for p in papers.values():
            if p['meta']['subject']:
                votes[p['meta']['subject']] += {'high': 3, 'medium': 2, 'low': 1}.get(p['meta']['subject_confidence'], 1)
        detected = votes.most_common(1)[0][0] if votes else None
        strong = any(p['meta']['subject'] == detected and p['meta']['subject_confidence'] == 'high' for p in papers.values())
        subject, detection_note = 'general', ''
        if subject_choice == 'auto':
            subject = detected or 'general'
            detection_note = 'Detected from paper text.' if detected else 'Could not identify the subject from the papers.'
        elif detected and detected != subject_choice and strong:
            subject = detected
            detection_note = (f"You selected {SUBJECT_NAMES[subject_choice]}, but the papers clearly look like "
                              f"{SUBJECT_NAMES[detected]}; analysed as {SUBJECT_NAMES[detected]}.")
            run_warnings.append(detection_note)
        else:
            subject = subject_choice
            detection_note = 'Using your selection.' + (f" (Text suggests {SUBJECT_NAMES[detected]}.)" if detected and detected != subject_choice else '')
        codes = Counter(p['meta']['subject_code'] for p in papers.values() if p['meta']['subject_code'])
        subject_code = codes.most_common(1)[0][0] if codes else None

        # ── question extraction ──
        all_questions, ai_note = [], False
        for paper_name, p in sorted(papers.items(), key=lambda x: x[1]['sort_key'], reverse=True):
            ctext = clean_pdf_text(p['text'])
            qs, src = [], 'ai'
            try:
                qs = extract_questions_as_json(paper_name, ctext, None)
            except GroqUnavailable as ge:
                p['warnings'].append(f"AI extraction unavailable ({ge.code}); rule-based parser used.")
            kept = [q for q in qs if isinstance(q, dict) and q.get('question') and question_supported_by_text(q['question'], p['text'])]
            if len(kept) < len(qs):
                p['warnings'].append(f"{len(qs) - len(kept)} AI-returned question(s) were dropped because their text was not found in the paper.")
            if not kept:
                kept, src = local_parse_questions(ctext, paper_name), 'local'
                if qs or not get_groq_keys() and not get_gemini_keys():
                    p['warnings'].append("Questions were extracted with the rule-based parser.")
            seen_ids = set()
            for q in kept:
                try: q['marks'] = int(q.get('marks') or 0)
                except Exception: q['marks'] = 0
                q['position'] = str(q.get('position', ''))
                qid = f"{paper_name}_{q['position'] or len(seen_ids)}"
                while qid in seen_ids: qid += '_x'
                seen_ids.add(qid)
                q.update({'id': qid, 'paper': paper_name, 'source': src})
                q['uncertain'] = len(q['question'].strip()) < 20 or '[illegible]' in q['question'].lower()
            if len(kept) < 4:
                p['warnings'].append(f"Only {len(kept)} question(s) detected; the extraction may be incomplete.")
            p['questions'] = kept
            all_questions.extend(kept)
            time.sleep(0.3); gc.collect()

        if not all_questions:
            return err_response('NO_QUESTIONS', 'No questions could be extracted.',
                                 'The text was readable but no question structure (Q1, Q2 ...) was found.', 422)

        clusters = cluster_questions_python(all_questions, subject, N)
        qlookup = {(q['paper'], q['position']): q for q in all_questions}
        annotate_clusters(clusters, N, papers, qlookup)
        paper_strategy = build_paper_strategy(clusters, all_questions, subject)
        enrichment = enrich_with_groq(clusters, user_name, subject, None)
        predictions = enrichment.get('predictions', [])
        last_hour_prep = enrichment.get('last_hour_prep', {})
        if not predictions:
            predictions = [{'question': c['questions'][0] if c['questions'] else c['topic'], 'topic': c['topic'],
                            'confidence': c['importance'], 'reason': f"Appeared in {c['frequency_label']}",
                            'likely_position': c['question_positions'][0] if c['question_positions'] else 'Q2A',
                            'likely_marks': next((m for m in c['marks_each_time'] if m), 10),
                            'frequency': c['frequency']} for c in clusters[:10]]
        if not last_hour_prep.get('must_do'):
            last_hour_prep = {'message': f"Stay focused, {user_name}.",
                'passing_strategy': "Attempt Q1 (20m) + any 2 from Q2-Q6 (40m) = 60m minimum.",
                'must_do': [{'rank': i + 1, 'topic': c['topic'], 'question': (c['questions'][0][:200] if c['questions'] else c['topic']),
                             'marks': next((m for m in c['marks_each_time'] if m), 10),
                             'position': c['question_positions'][0] if c['question_positions'] else '',
                             'why': f"Appeared in {c['frequency_label']}", 'quick_tip': 'Draw diagram, cover all components, give example.'}
                            for i, c in enumerate(clusters[:5])]}
        attach_prediction_evidence(predictions, clusters, N)
        backtest = run_backtest(all_questions, papers, subject)
        trends = build_year_trends(papers, clusters)
        paper_details = [{'paper': n, 'file': p['file'], 'period': p['meta']['period'], 'year': p['meta']['year'],
                          'session': p['meta']['session'], 'pages': p['pages'], 'method': p['method'],
                          'questions': len(p.get('questions', [])), 'warnings': p['warnings']}
                         for n, p in sorted(papers.items(), key=lambda x: x[1]['sort_key'])]
        paper_stats = {n: {'pages': p['pages'], 'chars': len(p['text']), 'ocr': n in ocr_used, 'method': p['method']}
                       for n, p in papers.items()}
        all_warn = list(run_warnings)
        for pd_ in paper_details:
            all_warn += [f"{pd_['paper']}: {w}" for w in pd_['warnings']]
    except GroqUnavailable as ge:
        return err_response(ge.code, ge.message, ge.details, ge.status, retryable=ge.retryable)
    except Exception as e:
        return err_response('SERVER_ERROR', 'Something went wrong during analysis.', _redact(e), 500, retryable=True)
    finally:
        gc.collect()
        _analysis_lock.release()

    return jsonify({
        'success': True,
        'clusters': clusters,
        'predictions': predictions,
        'prediction_disclaimer': PREDICTION_DISCLAIMER,
        'backtest': backtest,
        'last_hour_prep': last_hour_prep,
        'paper_strategy': paper_strategy,
        'user_name': user_name,
        'subject': subject.upper(),
        'subject_name': SUBJECT_NAMES.get(subject, subject),
        'subject_code': subject_code,
        'subject_detection': detection_note,
        'ocr_used': ocr_used,
        'duplicates': duplicates,
        'paper_details': paper_details,
        'year_trends': trends,
        'warnings': all_warn,
        'providers': provider_status(),
        'stats': {
            'uploaded': len(files),
            'papers': N,
            'duplicates_removed': len(duplicates),
            'total_questions': len(all_questions),
            'clusters': len(clusters),
            'high_priority': len([c for c in clusters if c.get('importance') == 'HIGH']),
            'paper_stats': paper_stats
        }
    })

@app.route('/export_pdf', methods=['POST'])
def export_pdf():
    from report_pdf import build_report_pdf
    data = request.get_json(silent=True) or {}
    if not data.get('clusters') and not data.get('predictions'):
        return err_response('NO_DATA', 'Nothing to export.', 'Run an analysis first.', 400)
    pdf = build_report_pdf(data)
    return send_file(io.BytesIO(pdf), mimetype='application/pdf', as_attachment=True,
                     download_name=f"ashlysis_report_{datetime.now().strftime('%Y%m%d')}.pdf")


@app.route('/export', methods=['POST'])
def export_results():
    data = request.json
    clusters      = data.get('clusters', [])
    predictions   = data.get('predictions', [])
    last_hour_prep = data.get('last_hour_prep', {})
    stats         = data.get('stats', {})
    user_name     = data.get('user_name', 'Student')
    subject       = data.get('subject', '')
    today         = datetime.now().strftime('%Y%m%d')
    D = "━" * 52

    lines = [
        "╔══════════════════════════════════════════════════╗",
        "║           ASHLYSIS — EXAM INTELLIGENCE           ║",
        "╚══════════════════════════════════════════════════╝", "",
        f"Student  : {user_name}",
        f"Subject  : {subject}",
        f"Date     : {datetime.now().strftime('%d %b %Y %I:%M %p')}",
        f"Papers   : {stats.get('papers', 0)}",
        f"HIGH     : {stats.get('high_priority', 0)}",
    ]

    # ── SECTION 1: REPEATING QUESTIONS — plain numbered list, ChatGPT/Claude ready ──
    lines += ["", D, "REPEATING QUESTIONS", D, ""]
    lines.append(f"Subject: {subject} | {stats.get('papers', 0)} papers analyzed")
    lines.append("")
    for i, c in enumerate(clusters, 1):
        best_q = c['questions'][0].strip() if c.get('questions') else c.get('topic', '')
        lines.append(f"{i}. {best_q}")
    lines.append("")

    # ── SECTION 2: PREDICTED QUESTIONS ──
    lines += [D, "PREDICTED QUESTIONS", D, ""]
    for i, p in enumerate(predictions, 1):
        lines += [
            f"{i}. [{p.get('confidence')}] {p.get('question', '')[:200]}",
            f"   Pos: {p.get('likely_position', '?')} | Marks: {p.get('likely_marks', '?')}m | {p.get('reason', '')}",
            ""
        ]

    # ── SECTION 3: LAST HOUR PREP ──
    lines += [D, "LAST HOUR PREP — MINIMUM PASSING (40%)", D, ""]
    if last_hour_prep:
        lines += [
            last_hour_prep.get('message', ''),
            f"Strategy: {last_hour_prep.get('passing_strategy', '')}",
            "",
            "MUST DO — 5 questions:",
            ""
        ]
        for q in last_hour_prep.get('must_do', []):
            lines += [
                f"{q.get('rank')}. {q.get('topic')} [{q.get('marks', 10)}m] — {q.get('position', '')}",
                f"   Q: {q.get('question', '')[:200]}",
                f"   Why: {q.get('why', '')}",
                f"   Include: {q.get('quick_tip', '')}",
                ""
            ]

    lines += [D, f"ASHLYSIS — generated for {user_name}", D]
    buf = io.BytesIO("\n".join(lines).encode('utf-8'))
    buf.seek(0)
    return send_file(buf, mimetype='text/plain', as_attachment=True,
                     download_name=f'ashlysis_{user_name.replace(" ", "_")}_{subject}_{today}.txt')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
