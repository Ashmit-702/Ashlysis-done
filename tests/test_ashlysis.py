"""Run: python tests/test_ashlysis.py   (all AI calls are mocked; no keys or network needed)"""
import os, sys, io, json, re, tempfile, httpx, groq
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for k in list(os.environ):
    if k.startswith(('GROQ', 'GEMINI')): del os.environ[k]
import app as A, ashlysis_core as C, make_sample_papers as M
A.time.sleep = lambda s: None
R = {}
def check(name, cond):
    R[name] = bool(cond); print(('PASS ' if cond else 'FAIL ') + name)

def herr(cls, status, msg, headers=None):
    r = httpx.Response(status, request=httpx.Request('POST', 'https://x'), headers=headers or {}, json={'error': {'message': msg}})
    return cls(msg, response=r, body=None)
class Msg:
    def __init__(s, t): s.choices = [type('C', (), {'message': type('M', (), {'content': t})()})()]
class FakeGroq:
    def __init__(s, fn): s.fn = fn; s.calls = []; s.chat = type('X', (), {'completions': s})()
    def create(s, **kw):
        s.calls.append(kw['model']); a = s.fn(kw)
        if isinstance(a, Exception): raise a
        return Msg(a)
class GResp:
    def __init__(s, t): s.text = t
class FakeGem:
    def __init__(s, fn): s.fn = fn; s.calls = []; s.models = s
    def generate_content(s, model, contents, config):
        s.calls.append(model); a = s.fn(model, contents)
        if isinstance(a, Exception): raise a
        return GResp(a)

def reset(groq_fn=None, gem_fn=None):
    for k in list(os.environ):
        if k.startswith(('GROQ', 'GEMINI')): del os.environ[k]
    A._clients.clear(); A._dead_keys.clear(); A._cooldowns.clear(); A._bad_models.clear(); A._no_reasoning.clear(); A._ocr_cache.clear()
    fg, fm = None, None
    if groq_fn: os.environ['GROQ_API_KEY'] = 'gsk_TESTKEY_aaaaaaaaaaaa'; fg = FakeGroq(groq_fn); A._client_for = lambda k: fg
    if gem_fn: os.environ['GEMINI_API_KEY'] = 'AIzaTESTKEY_bbbbbbbbbbbb'; fm = FakeGem(gem_fn); A._gemini_client_for = lambda k: fm
    return fg, fm

# ---------- core ----------
tmp = tempfile.mkdtemp(); P = M.build(tmp)
def rd(p): return A.extract_text_from_pdf(p)[0]
t24 = rd(P['2024 May']); meta = C.detect_metadata(t24, 'ML_2024_May.pdf')
check('metadata: Machine Learning detected (not SPCC)', meta['subject'] == 'ml')
check('metadata: code CSC701, period May 2024, session May/June', meta['subject_code'] == 'CSC701' and meta['period'] == 'May 2024' and meta['session'] == 'May/June')
check('similarity: same paper ~1.0, different years < 0.85', C.content_similarity(t24, rd(P['dup'])) > 0.99 and C.content_similarity(t24, rd(P['2022 May'])) < 0.85)
qs = C.local_parse_questions(A.clean_pdf_text(t24), 'May_2024')
pos = {q['position'] for q in qs}
check('local parser: Q1a-d, Q2A/B, Q3A/B found', {'Q1a', 'Q1b', 'Q1c', 'Q1d', 'Q2A', 'Q2B', 'Q3A', 'Q3B'} <= pos)
check('local parser: marks parsed (5 for Q1, 10 for Q2)', {q['position']: q['marks'] for q in qs}.get('Q1a') == 5 and {q['position']: q['marks'] for q in qs}.get('Q2A') == 10)
check('guard: invented question rejected', not C.question_supported_by_text('Explain quantum blockchain teleportation protocol', t24) and C.question_supported_by_text('Explain K-means clustering', t24))

# ---------- providers ----------
fg, fm = reset(lambda kw: 'ok', lambda m, c: 'gem-ok')
check('groq text works with GROQ_API_KEY only', A.ai_text('s', 'u') == 'ok' and not fm.calls)
reset(None, lambda m, c: 'gem-ok')
check('gemini text works with GEMINI_API_KEY only', A.ai_text('s', 'u') == 'gem-ok')
try: reset(); A.ai_text('s', 'u'); ok = False
except A.GroqUnavailable as e: ok = e.code == 'NOT_CONFIGURED'
check('no keys -> NOT_CONFIGURED (no crash)', ok)
fg, fm = reset(lambda kw: herr(groq.RateLimitError, 429, 'rl', {'retry-after': '900'}), lambda m, c: 'gem-fallback')
check('Groq down -> Gemini fallback for text', A.ai_text('s', 'u') == 'gem-fallback')
from google.genai import errors as GE
fg, fm = reset(None, lambda m, c: GE.ClientError(400, {'error': {'message': 'API key not valid. Please pass a valid API key.', 'status': 'INVALID_ARGUMENT'}}))
try: A.gemini_generate('x'); ok = False
except A.GroqUnavailable as e: ok = e.code == 'GEMINI_AUTH'
check('Gemini invalid key -> GEMINI_AUTH', ok)
def gm(m, c):
    if m == 'gemini-3.8-flash': return GE.ServerError(503, {'error': {'message': 'no available channels for model gemini-3.8-flash'}})
    if m == 'gemini-3-flash-preview': return GE.ClientError(404, {'error': {'message': 'models/gemini-3-flash-preview is not found for API version v1beta'}})
    return 'third-model-ok'
fg, fm = reset(None, gm)
check('Gemini withdrawn/unknown models -> next model in chain', A.gemini_generate('x') == 'third-model-ok' and 'gemini-3.5-flash-lite' in fm.calls)
fg, fm = reset(None, lambda m, c: GE.ClientError(429, {'error': {'message': 'Quota exceeded', 'status': 'RESOURCE_EXHAUSTED'}}))
try: A.gemini_generate('x'); ok = False
except A.GroqUnavailable as e: ok = e.code == 'RATE_LIMITED' and e.status == 429
check('Gemini quota -> RATE_LIMITED 429', ok)
fg, fm = reset(lambda kw: herr(groq.AuthenticationError, 401, 'Invalid API Key'))
try: A.groq_chat('text', [{}]); ok = False
except A.GroqUnavailable as e: ok = e.code == 'GROQ_AUTH'
check('Groq invalid key -> GROQ_AUTH', ok)
fg, fm = reset(lambda kw: herr(groq.APIConnectionError.__mro__[0], 500, 'x') if False else (_ for _ in ()).throw(groq.APIConnectionError(request=httpx.Request('POST', 'https://x'))))
try: A.groq_chat('text', [{}]); ok = False
except A.GroqUnavailable as e: ok = e.code == 'NETWORK_ERROR'
check('network failure -> NETWORK_ERROR', ok)
os.environ['GROQ_API_KEY'] = 'gsk_SECRET_zzzzzzzzzz'
check('redaction strips key values', 'gsk_SECRET' not in A._redact('boom gsk_SECRET_zzzzzzzzzz'))
st = json.dumps(A.provider_status())
check('provider status leaks no key material', 'gsk_' not in st and 'AIza' not in st)

# ---------- OCR paths ----------
def vis(kw):
    c = kw['messages'][-1]['content']
    if isinstance(c, list): return '\n'.join(M.paper_lines('December 2023', M.BASE['2023 Dec']))
    return 'x'
fg, fm = reset(vis, lambda m, c: '\n'.join(M.paper_lines('December 2023', M.BASE['2023 Dec'])))
txt, prov, w = A.ocr_pdf(P['2023 Dec'], 'h1')
check('OCR: Gemini PDF path preferred when configured', prov == 'gemini' and 'Support Vector' in txt or 'SVM' in txt)
reset(vis, None)
txt, prov, w = A.ocr_pdf(P['2023 Dec'], 'h2')
check('OCR: Groq vision path works alone', prov == 'groq' and len(txt) > 200)
fg, fm = reset(vis, lambda m, c: herr(groq.RateLimitError, 429, 'q', {'retry-after': '900'}))
txt, prov, w = A.ocr_pdf(P['2023 Dec'], 'h3')
check('OCR: Gemini fails -> Groq vision fallback', prov == 'groq')
n = len(fg.calls); A.ocr_pdf(P['2023 Dec'], 'h3')
check('OCR: same file not processed twice (cache)', len(fg.calls) == n)
reset(None, lambda m, c: 'Q1 a) Explain bias [illegible] tradeoff. ' * 3)
txt, prov, w = A.ocr_pdf(P['2023 Dec'], 'h4')
check('OCR: [illegible] raises extraction warning', any('illegible' in x for x in w))
try: reset(); A.ocr_pdf(P['2023 Dec'], 'h5'); ok = False
except A.GroqUnavailable as e: ok = e.code == 'NO_OCR_PROVIDER'
check('OCR: no provider -> clear NO_OCR_PROVIDER error', ok)

# ---------- end-to-end, six uploads ----------
def llm(kw=None, m=None, c=None):
    prompt = (kw['messages'][-1]['content'] if kw else c)
    if isinstance(prompt, list): return '\n'.join(M.paper_lines('December 2023', M.BASE['2023 Dec']))
    if 'Extract ALL exam questions' in prompt:
        paper = re.search(r'Paper: (\S+)', prompt).group(1)
        body = prompt.split('Text:\n', 1)[1].split('\n\nReturn ONLY')[0]
        out = C.local_parse_questions(body, paper)
        out.append({'id': 'x', 'paper': paper, 'position': 'Q9A', 'marks': 10, 'question': 'Explain quantum blockchain teleportation protocol design'})
        return json.dumps(out)
    return 'not json'
cl = A.app.test_client()
def run(subject='auto', names=None):
    order = names or ['2022 May', '2022 Dec', '2023 May', '2023 Dec', '2024 May', 'dup']
    files = [(open(P[k], 'rb'), os.path.basename(P[k])) for k in order]
    return cl.post('/analyze', data={'files': files, 'user_name': 'T', 'subject': subject}, content_type='multipart/form-data')
reset(lambda kw: llm(kw=kw), lambda m, c: llm(m=m, c=c))
r = run(subject='spcc'); j = r.get_json()
check('E2E: request succeeds', r.status_code == 200 and j.get('success'))
if not j.get('success'): print(j)
S = j['stats']
check('E2E: 6 uploaded -> 5 unique, 1 duplicate', S['uploaded'] == 6 and S['papers'] == 5 and S['duplicates_removed'] == 1)
check('E2E: duplicate identified as identical hash', j['duplicates'][0]['similarity'] == 1.0 and 'May_2024' in j['duplicates'][0]['file'])
check('E2E: subject = Machine Learning even though SPCC was selected', j['subject'] == 'ML' and j['subject_code'] == 'CSC701')
check('E2E: scanned paper read via OCR', len(j['ocr_used']) == 1)
check('E2E: invented AI question dropped and warned', not any('quantum' in q for c in j['clusters'] for q in c['questions']) and any('dropped' in w for w in j['warnings']))
check('E2E: every frequency <= unique paper count', all(c['frequency'] <= 5 and c['total_unique_papers'] == 5 for c in j['clusters']))
topics = {c['topic'].lower(): c for c in j['clusters']}
svm = [c for c in j['clusters'] if any('support vector' in q.lower() or 'svm' in q.lower() for q in c['questions'])]
check('E2E: SVM questions grouped into one cluster across reworded papers', svm and svm[0]['frequency'] == 5)
bv = [c for c in j['clusters'] if any('bias' in q.lower() for q in c['questions'])]
check('E2E: bias-variance grouped (>=4 papers, reworded)', bv and bv[0]['frequency'] >= 4)
check('E2E: clusters carry references + question wording + confidence', all(c['references'] and c['confidence'] for c in j['clusters']))
check('E2E: predictions labeled estimates with evidence/basis', all(p['label'] == 'Estimate' and 'historical_frequency' in p for p in j['predictions']) and 'not a measured accuracy' in j['prediction_disclaimer'])
check('E2E: back-test computed on unseen newest paper', j['backtest']['available'] and j['backtest']['held_out_paper'].startswith('May_2024'))
check('E2E: year trends cover 5 papers', len(j['year_trends']) == 5)
check('E2E: no API key in response', 'gsk_TESTKEY' not in json.dumps(j) and 'AIzaTESTKEY' not in json.dumps(j))
pr = cl.post('/export_pdf', json=j)
check('PDF report generated (valid %PDF, multi-KB)', pr.status_code == 200 and pr.data[:5] == b'%PDF-' and len(pr.data) > 5000)
import pymupdf
doc = pymupdf.open(stream=pr.data, filetype='pdf'); rt = ''.join(p.get_text() for p in doc)
check('PDF contains required sections', all(x in rt for x in ['Subject and paper details', 'Duplicates removed', 'Topic frequency', 'Year-wise', 'Predicted questions', 'Last-minute', 'Limitations']) and 'Machine Learning' in rt)
open('/tmp/sample_report.pdf', 'wb').write(pr.data)

# independent providers
reset(lambda kw: llm(kw=kw), None); j1 = run().get_json()
check('E2E: Groq only (no Gemini key) works incl. scanned paper', j1.get('success') and j1['stats']['papers'] == 5 and len(j1['ocr_used']) == 1)
reset(None, lambda m, c: llm(m=m, c=c)); j2 = run().get_json()
check('E2E: Gemini only (no Groq key) works incl. scanned paper', j2.get('success') and j2['stats']['papers'] == 5 and len(j2['ocr_used']) == 1)
reset(); j3 = run(names=['2022 May', '2022 Dec', '2023 May', '2024 May', 'dup']).get_json()
check('E2E: NO keys: text PDFs still analysed locally, duplicate removed', j3.get('success') and j3['stats']['papers'] == 4 and j3['subject'] == 'ML')
check('E2E: NO keys: local parser extracted questions, predictions are estimates', j3['stats']['total_questions'] >= 28 and all(p['label'] == 'Estimate' for p in j3['predictions']))
reset(); r4 = run(); j4 = r4.get_json()
check('E2E: NO keys + scanned paper -> clear NO_OCR_PROVIDER error', j4['error']['code'] == 'NO_OCR_PROVIDER')
reset(); j5 = run(names=['2022 May', 'dup']).get_json(); 
j5 = cl.post('/analyze', data={'files': [(open(P['2024 May'], 'rb'), 'a.pdf'), (open(P['dup'], 'rb'), 'b.pdf')], 'subject': 'auto'}, content_type='multipart/form-data').get_json()
check('E2E: two identical files -> NOT_ENOUGH_UNIQUE', j5['error']['code'] == 'NOT_ENOUGH_UNIQUE')
bad = cl.post('/analyze', data={'files': [(io.BytesIO(b'hello'), 'a.pdf'), (io.BytesIO(b'hello2'), 'b.pdf')]}, content_type='multipart/form-data').get_json()
check('validation: fake PDF rejected', bad['error']['code'] == 'CORRUPTED_PDF')
bad = cl.post('/analyze', data={'files': [(io.BytesIO(b'x'), 'a.txt'), (io.BytesIO(b'y'), 'b.pdf')]}, content_type='multipart/form-data').get_json()
check('validation: non-PDF rejected', bad['error']['code'] == 'INVALID_FILE_TYPE')
check('temp upload dir left clean', not [f for f in os.listdir(A.app.config['UPLOAD_FOLDER']) if f.endswith('.pdf')])
src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app.py')).read()
check('no hardcoded keys in source', not re.search(r'(gsk_|AIza)[A-Za-z0-9_\-]{12,}', src))
print(f"\n{sum(R.values())}/{len(R)} passed")
sys.exit(0 if all(R.values()) else 1)
