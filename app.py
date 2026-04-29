import os
import json
import re
import io
import pdfplumber
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from groq import Groq

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = '/tmp/ashlysis_uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


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


# ── MAIN ANALYSIS WITH GROQ ──
def analyze_with_groq(all_papers_text, user_name):
    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

    papers_content = ""
    for paper_name, text in all_papers_text.items():
        if not text or len(text.strip()) < 50:
            continue
        truncated = text[:3000] if len(text) > 3000 else text
        papers_content += f"\n\n=== PAPER: {paper_name} ===\n{truncated}"

    if not papers_content.strip():
        raise Exception("Could not read text from any of the papers")

    prompt = f"""You are an expert exam analyzer for engineering students in India (Mumbai University).
The student's name is {user_name}. They are analyzing {len(all_papers_text)} previous year exam papers.

Here are the papers:
{papers_content}

Analyze like a smart student would:
1. Read the LATEST paper first — those questions are the base
2. For each question in the latest paper, check how many other papers have the same/similar question
3. Note EXACTLY which question number it appears in (Q1, Q2, Q3 etc) across papers
4. Note the MARKS assigned each time
5. Find the PATTERN — does it always appear in Q1? Always 10 marks?

Respond ONLY with a valid JSON object:
{{
  "clusters": [
    {{
      "topic": "topic name (max 6 words)",
      "frequency": 3,
      "importance": "HIGH",
      "questions": ["exact question from paper 1", "same topic from paper 2", "same topic from paper 3"],
      "papers": ["Paper_2023", "Paper_2024", "Paper_2025"],
      "question_positions": ["Q1", "Q2", "Q1"],
      "marks_each_time": [5, 10, 5],
      "consistent_position": true,
      "consistent_marks": false,
      "pattern_note": "Always appears in Q1, marks vary between 5-10",
      "tip": "one practical exam tip",
      "keywords": ["keyword1", "keyword2"]
    }}
  ],
  "predictions": [
    {{
      "question": "full predicted question text",
      "topic": "topic name",
      "confidence": "HIGH",
      "reason": "one line why this is likely",
      "likely_position": "Q1",
      "likely_marks": 10,
      "frequency": 3
    }}
  ],
  "paper_pattern": {{
    "total_questions": 6,
    "compulsory_question": "Q1 is always compulsory with 4 sub-questions of 5 marks each",
    "optional_questions": "Q2 to Q6 — attempt any 3, each 20 marks",
    "total_marks": 80,
    "duration": "3 hours",
    "key_insight": "one key pattern insight about this paper"
  }},
  "study_plan": {{
    "strategy": "2-3 sentence overall exam strategy personalized for {user_name}",
    "days": [
      {{
        "day": 1,
        "focus": "topic name",
        "priority": "HIGH",
        "hours": 3,
        "tasks": ["specific task 1", "specific task 2", "specific task 3"]
      }}
    ],
    "golden_topics": ["most important topic 1", "topic 2", "topic 3"],
    "dont_skip": ["absolutely critical topic 1", "critical topic 2"]
  }}
}}

Rules:
- clusters: 6-15 items sorted by frequency descending
- HIGH = appeared 3+ papers, MEDIUM = 2 papers, LOW = 1 paper but important
- question_positions: exact Q number like "Q1a", "Q2", "Q4B" etc
- marks_each_time: actual marks from paper, 0 if not visible
- consistent_position: true if same Q number every time
- consistent_marks: true if same marks every time
- predictions: 8-10 items with likely position and marks
- study_plan.days: exactly 7 days
- Return ONLY the JSON, no markdown"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are an expert exam question analyzer. Always respond with valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=4000
    )

    response_text = response.choices[0].message.content.strip()
    response_text = re.sub(r'^```(?:json)?\n?', '', response_text)
    response_text = re.sub(r'\n?```$', '', response_text)
    return json.loads(response_text)


# ── ROUTES ──
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files = request.files.getlist('files')
    if len(files) < 2:
        return jsonify({'error': 'Please upload at least 2 PYQ papers'}), 400
    if len(files) > 10:
        return jsonify({'error': 'Maximum 10 papers allowed'}), 400

    user_name = request.form.get('user_name', 'Student').strip()
    if not user_name:
        user_name = 'Student'

    if not os.environ.get('GROQ_API_KEY'):
        return jsonify({'error': 'Service not configured. Contact admin.'}), 500

    all_papers_text = {}
    paper_stats = {}

    for file in files:
        if file and file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)
            paper_name = filename.rsplit('.', 1)[0]
            parts = paper_name.split('_')
            year = next((p for p in parts if re.match(r'20\d\d', p)), None)
            month = next((p for p in parts if p.lower() in ['may','nov','dec','jun','jan','feb','mar','apr']), None)

            if year and month:
                paper_name = f"{month.title()}_{year}"
            elif year:
                paper_name = f"Paper_{year}"
            else:
                paper_name = f"Paper_{len(all_papers_text)+1}"

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                text = extract_text_from_pdf(filepath)
                if not text or len(text.strip()) < 50:
                    return jsonify({
                        'error': f'Could not extract text from {filename}. Please convert it using OCR at smallpdf.com and try again.'
                    }), 400

                all_papers_text[paper_name] = text
                with pdfplumber.open(filepath) as pdf:
                    paper_stats[paper_name] = {
                        'questions': max(text.count('Q.'), text.count('?'), 5),
                        'pages': len(pdf.pages)
                    }
            except Exception as e:
                return jsonify({'error': f'Failed to read {filename}: {str(e)}'}), 500
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)

    if len(all_papers_text) < 2:
        return jsonify({'error': 'Need at least 2 readable papers to find patterns'}), 400

    try:
        result = analyze_with_groq(all_papers_text, user_name)
        clusters = result.get('clusters', [])
        predictions = result.get('predictions', [])
        study_plan = result.get('study_plan', {})
        paper_pattern = result.get('paper_pattern', {})
    except json.JSONDecodeError:
        return jsonify({'error': 'Analysis failed. Please try again.'}), 500
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

    return jsonify({
        'clusters': clusters,
        'predictions': predictions,
        'study_plan': study_plan,
        'paper_pattern': paper_pattern,
        'user_name': user_name,
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

    lines = [
        "╔══════════════════════════════════════════════════╗",
        "║           ASHLYSIS — EXAM INTELLIGENCE           ║",
        "╚══════════════════════════════════════════════════╝",
        "",
        f"Student         : {user_name}",
        f"Papers analyzed : {stats.get('papers', 0)}",
        f"Repeat clusters : {stats.get('clusters', 0)}",
        f"HIGH priority   : {stats.get('high_priority', 0)}",
    ]

    if paper_pattern:
        lines += [
            "", "━"*52,
            "PAPER PATTERN ANALYSIS",
            "━"*52,
            f"Structure  : {paper_pattern.get('compulsory_question','')}",
            f"Optional   : {paper_pattern.get('optional_questions','')}",
            f"Total marks: {paper_pattern.get('total_marks',80)}",
            f"Duration   : {paper_pattern.get('duration','3 hours')}",
            f"Key insight: {paper_pattern.get('key_insight','')}",
        ]

    lines += ["", "━"*52, "REPEATING QUESTIONS (ranked by frequency)", "━"*52]

    for i, c in enumerate(clusters, 1):
        positions = c.get('question_positions', [])
        marks = c.get('marks_each_time', [])
        consistent_pos = c.get('consistent_position', False)
        consistent_marks = c.get('consistent_marks', False)

        lines += [
            f"\n{i}. [{c.get('importance')}] {c.get('topic')}",
            f"   Repeated : {c.get('frequency')} times",
            f"   Papers   : {', '.join(c.get('papers', []))}",
            f"   Position : {', '.join(str(p) for p in positions)} {'← ALWAYS SAME POSITION ✓' if consistent_pos else ''}",
            f"   Marks    : {', '.join(str(m) for m in marks)} {'← ALWAYS SAME MARKS ✓' if consistent_marks else ''}",
            f"   Pattern  : {c.get('pattern_note','')}",
            f"   Tip      : {c.get('tip','')}",
            "   Questions:"
        ]
        for q in c.get('questions', [])[:3]:
            lines.append(f"   • {q[:200]}")

    lines += ["", "━"*52, "PREDICTED QUESTIONS FOR NEXT EXAM", "━"*52]
    for i, p in enumerate(predictions, 1):
        lines += [
            f"\n{i}. [{p.get('confidence')}] {p.get('question','')[:200]}",
            f"   Topic    : {p.get('topic')}",
            f"   Position : likely {p.get('likely_position','?')} | Marks: {p.get('likely_marks','?')}",
            f"   Why      : {p.get('reason','')}"
        ]

    lines += ["", "━"*52, "7-DAY STUDY PLAN", "━"*52]
    if study_plan:
        lines.append(f"\nStrategy: {study_plan.get('strategy','')}")
        lines.append(f"Golden Topics: {', '.join(study_plan.get('golden_topics',[]))}")
        lines.append(f"Never Skip: {', '.join(study_plan.get('dont_skip',[]))}")
        for day in study_plan.get('days', []):
            lines += [f"\nDay {day['day']} [{day['priority']}] — {day['focus']} ({day['hours']}hrs)"]
            for t in day.get('tasks', []):
                lines.append(f"  ✓ {t}")

    lines += ["", "━"*52, f"Generated by ASHLYSIS for {user_name}", "━"*52]
    buf = io.BytesIO("\n".join(lines).encode('utf-8'))
    buf.seek(0)
    return send_file(buf, mimetype='text/plain', as_attachment=True, download_name='ashlysis_report.txt')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
