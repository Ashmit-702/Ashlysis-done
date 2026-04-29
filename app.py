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
def analyze_with_groq(all_papers_text):
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

Here are {len(all_papers_text)} previous year exam papers:
{papers_content}

Analyze these papers and find:
1. Questions/topics that REPEAT across multiple papers (same concept, even if wording differs)
2. Rank them by frequency
3. Predict likely questions for the NEXT exam
4. Create a 7-day study plan

Respond ONLY with a valid JSON object:
{{
  "clusters": [
    {{
      "topic": "topic name (max 6 words)",
      "frequency": 3,
      "importance": "HIGH",
      "questions": ["exact question from paper 1", "same topic from paper 2"],
      "papers": ["Paper_2023", "Paper_2024"],
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
      "frequency": 3
    }}
  ],
  "study_plan": {{
    "strategy": "2-3 sentence overall exam strategy",
    "days": [
      {{
        "day": 1,
        "focus": "topic name",
        "priority": "HIGH",
        "hours": 3,
        "tasks": ["task 1", "task 2", "task 3"]
      }}
    ],
    "golden_topics": ["topic1", "topic2", "topic3"],
    "dont_skip": ["critical topic 1", "critical topic 2"]
  }}
}}

Rules:
- clusters: 6-15 items sorted by frequency descending
- HIGH = appeared 3+ papers or very core topic
- MEDIUM = appeared in 2 papers
- LOW = appeared once but important
- predictions: 8-10 items
- study_plan.days: exactly 7 days
- Return ONLY the JSON, no markdown, no explanation"""

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

    if not os.environ.get('GROQ_API_KEY'):
        return jsonify({'error': 'Service not configured. Contact admin.'}), 500

    all_papers_text = {}
    paper_stats = {}

    for file in files:
        if file and file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)

            # Clean paper name - extract year/month
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
        result = analyze_with_groq(all_papers_text)
        clusters = result.get('clusters', [])
        predictions = result.get('predictions', [])
        study_plan = result.get('study_plan', {})
    except json.JSONDecodeError:
        return jsonify({'error': 'Analysis failed. Please try again.'}), 500
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

    return jsonify({
        'clusters': clusters,
        'predictions': predictions,
        'study_plan': study_plan,
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
    stats = data.get('stats', {})

    lines = [
        "╔══════════════════════════════════════════════════╗",
        "║           ASHLYSIS — EXAM INTELLIGENCE           ║",
        "╚══════════════════════════════════════════════════╝",
        "",
        f"Papers analyzed : {stats.get('papers', 0)}",
        f"Repeat clusters : {stats.get('clusters', 0)}",
        f"HIGH priority   : {stats.get('high_priority', 0)}",
        "", "━"*52,
        "SECTION 1 — REPEATING QUESTIONS (ranked)",
        "━"*52,
    ]
    for i, c in enumerate(clusters, 1):
        lines += [
            f"\n{i}. [{c.get('importance')}] {c.get('topic')} — {c.get('frequency')}x",
            f"   Papers: {', '.join(c.get('papers', []))}",
            f"   Tip: {c.get('tip', '')}",
            "   Questions:"
        ]
        for q in c.get('questions', [])[:3]:
            lines.append(f"   • {q[:200]}")

    lines += ["", "━"*52, "SECTION 2 — PREDICTED QUESTIONS", "━"*52]
    for i, p in enumerate(predictions, 1):
        lines += [
            f"\n{i}. [{p.get('confidence')}] {p.get('question','')[:200]}",
            f"   Topic: {p.get('topic')} | Why: {p.get('reason','')}"
        ]

    lines += ["", "━"*52, "SECTION 3 — 7-DAY STUDY PLAN", "━"*52]
    if study_plan:
        lines.append(f"\nStrategy: {study_plan.get('strategy','')}")
        lines.append(f"Golden Topics: {', '.join(study_plan.get('golden_topics',[]))}")
        lines.append(f"Never Skip: {', '.join(study_plan.get('dont_skip',[]))}")
        for day in study_plan.get('days', []):
            lines += [f"\nDay {day['day']} [{day['priority']}] — {day['focus']} ({day['hours']}hrs)"]
            for t in day.get('tasks', []):
                lines.append(f"  ✓ {t}")

    lines += ["", "━"*52, "Generated by ASHLYSIS — Powered by Groq AI", "━"*52]
    buf = io.BytesIO("\n".join(lines).encode('utf-8'))
    buf.seek(0)
    return send_file(buf, mimetype='text/plain', as_attachment=True, download_name='ashlysis_report.txt')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
