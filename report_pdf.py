"""Professional PDF report for ASHLYSIS (reportlab). Input = the JSON the /analyze endpoint returned."""
import io
from datetime import datetime
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ACCENT = colors.HexColor('#4f46e5')
LIGHT = colors.HexColor('#f1f5f9')


def _p(text, style):
    return Paragraph(escape(str(text if text is not None else '')), style)


def build_report_pdf(d):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title='ASHLYSIS Report', author='ASHLYSIS')
    ss = getSampleStyleSheet()
    body = ParagraphStyle('b', parent=ss['BodyText'], fontSize=9, leading=12)
    small = ParagraphStyle('s', parent=body, fontSize=8, leading=10, textColor=colors.HexColor('#475569'))
    cell = ParagraphStyle('c', parent=body, fontSize=8, leading=10)
    h1 = ParagraphStyle('h1', parent=ss['Title'], fontSize=20, textColor=ACCENT, spaceAfter=4)
    h2 = ParagraphStyle('h2', parent=ss['Heading2'], fontSize=12, textColor=ACCENT, spaceBefore=10, spaceAfter=4)

    def table(rows, widths, header=True):
        t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
        st = [('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#cbd5e1')), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
              ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4)]
        if header:
            st += [('BACKGROUND', (0, 0), (-1, 0), ACCENT), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white)]
        t.setStyle(TableStyle(st))
        return t

    hdr = ParagraphStyle('hd', parent=cell, textColor=colors.white, fontName='Helvetica-Bold')
    stats = d.get('stats', {}) or {}
    N = stats.get('papers', 0)
    S = []
    S += [_p('ASHLYSIS — Exam Intelligence Report', h1),
          _p(f"Prepared for {d.get('user_name', 'Student')} · {datetime.now().strftime('%d %b %Y')}", small), Spacer(1, 6)]

    # 1. Subject and paper details
    S.append(_p('1. Subject and paper details', h2))
    S.append(table([
        [_p('Subject', cell), _p(d.get('subject_name') or d.get('subject', ''), cell)],
        [_p('Subject code', cell), _p(d.get('subject_code') or 'Not found in the papers', cell)],
        [_p('How subject was identified', cell), _p(d.get('subject_detection', ''), cell)],
    ], [45 * mm, 125 * mm], header=False))

    # 2-3. Counts
    S.append(_p('2. Uploaded vs unique papers', h2))
    S.append(table([
        [_p('Uploaded files', cell), _p(stats.get('uploaded', N), cell)],
        [_p('Duplicates removed', cell), _p(stats.get('duplicates_removed', 0), cell)],
        [_p('Unique papers analysed', cell), _p(N, cell)],
        [_p('Questions extracted', cell), _p(stats.get('total_questions', 0), cell)],
    ], [45 * mm, 125 * mm], header=False))
    for dup in d.get('duplicates', []) or []:
        S.append(_p(f"Duplicate: {dup.get('file')} = {dup.get('duplicate_of')} ({dup.get('reason')})", small))
    rows = [[_p('Paper', hdr), _p('Period', hdr), _p('Pages', hdr), _p('Read via', hdr), _p('Questions', hdr)]]
    for p in d.get('paper_details', []) or []:
        rows.append([_p(p.get('paper'), cell), _p(p.get('period') or '—', cell), _p(p.get('pages'), cell),
                     _p(p.get('method'), cell), _p(p.get('questions'), cell)])
    if len(rows) > 1:
        S += [Spacer(1, 4), table(rows, [48 * mm, 34 * mm, 18 * mm, 42 * mm, 28 * mm])]

    # 4. OCR quality / warnings
    S.append(_p('3. OCR quality and extraction warnings', h2))
    warns = d.get('warnings', []) or []
    if warns:
        for w in warns:
            S.append(_p('• ' + w, body))
    else:
        S.append(_p('No extraction warnings were raised. Text was read cleanly.', body))

    # 5. Repeated questions + 6. frequency table
    cl = d.get('clusters', []) or []
    S.append(_p('4. Topic frequency (unique papers only)', h2))
    rows = [[_p('Topic', hdr), _p('Frequency', hdr), _p('Priority', hdr), _p('Marks', hdr), _p('Confidence', hdr)]]
    for c in cl:
        marks = sorted({m for m in c.get('marks_each_time', []) if m})
        rows.append([_p(c.get('topic'), cell), _p(c.get('frequency_label') or c.get('frequency'), cell),
                     _p(c.get('importance'), cell), _p('/'.join(map(str, marks)) or '—', cell), _p(c.get('confidence', ''), cell)])
    S.append(table(rows, [62 * mm, 40 * mm, 22 * mm, 22 * mm, 24 * mm]) if len(rows) > 1 else _p('No repeating topics found.', body))

    S.append(_p('5. Repeated questions with references', h2))
    for c in cl[:12]:
        S.append(_p(f"{c.get('topic')} — {c.get('frequency_label', '')}", ParagraphStyle('t', parent=body, fontName='Helvetica-Bold')))
        for r in c.get('references', []) or []:
            S.append(_p(f"{r.get('paper')} {r.get('position')} [{r.get('marks') or '?'}m]: {r.get('question')}", cell))
        for u in c.get('uncertainty', []) or []:
            S.append(_p('⚠ ' + u, small))
        S.append(Spacer(1, 3))

    # 7. Year-wise
    S.append(_p('6. Year-wise trends', h2))
    rows = [[_p('Paper', hdr), _p('Questions', hdr), _p('Marks parsed', hdr), _p('Repeating topics present', hdr)]]
    for t in d.get('year_trends', []) or []:
        rows.append([_p(t.get('paper'), cell), _p(t.get('questions'), cell), _p(t.get('marks_total') or '—', cell),
                     _p(', '.join(t.get('repeated_topics', [])[:6]) or '—', cell)])
    if len(rows) > 1:
        S.append(table(rows, [34 * mm, 22 * mm, 26 * mm, 88 * mm]))

    # 8. High priority
    S.append(_p('7. High-priority topics', h2))
    hp = [c for c in cl if c.get('importance') == 'HIGH']
    S.append(_p(', '.join(c['topic'] for c in hp) if hp else
                'No topic met the HIGH bar (3+ papers and at least half of the unique papers).', body))
    mp = [c['topic'] for c in cl if c.get('importance') == 'MEDIUM']
    lp = [c['topic'] for c in cl if c.get('importance') == 'LOW']
    if mp: S.append(_p('Medium: ' + ', '.join(mp), body))
    if lp: S.append(_p('Lower: ' + ', '.join(lp), body))

    # 9. Predictions
    S.append(_p('8. Predicted questions (estimates)', h2))
    S.append(_p(d.get('prediction_disclaimer', ''), small))
    for i, p in enumerate(d.get('predictions', []) or [], 1):
        S.append(_p(f"{i}. {p.get('question', '')}", ParagraphStyle('pq', parent=body, fontName='Helvetica-Bold')))
        S.append(_p(f"Likely {p.get('likely_position', '?')} · {p.get('likely_marks', '?')} marks · Historical frequency: "
                    f"{p.get('historical_frequency', '')}", small))
        if p.get('reason'): S.append(_p('Reason: ' + str(p.get('reason')), small))
        for r in (p.get('evidence') or [])[:4]:
            S.append(_p(f"   Evidence — {r.get('paper')} {r.get('position')}: {str(r.get('question', ''))[:160]}", small))
        S.append(Spacer(1, 3))
    bt = d.get('backtest') or {}
    S.append(_p('Validation: ' + (bt.get('note') if bt.get('available') else
                'Not validated. ' + str(bt.get('reason', 'No unseen paper available.'))), body))

    # 10. Study plan
    S.append(_p('9. Last-minute study plan', h2))
    lh = d.get('last_hour_prep', {}) or {}
    if lh.get('passing_strategy'): S.append(_p(lh['passing_strategy'], body))
    for m in lh.get('must_do', []) or []:
        S.append(_p(f"#{m.get('rank')} {m.get('topic')} ({m.get('marks')}m) — {m.get('question', '')[:180]}", cell))
        if m.get('quick_tip'): S.append(_p('   Tip: ' + m['quick_tip'], small))

    # 11. Limitations
    S.append(_p('10. Limitations and confidence', h2))
    for t in [
        'Frequencies count unique papers only; identical or near-identical uploads are removed before counting.',
        'Topic grouping uses keyword and text-similarity matching, not a trained semantic model; unusual rewordings can be missed.',
        'OCR can misread blurry or rotated scans. Items marked [illegible] or flagged above should be checked against the original.',
        'Historical frequency is not prediction accuracy. Predictions are estimates, not guaranteed questions.',
        'Validation is only reported when a separate unseen paper exists (see Validation above).',
    ]:
        S.append(_p('• ' + t, body))

    def footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 7)
        canvas.setFillColor(colors.HexColor('#64748b'))
        canvas.drawString(16 * mm, 9 * mm, 'ASHLYSIS — estimates based on historical papers; not guaranteed.')
        canvas.drawRightString(A4[0] - 16 * mm, 9 * mm, f'Page {_doc.page}')
        canvas.restoreState()

    doc.build(S, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
