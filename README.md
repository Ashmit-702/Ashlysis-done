# ASHLYSIS — Powered by Groq (free tier)

## Render environment variables (both AI providers are optional)
- `GROQ_API_KEY` (+ optional `GROQ_API_KEY_2`, `_3`... or `GROQ_API_KEYS=a,b`) — Groq text + vision
- `GEMINI_API_KEY` — Gemini PDF understanding + text
- Optional model overrides: `GROQ_TEXT_MODEL`, `GROQ_VISION_MODEL`, `GEMINI_MODEL`

Local run: `pip install -r requirements.txt && cp .env.example .env` (fill keys, export them) then `python app.py`.
Tests: `python tests/test_ashlysis.py` (AI calls are mocked).

## Deploy on Render

1. Push this folder to a GitHub repo
2. Render → New Web Service → connect repo
3. **Environment Variables** → Add:
   - Key: `GROQ_API_KEY`
   - Value: your Groq key (`gsk_...`)
4. Deploy → live in 3 minutes

## No cost to you, no cost to users. Ever.
