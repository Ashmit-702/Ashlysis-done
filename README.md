# ASHLYSIS — Powered by Groq (free tier)

## Render environment variables
- `GROQ_API_KEY` (first key), `GROQ_API_KEY_2`, `GROQ_API_KEY_3` ... up to `GROQ_API_KEY_20`
- or one variable `GROQ_API_KEYS` = comma-separated keys
- Optional: `GROQ_TEXT_MODEL` (default `openai/gpt-oss-120b`), `GROQ_VISION_MODEL` (default `qwen/qwen3.6-27b`)

## Deploy on Render

1. Push this folder to a GitHub repo
2. Render → New Web Service → connect repo
3. **Environment Variables** → Add:
   - Key: `GROQ_API_KEY`
   - Value: your Groq key (`gsk_...`)
4. Deploy → live in 3 minutes

## No cost to you, no cost to users. Ever.
