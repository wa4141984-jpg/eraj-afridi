# 📄 ATS Resume Checker

A Streamlit app that scores a resume for Applicant Tracking Systems (ATS) and suggests concrete improvements, powered by Google Gemini Flash.

## Features
- Upload a resume as **PDF, DOCX or TXT**
- Optional **job description** for a role-specific keyword match
- **ATS score out of 100** with a breakdown: Keywords, Content, Impact, Formatting
- Strengths, weaknesses, matched and missing keywords
- Prioritised improvements and **before/after bullet rewrites**
- Instant rule-based checks (email, phone, sections, length, metrics)
- Download the full report as Markdown

## How the score works
Gemini rates four areas (0-100 each). The overall score is calculated in code, not by the model, so it is consistent:

| Area | Weight |
|---|---|
| Keywords & Relevance | 30% |
| Content Quality | 25% |
| Impact & Achievements | 25% |
| Formatting & Parsability | 20% |

> No tool can reproduce the exact score of a specific employer's ATS. Treat this as a guide.

## Run locally
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Get a free API key at https://aistudio.google.com/apikey and either paste it in the sidebar or set it up as below.

### Configure the API key (recommended)
Create `.streamlit/secrets.toml` (already git-ignored):
```toml
GEMINI_API_KEY = "your-key-here"
# optional - defaults to the "gemini-flash-latest" alias
# GEMINI_MODEL = "gemini-flash-latest"
```
Or use environment variables: `GEMINI_API_KEY` and `GEMINI_MODEL`.

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (never commit your API key).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**.

## Troubleshooting
- **Model not found**: set `GEMINI_MODEL` (or the sidebar field) to a current Flash model name listed in Google AI Studio.
- **Could not extract enough text**: the PDF is probably a scanned image. Export a text-based PDF or upload a DOCX.
- **Rate limit / quota**: wait a minute, or check your quota in AI Studio.

## Privacy
Resume text is sent to the Gemini API for analysis and is not stored by this app.

## Project structure
```
app.py            # Streamlit UI + analysis logic
requirements.txt  # dependencies
README.md
```# eraj-afridi
