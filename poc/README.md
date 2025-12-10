# 🎧 AI-Powered Call Quality Auditor POC

An AI-driven solution for automated call quality auditing and customer sentiment analysis. This POC uses **OpenAI Whisper** for speech-to-text transcription and **GPT-3.5 Turbo** for sentiment analysis and quality scoring.

## ✨ Features

- **Audio Transcription**: Upload call recordings and get accurate transcriptions using OpenAI Whisper API
- **Sentiment Analysis**: Analyze customer emotions, urgency levels, and escalation risk
- **Agent Scoring**: Questionnaire-based evaluation of agent performance across multiple categories
- **Compliance Checking**: Automated fraud detection and compliance risk assessment
- **Interactive Dashboard**: Beautiful visualizations with Plotly charts
  - Sentiment distribution pie chart
  - Urgency level donut chart
  - Agent performance bar chart
  - Daily trends line chart
  - Category score breakdown
  - Escalation risk histogram

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python)
- **Transcription**: OpenAI Whisper API
- **AI Analysis**: GPT-3.5 Turbo
- **Charts**: Plotly.js
- **Database**: SQLite (default)
- **Frontend**: HTML/CSS/JavaScript with Jinja2 templates

## 📋 Prerequisites

- Python 3.9+
- OpenAI API Key

## 🚀 Quick Start

1. **Clone and navigate to the POC directory:**
   ```bash
   cd "Ai Sentiment Anaalysis/poc"
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

5. **Run the application:**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Open your browser:**
   Navigate to `http://localhost:8000`

## 📊 Dashboard Screenshots

The dashboard provides:

| Feature | Description |
|---------|-------------|
| **Summary Cards** | Total calls, average score, positive sentiment rate, escalation risk |
| **Sentiment Pie Chart** | Distribution of Positive/Neutral/Negative/Mixed sentiments |
| **Urgency Donut Chart** | High/Medium/Low urgency level distribution |
| **Agent Performance** | Bar chart showing agent scores comparison |
| **Daily Trends** | Line chart showing call volume and sentiment trends |
| **Category Scores** | Horizontal bar chart of scores by question category |

## 📝 API Endpoints

### Calls
- `POST /api/calls/upload` - Upload and analyze a call recording
- `GET /api/calls/` - List all analyzed calls
- `GET /api/calls/{call_id}` - Get detailed analysis for a specific call
- `DELETE /api/calls/{call_id}` - Delete a call analysis

### Dashboard
- `GET /api/dashboard/metrics` - Get aggregated dashboard metrics
- `GET /api/dashboard/charts/sentiment-pie` - Sentiment distribution data
- `GET /api/dashboard/charts/agent-performance` - Agent performance data
- `GET /api/dashboard/charts/daily-trends` - Daily trends data
- `GET /api/dashboard/charts/category-scores` - Category scores data
- `GET /api/dashboard/charts/urgency-distribution` - Urgency distribution data
- `GET /api/dashboard/charts/escalation-risk` - Escalation risk data

## 🎯 Evaluation Categories

The system evaluates calls across these categories:

1. **Call Opening** (10 points)
   - Customer name verification
   - Script adherence
   - Response timing
   - Language greeting

2. **Soft Skills** (16 points)
   - Helpfulness
   - Grammar & communication
   - Confidence
   - Empathy
   - Professional tone

3. **Probing & Understanding** (10 points)
   - Effective questioning
   - First-instance understanding
   - Diagnostic questions

4. **Problem Resolution** (14 points)
   - Information accuracy
   - Solution appropriateness
   - Objection handling

5. **Call Closing** (8 points)
   - Closing format
   - Call summarization
   - Further assistance offer

6. **Critical Parameters** (15 points)
   - No premature disconnection
   - Correct categorization

## 📁 Project Structure

```
poc/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database.py         # SQLAlchemy models
│   │   └── schemas.py          # Pydantic schemas
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── calls.py            # Call upload & analysis endpoints
│   │   └── dashboard.py        # Dashboard & charts endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── transcription.py    # Whisper API integration
│   │   └── sentiment_analysis.py  # GPT-3.5 analysis
│   ├── static/
│   │   └── css/
│   │       └── style.css       # Application styles
│   └── templates/
│       ├── base.html           # Base template
│       ├── dashboard.html      # Main dashboard
│       ├── upload.html         # Call upload page
│       └── call_detail.html    # Call analysis detail
├── uploads/                    # Uploaded audio files
├── .env.example
├── requirements.txt
└── README.md
```

## 🔒 Supported Audio Formats

- MP3
- WAV
- M4A
- WebM
- OGG

Maximum file size: 25MB (Whisper API limit)

## 💡 Tips for Best Results

1. **Audio Quality**: Use clear audio with minimal background noise
2. **File Size**: Keep recordings under 25MB
3. **Language**: The system auto-detects language, but English works best
4. **Duration**: Shorter calls (< 10 minutes) process faster

## 📄 License

This is a Proof of Concept (POC) for demonstration purposes.

## 🤝 Contact

For questions or support, refer to the original requirements document: `AI Powered Call Auditor.pdf`

