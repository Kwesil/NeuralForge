# NeuralForge
### Emotion-Aware Restaurant Intelligence
**DSN × BCT LLM Agent Challenge — Hackathon 3.0**

---

## Overview

NeuralForge is an emotion-aware restaurant recommendation and user modeling agent built for the DSN × BCT Hackathon 3.0. It addresses both competition tasks:

- **Task A — User Modeling:** Simulates contextually-aware, Nigerian-voiced restaurant reviews based on a user's detected emotional state using Gemini 2.0 Flash
- **Task B — Recommendation:** Delivers semantically-ranked, diversity-optimised restaurant recommendations driven by real-time facial emotion detection

The core idea: instead of treating users as static profiles, NeuralForge reads the user's emotional state in the moment and adapts both recommendations and review simulation to match their current behavioral context.

You can access it live here: https://neuralforge.streamlit.app/

---

## Architecture

```
Webcam / Manual Mood Input
        ↓
Emotion Detection (DeepFace — multi-frame averaging)
        ↓
Behavioral Context Mapping (comfort_seeking | stress_relief | social_exploration | balanced)
        ↓
Semantic Query Generation
        ↓
Embedding-Based Restaurant Ranking (all-MiniLM-L6-v2 + cosine similarity)
        ↓
Diversity Filtering (category deduplication across top-15 candidates)
        ↓
AI Review Simulation (Gemini 2.0 Flash — Nigerian voice, style variation)
        ↓
Streamlit UI (session behavioral memory, intent modeling, confidence scoring)
```

---

## Project Structure

```
neuralforge/
├── app.py                          # Main Streamlit application
├── emotion/
│   └── detect_emotion.py           # DeepFace multi-frame emotion detection
├── recommender/
│   ├── embedding_model.py          # SentenceTransformer embeddings
│   ├── ranker.py                   # Cosine similarity ranking
│   └── recommend.py                # Keyword-based filter (fallback)
├── review_generator/
│   └── review_agent.py             # Gemini 2.0 Flash review simulation
├── utils/
│   ├── context_mapper.py           # Emotion scores → behavioral context
│   ├── context_query.py            # Behavioral context → semantic query
│   └── load_data.py                # Yelp dataset loader
├── data/
│   └── yelp_academic_dataset_business.json
├── Dockerfile
├── requirements.txt
└── .env
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- A webcam (optional — manual mood input available)
- Gemini API key (free at [aistudio.google.com](https://aistudio.google.com))
- Yelp Academic Dataset (`yelp_academic_dataset_business.json`)

### Local Setup

**1. Clone the repository**
```bash
git clone https://github.com/Kwesil/NeuralForge.git
cd NeuralForge
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure environment variables**

Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

**4. Add the Yelp dataset**

Download the Yelp Academic Dataset and place `yelp_academic_dataset_business.json` inside the `data/` folder.

**5. Run the app**
```bash
streamlit run  app/app.py
```

Open `http://localhost:8501` in your browser.

---

## Docker Deployment

**Build the image**
```bash
docker build -t neuralforge .
```

**Run the container**
```bash
docker run -p 8501:8501 --env-file .env neuralforge
```

Open `http://localhost:8501` in your browser.

> **Note:** If the Yelp dataset is large, mount it as a volume instead:
> ```bash
> docker run -p 8501:8501 --env-file .env \
>   -v /path/to/your/data:/app/data neuralforge
> ```

---

## How It Works

### Emotion Detection
DeepFace analyzes 5 webcam frames and averages the emotion scores for stability. The dominant emotion is mapped to one of four behavioral contexts using threshold-based rules in `context_mapper.py`.

| Emotion Signal | Behavioral Context |
|---|---|
| Sad > 35% or Neutral > 40% with Sad > 20% | `comfort_seeking` |
| Angry > 25% or Fear > 25% | `stress_relief` |
| Happy > 45% or Surprise > 30% | `social_exploration` |
| Default | `balanced` |

### Recommendation (Task B)
Each behavioral context maps to a semantic query string. Restaurant descriptions are encoded using `all-MiniLM-L6-v2` and ranked by cosine similarity against the query embedding. The top 15 candidates are then filtered for category diversity, returning 5 varied recommendations.

### Review Simulation (Task A)
Gemini 2.0 Flash generates a 2–3 sentence Nigerian-English review with natural Pidgin blending. The prompt includes the user's emotional state, top 3 detected emotion scores, restaurant metadata, a random style variation angle, and a reasoning field explaining the emotional match. Outputs vary across scans via jittered emotion scores and randomised prompt angles.

### Session Memory
NeuralForge tracks behavioral contexts across scans within a session (rolling window of 10) and surfaces the user's dominant dining pattern — enabling multi-turn contextual awareness.

---

## Dataset

**Yelp Academic Dataset** — used under Yelp's academic dataset terms. Only businesses categorised as "Restaurants" are loaded. Features used: `name`, `city`, `stars`, `review_count`, `categories`.

> The dataset is not included in this repository. Download it from [yelp.com/dataset](https://www.yelp.com/dataset).

---

## Models & Libraries

| Component | Model / Library |
|---|---|
| Emotion Detection | DeepFace (FER-2013 backbone) |
| Text Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Review Generation | Gemini 2.0 Flash (Google AI) |
| Similarity Ranking | scikit-learn cosine similarity |
| Frontend | Streamlit |

---

## Acknowledgements

- [Yelp Academic Dataset](https://www.yelp.com/dataset)
- [DeepFace](https://github.com/serengil/deepface)
- [SentenceTransformers](https://www.sbert.net/)
- [Google Gemini](https://aistudio.google.com/)
- DSN × Bluechip Technologies for organizing Hackathon 3.0