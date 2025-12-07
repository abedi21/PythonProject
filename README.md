# VibeMatch: Tkinter Music Recommendation Simulator

## Project Structure

- `main.py` – Tkinter GUI (tabs: Discover Music, Quality & Comparison, Insights)
- `rec.py` – RecommendationSystem class (dataset loading, recommenders, evaluation)
- `high_popularity_spotify_data.csv` – Spotify-like dataset used as song catalog
- `ReportHady.docx` – Final project report

## How to Run

1. Start a local MongoDB instance (optional but required for people-based mode).
2. Install dependencies:
   - `pandas`, `numpy`, `scikit-learn`, `pymongo`, `matplotlib`
3. Run the application:
   ```bash
   python main.py
