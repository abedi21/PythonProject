# recommender.py

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# We’ll use these numeric columns as audio features
FEATURE_COLS = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "loudness",
    "speechiness",
    "liveness",
    "acousticness",
    "instrumentalness",
]

class MusicRecommender:
    def __init__(self, csv_path: str):
        # Load your Spotify data
        self.df = pd.read_csv(csv_path)

        # Make a simple "genre" column from playlist_genre if available
        if "playlist_genre" in self.df.columns:
            self.df["genre"] = self.df["playlist_genre"]
        else:
            self.df["genre"] = ""

        # Keep only rows that have all feature values
        self.df = self.df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

        # Store a clean version of the feature matrix
        self.features = self.df[FEATURE_COLS].copy()

        # Standardize features (helps cosine similarity)
        self.features = (self.features - self.features.mean()) / self.features.std()

        # Precompute cosine similarity matrix between all songs
        self.sim_matrix = cosine_similarity(self.features.values)

    # ---------- helpers ----------

    def _find_song_indices(self, track_name: str):
        """
        Return indices of songs whose name matches the input.
        First try exact (case-insensitive), then contains.
        """
        name = track_name.strip().lower()
        if not name:
            return []

        exact = self.df[self.df["track_name"].str.lower() == name]
        if not exact.empty:
            return exact.index.tolist()

        contains = self.df[self.df["track_name"].str.lower().str.contains(name)]
        return contains.index.tolist()

    def _find_artist_indices(self, artist_name: str):
        """
        Return indices of songs whose artist matches the input.
        """
        name = artist_name.strip().lower()
        if not name:
            return []

        exact = self.df[self.df["track_artist"].str.lower() == name]
        if not exact.empty:
            return exact.index.tolist()

        contains = self.df[self.df["track_artist"].str.lower().str.contains(name)]
        return contains.index.tolist()

    # ---------- main methods ----------

    def recommend_by_song(self, track_name: str, n_recs: int = 10):
        indices = self._find_song_indices(track_name)
        if not indices:
            return None  # means “not found”

        # Take the first match as the “reference” song
        idx = indices[0]
        sim_scores = self.sim_matrix[idx]

        # Get indices of most similar songs (excluding the song itself)
        ranked_idx = np.argsort(sim_scores)[::-1]
        ranked_idx = [i for i in ranked_idx if i != idx][:n_recs]

        return self.df.loc[ranked_idx]

    def recommend_by_artist(self, artist_name: str, n_recs: int = 10):
        artist_indices = self._find_artist_indices(artist_name)
        if not artist_indices:
            return None

        # Build an “artist profile” = average of all that artist's songs
        artist_vector = self.features.iloc[artist_indices].mean(axis=0).values.reshape(1, -1)

        # Similarity of all songs to that artist profile
        all_features = self.features.values
        sim_scores = cosine_similarity(artist_vector, all_features)[0]

        # Exclude songs by the same artist (optional)
        exclude = set(artist_indices)
        ranked_idx = np.argsort(sim_scores)[::-1]
        filtered_idx = [i for i in ranked_idx if i not in exclude][:n_recs]

        return self.df.loc[filtered_idx]

    def recommend_by_favorites(self, track_names, n_recs: int = 10):
        """
        track_names: list of song names the user likes.
        Build a “user taste” profile = average of those songs' features.
        """
        fav_indices = []
        for name in track_names:
            idx_list = self._find_song_indices(name)
            if idx_list:
                fav_indices.extend(idx_list[:1])  # take first match of each

        if not fav_indices:
            return None

        user_vector = self.features.iloc[fav_indices].mean(axis=0).values.reshape(1, -1)
        all_features = self.features.values
        sim_scores = cosine_similarity(user_vector, all_features)[0]

        exclude = set(fav_indices)
        ranked_idx = np.argsort(sim_scores)[::-1]
        filtered_idx = [i for i in ranked_idx if i not in exclude][:n_recs]

        return self.df.loc[filtered_idx]
