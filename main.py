# main.py

import tkinter as tk
from tkinter import ttk, messagebox

from recommender import MusicRecommender, FEATURE_COLS

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

CSV_PATH = "high_popularity_spotify_data.csv"


class MusicRecApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Music Recommendation System Builder - Spotify")
        self.root.geometry("1100x650")

        # state variables
        self.current_results = None   # last recommendations dataframe
        self.last_input_song = None   # for radar chart comparisons

        # Try to load the dataset + recommender
        try:
            self.rec = MusicRecommender(CSV_PATH)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset:\n{e}")
            self.root.destroy()
            return

        # Tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self._build_by_song_tab()
        self._build_by_artist_tab()
        self._build_by_favorites_tab()
        self._build_visualizations_tab()

        # Results table (bottom)
        self._build_results_table()

    # ---------- Tabs ----------

    def _build_by_song_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="By Song")

        ttk.Label(frame, text="Song name:").pack(anchor="w", padx=10, pady=5)
        self.song_entry = ttk.Entry(frame, width=40)
        self.song_entry.pack(anchor="w", padx=10)

        ttk.Button(
            frame,
            text="Recommend Similar Songs",
            command=self.on_recommend_by_song,
        ).pack(padx=10, pady=10)

        info = (
            "Tip: you can type part of the name,\n"
            "e.g., 'blinding lights' or 'dance monkey'."
        )
        ttk.Label(frame, text=info, foreground="gray").pack(anchor="w", padx=10, pady=5)

    def _build_by_artist_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="By Artist")

        ttk.Label(frame, text="Artist name:").pack(anchor="w", padx=10, pady=5)
        self.artist_entry = ttk.Entry(frame, width=40)
        self.artist_entry.pack(anchor="w", padx=10)

        ttk.Button(
            frame,
            text="Recommend Songs for Artist",
            command=self.on_recommend_by_artist,
        ).pack(padx=10, pady=10)

        ttk.Label(
            frame,
            text="Example: 'Taylor Swift', 'The Weeknd', 'Billie Eilish'...",
            foreground="gray",
        ).pack(anchor="w", padx=10, pady=5)

    def _build_by_favorites_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="By Favorites")

        ttk.Label(
            frame,
            text="Enter some favorite songs (one per line):",
        ).pack(anchor="w", padx=10, pady=5)

        self.fav_text = tk.Text(frame, height=8, width=50)
        self.fav_text.pack(anchor="w", padx=10)

        ttk.Button(
            frame,
            text="Recommend Based on Favorites",
            command=self.on_recommend_by_favorites,
        ).pack(padx=10, pady=10)

        ttk.Label(
            frame,
            text="Example:\nBlinding Lights\nbad guy\nShape of You",
            foreground="gray",
        ).pack(anchor="w", padx=10, pady=5)

    def _build_visualizations_tab(self):
        """Tab that shows radar charts & scatter plots."""
        self.vis_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.vis_tab, text="Visualizations")

        top = ttk.Frame(self.vis_tab)
        top.pack(side="top", fill="x", padx=10, pady=5)

        ttk.Label(
            top,
            text="Select a song in the results table, "
                 "then choose a visualization:",
        ).pack(anchor="w", pady=2)

        btn_frame = ttk.Frame(top)
        btn_frame.pack(anchor="w", pady=5)

        ttk.Button(
            btn_frame,
            text="Radar: Input Song vs Selected",
            command=self.plot_radar_input_vs_selected,
        ).pack(side="left", padx=5)

        ttk.Button(
            btn_frame,
            text="Energy vs Valence Scatter",
            command=self.plot_energy_valence_scatter,
        ).pack(side="left", padx=5)

        # Matplotlib figure + canvas
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas_frame = ttk.Frame(self.vis_tab)
        self.canvas_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

    def _build_results_table(self):
        bottom = ttk.Frame(self.root)
        bottom.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        columns = ("track_name", "track_artist", "genre")
        self.tree = ttk.Treeview(bottom, columns=columns, show="headings")

        self.tree.heading("track_name", text="Track")
        self.tree.heading("track_artist", text="Artist")
        self.tree.heading("genre", text="Genre")

        self.tree.column("track_name", width=400)
        self.tree.column("track_artist", width=250)
        self.tree.column("genre", width=150)

        self.tree.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(bottom, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

    # ---------- Helpers ----------

    def _update_results(self, df):
        # Clear table
        for row in self.tree.get_children():
            self.tree.delete(row)

        self.current_results = df  # store for later use in visualizations

        if df is None or len(df) == 0:
            return

        for _, row in df.iterrows():
            self.tree.insert(
                "",
                tk.END,
                values=(
                    row.get("track_name", ""),
                    row.get("track_artist", ""),
                    row.get("genre", ""),
                ),
            )

    def _get_selected_song_from_table(self):
        """Return (track_name, artist_name) of selected row in table."""
        selection = self.tree.selection()
        if not selection:
            return None, None

        item_id = selection[0]
        values = self.tree.item(item_id, "values")
        if len(values) < 2:
            return None, None
        track_name, artist_name = values[0], values[1]
        return track_name, artist_name

    # ---------- Visualization methods ----------

    def plot_radar_input_vs_selected(self):
        """Radar chart comparing input song vs selected recommended song."""
        # Need a last input song (from By Song tab)
        if not self.last_input_song:
            messagebox.showinfo(
                "Info",
                "Radar chart works after using the 'By Song' tab.\n"
                "Please first request recommendations by song.",
            )
            return

        sel_track, sel_artist = self._get_selected_song_from_table()
        if not sel_track:
            messagebox.showwarning(
                "No selection",
                "Please click on a recommended song in the table first.",
            )
            return

        # Find index of input song
        input_indices = self.rec._find_song_indices(self.last_input_song)
        if not input_indices:
            messagebox.showinfo(
                "Not found",
                "The original input song is no longer found in the dataset.",
            )
            return
        idx_input = input_indices[0]

        # Find index of selected song (track + artist match)
        df = self.rec.df
        mask = (df["track_name"] == sel_track) & (df["track_artist"] == sel_artist)
        if not mask.any():
            messagebox.showinfo(
                "Not found",
                "Selected song not found in the full dataset.",
            )
            return
        idx_sel = mask[mask].index[0]

        # Get standardized feature vectors (from rec.features)
        vec_input = self.rec.features.iloc[idx_input].values
        vec_sel = self.rec.features.iloc[idx_sel].values

        # Radar chart
        self.fig.clear()
        ax = self.fig.add_subplot(111, polar=True)

        num_vars = len(FEATURE_COLS)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))

        vals1 = np.concatenate((vec_input, [vec_input[0]]))
        vals2 = np.concatenate((vec_sel, [vec_sel[0]]))

        ax.plot(angles, vals1, label=f"Input: {self.last_input_song}", linewidth=2)
        ax.fill(angles, vals1, alpha=0.2)

        ax.plot(angles, vals2, label=f"Selected: {sel_track}", linewidth=2)
        ax.fill(angles, vals2, alpha=0.2)

        ax.set_thetagrids(angles[:-1] * 180 / np.pi, FEATURE_COLS, fontsize=8)
        ax.set_title("Audio Feature Profile (standardized)", fontsize=12)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

        self.canvas.draw()

    def plot_energy_valence_scatter(self):
        """Scatter plot of Energy vs Valence for full dataset, highlighting selected song."""
        df = self.rec.df

        sel_track, sel_artist = self._get_selected_song_from_table()

        self.fig.clear()
        ax = self.fig.add_subplot(111)

        ax.scatter(df["valence"], df["energy"], alpha=0.3, s=10)
        ax.set_xlabel("Valence (positivity of mood)")
        ax.set_ylabel("Energy")
        ax.set_title("Energy vs Valence for all songs")

        # highlight selected song if any
        if sel_track and sel_artist:
            mask = (df["track_name"] == sel_track) & (df["track_artist"] == sel_artist)
            if mask.any():
                row = df[mask].iloc[0]
                ax.scatter(
                    row["valence"],
                    row["energy"],
                    s=80,
                    edgecolors="red",
                    facecolors="none",
                    linewidths=2,
                    label=f"Selected: {sel_track}",
                )
                ax.legend()

        self.canvas.draw()

    # ---------- Button callbacks ----------

    def on_recommend_by_song(self):
        name = self.song_entry.get().strip()
        if not name:
            messagebox.showwarning("Input required", "Please enter a song name.")
            return

        res = self.rec.recommend_by_song(name, n_recs=10)
        if res is None:
            messagebox.showinfo(
                "No results", "Song not found in the dataset. Try another name."
            )
            return

        self.last_input_song = name  # store for radar chart
        self._update_results(res)

    def on_recommend_by_artist(self):
        artist = self.artist_entry.get().strip()
        if not artist:
            messagebox.showwarning("Input required", "Please enter an artist name.")
            return

        res = self.rec.recommend_by_artist(artist, n_recs=10)
        if res is None:
            messagebox.showinfo(
                "No results", "Artist not found in the dataset. Try another name."
            )
            return

        # no specific last_input_song in this case
        self._update_results(res)

    def on_recommend_by_favorites(self):
        text = self.fav_text.get("1.0", "end").strip()
        favs = [line.strip() for line in text.splitlines() if line.strip()]
        if not favs:
            messagebox.showwarning(
                "Input required",
                "Please enter at least one favorite song (one per line).",
            )
            return

        res = self.rec.recommend_by_favorites(favs, n_recs=10)
        if res is None:
            messagebox.showinfo(
                "No results",
                "None of the favorite songs were found in the dataset.\n"
                "Try typing their exact names as they appear on Spotify.",
            )
            return

        # last_input_song not a single song here
        self._update_results(res)


if __name__ == "__main__":
    root = tk.Tk()
    app = MusicRecApp(root)
    root.mainloop()
