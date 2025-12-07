import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import webbrowser
import urllib.parse

from rec import RecommendationSystem

DATA_FILE = "high_popularity_spotify_data.csv"

class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        # Keep a reference to the Canvas so we can recolor it in dark mode
        self.canvas = tk.Canvas(
            self,
            borderwidth=0,
            background="#f0f0f0",  # overridden by apply_theme()
            highlightthickness=0,
        )
        self.scrollbar = ttk.Scrollbar(
            self,
            orient="vertical",
            command=self.canvas.yview
        )

        self.scrollable_frame = ttk.Frame(self.canvas, padding="10")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

    # NEW → called by dark mode toggle
    def update_theme(self, bg):
        """Recolor the scrollable canvas background."""
        self.canvas.configure(bg=bg)
        self.scrollable_frame.configure(style="TFrame")

class RecommenderApp:
    def __init__(self, master):
        self.master = master
        master.title("VibeMatch: Smart Music Recommender")
        master.geometry("1100x850")
        master.configure(bg="#f0f0f0")

        self.style = self._setup_styles()

        try:
            self.model = RecommendationSystem(DATA_FILE)
        except FileNotFoundError as e:
            messagebox.showerror("Fatal Error", str(e))
            master.destroy()
            return

        self.k_value = tk.IntVar(value=10)
        self.search_term = tk.StringVar()
        self.selected_track_id = tk.StringVar(
            value=self.model.df.index[0] if not self.model.df.empty else ""
        )

        self.current_listener_name = tk.StringVar()
        self.selected_user_id = tk.StringVar()
        self.test_user_var = tk.StringVar()

        self.found_tracks = []

        self._setup_ui()
        self._initial_load()
        self._refresh_listener_lists()
        self.theme = "light"

        self.themes = {
            "light": {
                "bg": "#f0f0f0",
                "fg": "black",
                "text_bg": "white",
                "text_fg": "black",
                "frame_bg": "#f0f0f0",
                "button_bg": "#1DB954",
                "button_fg": "black",
            },
            "dark": {
                "bg": "#1e1e1e",
                "fg": "#dddddd",
                "text_bg": "#000000",
                "text_fg": "#00ff00",
                "frame_bg": "#0f0f0f",
                "button_bg": "#1DB954",
                "button_fg": "black",
            },
        }

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", background="#f0f0f0", font=("Arial", 10))
        style.configure(
            "Title.TLabel", font=("Arial", 18, "bold"), foreground="#1DB954"
        )
        style.configure("Header.TLabel", font=("Arial", 12, "bold"))
        style.configure("Subtitle.TLabel", font=("Arial", 10))
        style.configure(
            "TButton",
            font=("Arial", 10, "bold"),
            background="#1DB954",
            foreground="black",
        )
        style.map("TButton", background=[("active", "#1ED760")])
        style.configure("TNotebook.Tab", font=("Arial", 10, "bold"))
        return style

    def _setup_ui(self):
        title_frame = ttk.Frame(self.master, padding="10")
        title_frame.pack(fill="x")
        self.dark_mode_button = ttk.Button(
            title_frame,
            text="🌙 Dark Mode",
            command=self.toggle_dark_mode
        )
        self.dark_mode_button.pack(side="right")

        ttk.Label(
            title_frame,
            text="VibeMatch: Smart Music Recommender",
            style="Title.TLabel",
        ).pack(side="top", anchor="center", pady=(5, 2))

        ttk.Label(
            title_frame,
            text="Compare song-based and people-based recommendations with live listeners",
            style="Subtitle.TLabel",
        ).pack(side="top", anchor="center", pady=(0, 5))

        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(pady=10, padx=10, expand=True, fill="both")

        recommender_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(recommender_frame, text="Discover Music")
        self._setup_recommendation_tab(recommender_frame)

        evaluation_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(evaluation_frame, text="Quality & Comparison")
        self._setup_evaluation_tab(evaluation_frame)

        insights_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(insights_frame, text="Insights")
        self._setup_insights_tab(insights_frame)

    def toggle_dark_mode(self):
        self.theme = "dark" if self.theme == "light" else "light"
        self.apply_theme()

    def apply_theme(self):
        t = self.themes[self.theme]

        # Window background
        self.master.configure(bg=t["bg"])

        # ttk styles
        self.style.configure("TFrame", background=t["frame_bg"])
        self.style.configure("TLabel", background=t["frame_bg"], foreground=t["fg"])
        self.style.configure("Header.TLabel", background=t["frame_bg"], foreground=t["fg"])
        self.style.configure("Title.TLabel", background=t["frame_bg"], foreground="#1DB954")
        self.style.configure("Subtitle.TLabel", background=t["frame_bg"], foreground=t["fg"])

        self.style.configure(
            "TButton",
            background=t["button_bg"],
            foreground=t["button_fg"],
        )
        self.style.map("TButton", background=[("active", "#1ED760")])

        # Notebook background so the right side isn't white
        self.style.configure("TNotebook", background=t["frame_bg"])
        self.style.configure(
            "TNotebook.Tab",
            background=t["frame_bg"],
            foreground=t["fg"],
        )

        # Walk through all *tk* widgets (ttk are handled by style)
        def recolor(widget):
            cls = widget.winfo_class()

            if cls in ("Frame", "LabelFrame"):
                widget.configure(bg=t["frame_bg"])

            elif cls == "Label":
                widget.configure(bg=t["frame_bg"], fg=t["fg"])

            elif cls == "Entry":
                widget.configure(bg=t["text_bg"], fg=t["text_fg"])

            elif cls == "Text":
                widget.configure(
                    bg=t["text_bg"],
                    fg=t["text_fg"],
                    insertbackground=t["fg"],
                )

            elif cls == "Listbox":
                widget.configure(bg=t["text_bg"], fg=t["text_fg"])

            elif cls == "Canvas":
                # ScrollableFrame canvases etc.
                widget.configure(bg=t["frame_bg"], highlightbackground=t["frame_bg"])

            for child in widget.winfo_children():
                recolor(child)

        recolor(self.master)

        # ── Matplotlib figures ─────────────────────────────────────────
        bg = t["frame_bg"]
        fg = t["fg"]

        def restyle_figure(fig, ax, canvas):
            if fig is None or ax is None:
                return

            fig.patch.set_facecolor(bg)
            ax.set_facecolor(bg)

            ax.tick_params(colors=fg)
            for spine in ax.spines.values():
                spine.set_color(fg)

            ax.yaxis.label.set_color(fg)
            ax.xaxis.label.set_color(fg)
            ax.title.set_color(fg)

            if canvas is not None:
                canvas.draw_idle()

        # Main AP@10 bar chart
        if hasattr(self, "fig") and hasattr(self, "ax") and hasattr(self, "canvas"):
            restyle_figure(self.fig, self.ax, self.canvas)

        # Genre bar chart
        if (
                hasattr(self, "fig_genres")
                and hasattr(self, "ax_genres")
                and hasattr(self, "canvas_genres")
        ):
            restyle_figure(self.fig_genres, self.ax_genres, self.canvas_genres)

        # Dark / light mode button text
        self.dark_mode_button.configure(
            text="☀ Light Mode" if self.theme == "dark" else "🌙 Dark Mode"
        )

    def _apply_dark_to_widgets(self, bg, fg, darker):
        for widget in self.master.winfo_children():
            try:
                widget.configure(bg=bg, fg=fg)
            except:
                pass

        # Text widgets (manual)
        widgets = [
            self.results_text,
            self.eval_text,
            self.track_listbox,
            getattr(self, "likes_manager_listbox", None)
        ]

        for w in widgets:
            if w:
                try:
                    w.configure(bg=darker, fg=fg, insertbackground=fg)
                except:
                    pass

    def _setup_recommendation_tab(self, parent):
        main_paned_window = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        main_paned_window.pack(fill="both", expand=True)

        # LEFT side: controls in scrollable frame
        control_scroll = ScrollableFrame(main_paned_window)
        control_frame = control_scroll.scrollable_frame
        main_paned_window.add(control_scroll, weight=0)

        ttk.Label(
            control_frame,
            text="Listener setup",
            style="Header.TLabel",
        ).pack(pady=(5, 5), anchor="w")

        self.name_label = ttk.Label(
            control_frame, text="Who are we saving this like for?"
        )
        self.name_label.pack(anchor="w")
        self.listener_name_entry = ttk.Entry(
            control_frame, textvariable=self.current_listener_name, width=25
        )
        self.listener_name_entry.pack(anchor="w", pady=(0, 5))

        self.add_button = ttk.Button(
            control_frame,
            text="Save selected song as liked",
            command=self.add_song_to_listener,
            style="TButton",
        )
        self.add_button.pack(anchor="w", pady=(0, 10), fill="x")

        ttk.Label(
            control_frame,
            text="Recommendation mode",
            style="Header.TLabel",
        ).pack(pady=(10, 5), anchor="w")

        self.model_choice = tk.StringVar(value="Song-Based")

        ttk.Radiobutton(
            control_frame,
            text="Song-based: similar audio to a track",
            variable=self.model_choice,
            value="Song-Based",
            command=self._toggle_input_mode,
        ).pack(anchor="w")

        ttk.Radiobutton(
            control_frame,
            text="People-based: similar listeners like you",
            variable=self.model_choice,
            value="People-Based",
            command=self._toggle_input_mode,
        ).pack(anchor="w")

        ttk.Label(
            control_frame,
            text="\nHow many songs should we suggest?",
            style="Header.TLabel",
        ).pack(pady=(10, 5), anchor="w")

        self.k_entry = ttk.Entry(control_frame, textvariable=self.k_value, width=10)
        self.k_entry.pack(anchor="w", pady=(0, 10))

        self.cbf_input_frame = ttk.Frame(control_frame)
        self.cbf_input_frame.pack(fill="x")

        ttk.Label(
            self.cbf_input_frame,
            text="Pick a starting song",
            style="Header.TLabel",
        ).pack(pady=(10, 5), anchor="w")

        ttk.Label(self.cbf_input_frame, text="Search by title or artist:").pack(
            anchor="w"
        )
        self.search_entry = ttk.Entry(
            self.cbf_input_frame, textvariable=self.search_term, width=40
        )
        self.search_entry.pack(fill="x", pady=2)
        self.search_entry.bind("<KeyRelease>", self._search_tracks)

        ttk.Label(self.cbf_input_frame, text="Search results:").pack(
            anchor="w", pady=(5, 0)
        )
        self.track_listbox = tk.Listbox(
            self.cbf_input_frame,
            height=10,
            width=40,
            exportselection=0,
        )
        self.track_listbox.pack(fill="both", expand=True, pady=(0, 10))
        self.track_listbox.bind("<<ListboxSelect>>", self._on_track_select)

        # 🔊 New: play 30-second preview of the selected song
        self.preview_button = ttk.Button(
            control_frame,
            text="▶ Play 30s preview",
            command=self.play_preview,
            style="TButton",
        )
        self.preview_button.pack(pady=(0, 10), fill="x")

        self.cf_input_frame = ttk.Frame(control_frame)

        ttk.Label(
            self.cf_input_frame,
            text="Who are we recommending for?",
            style="Header.TLabel",
        ).pack(pady=(10, 5), anchor="w")

        self.user_selector = ttk.Combobox(
            self.cf_input_frame,
            textvariable=self.selected_user_id,
            values=[],
            state="readonly",
            width=30,
        )
        self.user_selector.pack(fill="x", pady=(0, 15))

        ttk.Button(
            self.cf_input_frame,
            text="Manage this listener's liked songs",
            command=self.open_liked_songs_manager,
            style="TButton",
        ).pack(fill="x", pady=(0, 10))

        self.view_likes_button = ttk.Button(
            self.cf_input_frame,
            text="View this listener's liked songs",
            command=self.show_liked_songs,
            style="TButton",
        )
        self.view_likes_button.pack(fill="x", pady=(0, 10))

        ttk.Button(
            control_frame,
            text="GET SUGGESTIONS",
            command=self.run_recommendation,
            style="TButton",
        ).pack(pady=(10, 10), fill="x")

        # RIGHT side: results
        results_frame = ttk.Frame(main_paned_window, padding="10")
        main_paned_window.add(results_frame, weight=1)

        ttk.Label(
            results_frame,
            text="Suggestions",
            style="Header.TLabel",
        ).pack(pady=(5, 10), anchor="w")

        self.results_text = tk.Text(
            results_frame,
            height=20,
            width=70,
            state=tk.DISABLED,
            wrap=tk.WORD,
            font=("Courier", 10),
        )
        self.results_text.pack(fill="both", expand=True)

        self.details_label = ttk.Label(
            results_frame,
            text="",
            font=("Arial", 10),
            justify="left",
        )
        self.details_label.pack(fill="x", pady=(10, 0), anchor="w")

        # For the "liked songs manager" window
        self.likes_manager_window = None
        self.likes_manager_listbox = None
        self.likes_manager_tracks = []
        self.likes_manager_listener = None

    def open_liked_songs_manager(self):
        """Open a small window showing this listener's liked songs and allow removals."""
        listener = self.user_selector.get().strip()
        if not listener:
            messagebox.showwarning(
                "No listener selected", "Please choose a listener first."
            )
            return

        df, error = self.model.get_liked_tracks_for_listener(listener)
        if error:
            messagebox.showerror("MongoDB error", error)
            return

        if df is None or df.empty:
            messagebox.showinfo(
                "No liked songs",
                f"{listener} has no liked songs stored yet.",
            )
            return

        if self.likes_manager_window is not None and self.likes_manager_window.winfo_exists():
            self.likes_manager_window.destroy()

        win = tk.Toplevel(self.master)
        win.title(f"{listener}'s liked songs")
        win.geometry("550x400")
        win.configure(bg="#f0f0f0")
        self.likes_manager_window = win
        self.likes_manager_listener = listener

        ttk.Label(
            win,
            text=f"Liked songs for {listener}",
            style="Header.TLabel",
        ).pack(pady=5, anchor="center")

        listbox = tk.Listbox(
            win,
            height=15,
            width=70,
            exportselection=False,
        )
        listbox.pack(fill="both", expand=True, padx=10, pady=5)
        self.likes_manager_listbox = listbox

        self.likes_manager_tracks = list(df.index)

        for tid, row in df.iterrows():
            display = f"{row['Track']} – {row['Artist']} ({row['playlist_genre']})"
            listbox.insert(tk.END, display)

        ttk.Button(
            win,
            text="Remove selected song from likes",
            command=self.remove_selected_like,
            style="TButton",
        ).pack(pady=10)

    def remove_selected_like(self):
        """Remove the selected song from the current listener's liked songs."""
        if not self.likes_manager_window or not self.likes_manager_listbox:
            return

        selection = self.likes_manager_listbox.curselection()
        if not selection:
            messagebox.showwarning(
                "No song selected",
                "Please select a song to remove from this listener's likes.",
            )
            return

        idx = selection[0]
        track_id = self.likes_manager_tracks[idx]
        listener = self.likes_manager_listener

        confirm = messagebox.askyesno(
            "Confirm removal",
            f"Remove this song from {listener}'s liked songs?",
        )
        if not confirm:
            return

        error = self.model.remove_like_for_listener(listener, track_id)
        if error:
            messagebox.showerror("MongoDB error", error)
            return

        self.likes_manager_listbox.delete(idx)
        del self.likes_manager_tracks[idx]

        matrix_error = self.model.build_user_item_matrix_from_mongo()
        if matrix_error:
            messagebox.showwarning("Matrix warning", matrix_error)

        self._refresh_listener_lists()

        if self.likes_manager_listbox.size() == 0:
            messagebox.showinfo(
                "No more liked songs",
                f"{listener} has no more liked songs.",
            )
            self.likes_manager_window.destroy()
            self.likes_manager_window = None

    def _setup_evaluation_tab(self, parent):
        eval_paned_window = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        eval_paned_window.pack(fill="both", expand=True)

        # LEFT: controls + text (with its own scrollbar)
        metrics_frame = ttk.Frame(eval_paned_window, width=320, padding="10")
        eval_paned_window.add(metrics_frame, weight=0)

        ttk.Label(
            metrics_frame,
            text="Evaluate recommendation quality",
            style="Header.TLabel",
        ).pack(pady=(10, 5), anchor="w")

        ttk.Label(
            metrics_frame,
            text="Which listener do we test the engines on?",
        ).pack(anchor="w")
        self.test_user_selector = ttk.Combobox(
            metrics_frame,
            textvariable=self.test_user_var,
            values=[],
            state="readonly",
            width=30,
        )
        self.test_user_selector.pack(fill="x", pady=(0, 15))

        ttk.Button(
            metrics_frame,
            text="Run quality comparison",
            command=self.run_ab_test,
            style="TButton",
        ).pack(pady=(10, 20), fill="x")

        ttk.Label(
            metrics_frame,
            text="Results:",
            style="Header.TLabel",
        ).pack(anchor="w")

        # Text + scrollbar container
        text_frame = ttk.Frame(metrics_frame)
        text_frame.pack(fill="both", expand=True, pady=(5, 10))

        text_scroll = ttk.Scrollbar(text_frame, orient="vertical")
        self.eval_text = tk.Text(
            text_frame,
            height=15,
            width=40,
            state=tk.DISABLED,
            wrap=tk.WORD,
            font=("Courier", 10),
            yscrollcommand=text_scroll.set,
        )
        text_scroll.config(command=self.eval_text.yview)
        self.eval_text.pack(side="left", fill="both", expand=True)
        text_scroll.pack(side="right", fill="y")

        # RIGHT: plots inside a scrollable frame
        plot_scroll = ScrollableFrame(eval_paned_window)
        plot_container = plot_scroll.scrollable_frame
        eval_paned_window.add(plot_scroll, weight=1)

        ttk.Label(
            plot_container,
            text="Song-based vs people-based score",
            style="Header.TLabel",
        ).pack(pady=(5, 10), anchor="w")

        # Bar chart: AP@10 scores
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.ax.set_title("Recommendation quality (AP@10)", fontsize=12)
        self.ax.set_ylim(0, 1.0)
        self.ax.set_ylabel("Match score")
        self.ax.set_xticks([0, 1])
        self.ax.set_xticklabels(["Song-based", "People-based"])
        self.ax.bar([0, 1], [0, 0])

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_container)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

        # Second plot: listener's favourite genres (blue)
        self.fig_genres, self.ax_genres = plt.subplots(figsize=(6, 3))
        self.ax_genres.set_title("Listener's top liked genres", fontsize=12)
        self.ax_genres.set_ylabel("Count")

        self.canvas_genres = FigureCanvasTkAgg(self.fig_genres, master=plot_container)
        self.canvas_genres_widget = self.canvas_genres.get_tk_widget()
        self.canvas_genres_widget.pack(fill="both", expand=True, pady=(10, 0))
    def _setup_insights_tab(self, parent):
        """
        Third tab: high-level insights about the dataset and listeners.
        Top row: popular genres + similarity distribution
        Bottom row: listener clusters + PCA scatter of songs
        """
        # Layout: two rows
        top_row = ttk.Frame(parent)
        bottom_row = ttk.Frame(parent)
        top_row.pack(side="top", fill="both", expand=True)
        bottom_row.pack(side="bottom", fill="both", expand=True)

        # --- Popular genres (all tracks) ---
        self.fig_popular_genres, self.ax_popular_genres = plt.subplots(figsize=(5, 3))
        self.canvas_popular_genres = FigureCanvasTkAgg(
            self.fig_popular_genres, master=top_row
        )
        self.canvas_popular_genres.get_tk_widget().pack(
            side="left", fill="both", expand=True, padx=5, pady=5
        )

        # --- Similarity score distribution (between songs) ---
        self.fig_similarities, self.ax_similarities = plt.subplots(figsize=(5, 3))
        self.canvas_similarities = FigureCanvasTkAgg(
            self.fig_similarities, master=top_row
        )
        self.canvas_similarities.get_tk_widget().pack(
            side="left", fill="both", expand=True, padx=5, pady=5
        )

        # --- Listener clusters (KMeans on liked-song profiles) ---
        self.fig_listener_clusters, self.ax_listener_clusters = plt.subplots(
            figsize=(5, 3)
        )
        self.canvas_listener_clusters = FigureCanvasTkAgg(
            self.fig_listener_clusters, master=bottom_row
        )
        self.canvas_listener_clusters.get_tk_widget().pack(
            side="left", fill="both", expand=True, padx=5, pady=5
        )

        # --- PCA scatter of songs (audio features) ---
        self.fig_pca_songs, self.ax_pca_songs = plt.subplots(figsize=(5, 3))
        self.canvas_pca_songs = FigureCanvasTkAgg(
            self.fig_pca_songs, master=bottom_row
        )
        self.canvas_pca_songs.get_tk_widget().pack(
            side="left", fill="both", expand=True, padx=5, pady=5
        )

        # Fill all 4 plots
        self._refresh_insights_plots()
    def _update_popular_genres_insight(self):
        self.ax_popular_genres.clear()

        if "playlist_genre" not in self.model.df.columns:
            self.ax_popular_genres.text(
                0.5, 0.5, "No genre info", ha="center", va="center",
                transform=self.ax_popular_genres.transAxes,
            )
        else:
            counts = self.model.df["playlist_genre"].value_counts().head(10)
            self.ax_popular_genres.bar(range(len(counts.index)), counts.values)
            self.ax_popular_genres.set_xticks(range(len(counts.index)))
            self.ax_popular_genres.set_xticklabels(
                counts.index, rotation=30, ha="right"
            )
            self.ax_popular_genres.set_title("Most popular genres (tracks dataset)")
            self.ax_popular_genres.set_ylabel("Number of tracks")

        self.fig_popular_genres.tight_layout()
        self.canvas_popular_genres.draw_idle()
    def _update_similarity_distribution_insight(self):
        self.ax_similarities.clear()

        # Use the numeric audio feature columns from the model
        feature_cols = getattr(self.model, "feature_cols", None)
        if feature_cols is None:
            self.ax_similarities.text(
                0.5, 0.5, "No feature columns found", ha="center", va="center",
                transform=self.ax_similarities.transAxes,
            )
            self.fig_similarities.tight_layout()
            self.canvas_similarities.draw_idle()
            return

        X = self.model.df[feature_cols].dropna().values

        # If there are too few songs, just display a message
        if X.shape[0] < 3:
            self.ax_similarities.text(
                0.5, 0.5, "Not enough songs for similarity histogram",
                ha="center", va="center", transform=self.ax_similarities.transAxes,
            )
            self.fig_similarities.tight_layout()
            self.canvas_similarities.draw_idle()
            return

        # Compute pairwise cosine similarities, sample some to keep it light
        sims = cosine_similarity(X)
        iu = np.triu_indices_from(sims, k=1)
        sim_values = sims[iu]

        # For very large datasets, subsample
        if sim_values.size > 20000:
            sim_values = np.random.choice(sim_values, size=20000, replace=False)

        self.ax_similarities.hist(sim_values, bins=30, range=(0.0, 1.0))
        self.ax_similarities.set_title("Distribution of song similarity scores")
        self.ax_similarities.set_xlabel("Cosine similarity")
        self.ax_similarities.set_ylabel("Number of pairs")

        self.fig_similarities.tight_layout()
        self.canvas_similarities.draw_idle()
    def _update_listener_clusters_insight(self):
        self.ax_listener_clusters.clear()

        # Build matrix from Mongo
        build_err = self.model.build_user_item_matrix_from_mongo()
        if build_err:
            self.ax_listener_clusters.text(
                0.5, 0.5,
                f"Cannot build listener matrix:\n{build_err}",
                ha="center", va="center",
                transform=self.ax_listener_clusters.transAxes,
            )
            self.fig_listener_clusters.tight_layout()
            self.canvas_listener_clusters.draw_idle()
            return

        uim = self.model.user_item_matrix
        if uim is None or uim.empty:
            self.ax_listener_clusters.text(
                0.5, 0.5, "No listeners with liked songs yet",
                ha="center", va="center",
                transform=self.ax_listener_clusters.transAxes,
            )
            self.fig_listener_clusters.tight_layout()
            self.canvas_listener_clusters.draw_idle()
            return

        feature_cols = getattr(self.model, "feature_cols", None)
        if feature_cols is None:
            self.ax_listener_clusters.text(
                0.5, 0.5, "No audio feature columns found",
                ha="center", va="center",
                transform=self.ax_listener_clusters.transAxes,
            )
            self.fig_listener_clusters.tight_layout()
            self.canvas_listener_clusters.draw_idle()
            return

        # For each listener, average the features of their liked songs
        song_features = self.model.df[feature_cols]
        listener_vectors = []
        listener_names = []

        for user_id, row in uim.iterrows():
            liked_track_ids = row[row > 0].index
            if len(liked_track_ids) == 0:
                continue
            common_ids = [tid for tid in liked_track_ids if tid in song_features.index]
            if not common_ids:
                continue
            avg_vec = song_features.loc[common_ids].mean(axis=0).values
            listener_vectors.append(avg_vec)
            listener_names.append(user_id)

        if not listener_vectors:
            self.ax_listener_clusters.text(
                0.5, 0.5, "Listeners have no matching tracks in dataset",
                ha="center", va="center",
                transform=self.ax_listener_clusters.transAxes,
            )
            self.fig_listener_clusters.tight_layout()
            self.canvas_listener_clusters.draw_idle()
            return

        X = np.vstack(listener_vectors)

        # Project to 2D with PCA, then cluster
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X)

        n_clusters = min(4, X_2d.shape[0])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(X_2d)

        scatter = self.ax_listener_clusters.scatter(
            X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10"
        )
        self.ax_listener_clusters.set_title("Listener clusters (KMeans on taste)")
        self.ax_listener_clusters.set_xlabel("PC1 (listener profile)")
        self.ax_listener_clusters.set_ylabel("PC2 (listener profile)")

        # optional small legend: cluster numbers
        handles, _ = scatter.legend_elements(num=n_clusters)
        self.ax_listener_clusters.legend(
            handles, [f"Cluster {i}" for i in range(n_clusters)],
            title="Clusters", loc="best",
        )

        self.fig_listener_clusters.tight_layout()
        self.canvas_listener_clusters.draw_idle()
    def _update_song_pca_insight(self):
        self.ax_pca_songs.clear()

        feature_cols = getattr(self.model, "feature_cols", None)
        if feature_cols is None:
            self.ax_pca_songs.text(
                0.5, 0.5, "No audio feature columns found",
                ha="center", va="center",
                transform=self.ax_pca_songs.transAxes,
            )
            self.fig_pca_songs.tight_layout()
            self.canvas_pca_songs.draw_idle()
            return

        df = self.model.df.dropna(subset=feature_cols)
        if df.shape[0] < 3:
            self.ax_pca_songs.text(
                0.5, 0.5, "Not enough songs for PCA",
                ha="center", va="center",
                transform=self.ax_pca_songs.transAxes,
            )
            self.fig_pca_songs.tight_layout()
            self.canvas_pca_songs.draw_idle()
            return

        # To keep it light, sample up to 1000 songs
        if df.shape[0] > 1000:
            df = df.sample(1000, random_state=42)

        X = df[feature_cols].values
        genres = df["playlist_genre"] if "playlist_genre" in df.columns else None

        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X)

        if genres is None:
            self.ax_pca_songs.scatter(X_2d[:, 0], X_2d[:, 1], s=10)
        else:
            # Color by genre (top few genres get separate colors, others merged)
            unique_genres = genres.value_counts().index.tolist()
            top_genres = unique_genres[:6]
            colors = {}
            cmap = plt.get_cmap("tab10")
            for i, g in enumerate(top_genres):
                colors[g] = cmap(i)
            colors["Other"] = "gray"

            mapped_colors = []
            for g in genres:
                if g in colors:
                    mapped_colors.append(colors[g])
                else:
                    mapped_colors.append(colors["Other"])

            self.ax_pca_songs.scatter(X_2d[:, 0], X_2d[:, 1], c=mapped_colors, s=10)

        self.ax_pca_songs.set_title("PCA scatter of songs (audio features)")
        self.ax_pca_songs.set_xlabel("PC1")
        self.ax_pca_songs.set_ylabel("PC2")

        self.fig_pca_songs.tight_layout()
        self.canvas_pca_songs.draw_idle()

    def _refresh_insights_plots(self):
        self._update_popular_genres_insight()
        self._update_similarity_distribution_insight()
        self._update_listener_clusters_insight()
        self._update_song_pca_insight()

    def _initial_load(self):
        self._search_tracks(None)
        if self.track_listbox.size() > 0:
            self.track_listbox.select_set(0)
            self.track_listbox.event_generate("<<ListboxSelect>>")
        self._toggle_input_mode()

    def _toggle_input_mode(self):
        mode = self.model_choice.get()

        if mode == "Song-Based":
            # Show TRACK INPUT (search bar) + name + like button
            if not self.name_label.winfo_ismapped():
                self.name_label.pack(anchor="w")
                self.listener_name_entry.pack(anchor="w", pady=(0, 5))
            if not self.add_button.winfo_ismapped():
                self.add_button.pack(anchor="w", pady=(0, 10), fill="x")

            self.cf_input_frame.pack_forget()
            self.cbf_input_frame.pack(fill="x")

        elif mode == "People-Based":
            # Hide like button + name entry, show listener selector
            if self.add_button.winfo_ismapped():
                self.add_button.pack_forget()
            if self.name_label.winfo_ismapped():
                self.name_label.pack_forget()
                self.listener_name_entry.pack_forget()

            self.cbf_input_frame.pack_forget()
            self.cf_input_frame.pack(fill="x")

    def _search_tracks(self, event):
        search = self.search_term.get().lower().strip()
        self.track_listbox.delete(0, tk.END)
        self.found_tracks = []

        if len(search) <= 1:
            filtered_df = self.model.df.head(50)
        else:
            filtered_df = self.model.df[
                self.model.df["Track"].str.lower().str.contains(search, na=False)
                | self.model.df["Artist"].str.lower().str.contains(search, na=False)
            ].head(100)

        for index, row in filtered_df.iterrows():
            display_text = f"{row['Track']} by {row['Artist']}"
            self.track_listbox.insert(tk.END, display_text)
            self.found_tracks.append(index)

    def _on_track_select(self, event):
        selected_indices = self.track_listbox.curselection()
        if selected_indices:
            idx = selected_indices[0]
            if 0 <= idx < len(self.found_tracks):
                track_id = self.found_tracks[idx]
                self.selected_track_id.set(track_id)

    def _update_results_text(self, title: str, recommendations, engine_label: str):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)

        header = f"{title}\n"
        header += f"Engine: {engine_label}\n\n"

        if recommendations is None or recommendations.empty:
            self.results_text.insert(tk.END, f"{header}No suggestions found.")
        else:
            formatted_table = []
            formatted_table.append(
                f"{'#':<3}{'Track Name':<40}{'Artist':<30}{'Genre':<15}"
            )
            formatted_table.append("-" * 90)

            for i, (_, row) in enumerate(recommendations.iterrows()):
                formatted_table.append(
                    f"{i + 1:<3}"
                    f"{row['Track'][:38]:<40}"
                    f"{row['Artist'][:28]:<30}"
                    f"{row['playlist_genre'][:13]:<15}"
                )

            self.results_text.insert(tk.END, header + "\n".join(formatted_table))

        self.results_text.config(state=tk.DISABLED)

    def _update_details_panel_for_song_based(self, seed_id: str):
        if not seed_id or seed_id not in self.model.df.index:
            self.details_label.config(text="")
            return

        track_row = self.model.df.loc[seed_id]
        details = (
            "Song-based mode\n"
            "We find tracks with similar audio characteristics.\n\n"
            "Starting song:\n"
            f"• Title : {track_row['Track']}\n"
            f"• Artist: {track_row['Artist']}\n"
            f"• Genre : {track_row['playlist_genre']}\n"
        )
        self.details_label.config(text=details)

    def _update_details_panel_for_people_based(self, user_id: str):
        if (
            self.model.user_item_matrix is None
            or user_id not in self.model.user_item_matrix.index
        ):
            self.details_label.config(text="")
            return

        user_likes = self.model.user_item_matrix.loc[user_id]
        n_likes = int((user_likes > 0).sum())
        details = (
            "People-based mode\n"
            "We look for listeners who like similar songs,\n"
            "then suggest tracks they also enjoy.\n\n"
            f"Personalized for: {user_id}\n"
            f"• Liked songs stored: {n_likes}\n"
        )
        self.details_label.config(text=details)

    def show_liked_songs(self):
        user = self.user_selector.get().strip()
        if not user:
            messagebox.showwarning(
                "Missing listener", "Please choose a listener from the list first."
            )
            return

        liked_df, error = self.model.get_liked_tracks_for_listener(user)

        if error:
            messagebox.showinfo("Liked songs", error)
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"{error}")
            self.results_text.config(state=tk.DISABLED)
            return

        self._update_results_text(
            f"{user}'s liked songs",
            liked_df,
            "Saved likes",
        )
        self._update_details_panel_for_people_based(user)

    def add_song_to_listener(self):
        name = self.current_listener_name.get().strip()
        if not name:
            messagebox.showwarning(
                "Missing name", "Please enter the listener's name first."
            )
            return

        track_id = self.selected_track_id.get()
        if not track_id:
            messagebox.showwarning(
                "Missing song", "Please select a song from the list."
            )
            return

        error = self.model.add_like_for_listener(name, track_id)
        if error:
            messagebox.showerror("Database error", error)
            return

        self._refresh_listener_lists()
        messagebox.showinfo(
            "Profile updated", f"{name}'s liked songs were updated in MongoDB."
        )

    def _refresh_listener_lists(self):
        names = self.model.get_listener_names()
        self.user_selector["values"] = names
        self.test_user_selector["values"] = names

        if names:
            if self.selected_user_id.get() not in names:
                self.selected_user_id.set(names[0])
            if self.test_user_var.get() not in names:
                self.test_user_var.set(names[0])

    def play_preview(self):
        """
        Try to play a 30s preview if we have a URL.
        Otherwise open a YouTube search for the song.
        """
        track_id = self.selected_track_id.get()
        if not track_id:
            messagebox.showwarning(
                "No song selected",
                "Please select a song first."
            )
            return

        # Get the row for this track
        try:
            row = self.model.df.loc[track_id]
        except KeyError:
            messagebox.showerror("Error", "Could not find this song in the dataset.")
            return

        # 1) Try to use preview_url if the column exists and is non-empty
        preview_url = None
        if "preview_url" in self.model.df.columns:
            val = row.get("preview_url", "")
            if isinstance(val, str) and val.strip():
                preview_url = val.strip()

        if preview_url:
            # Best case: dataset actually has a preview URL
            webbrowser.open(preview_url)
            return

        # 2) Fallback: open a YouTube search for the track
        track_name = str(row.get("Track", ""))
        artist_name = str(row.get("Artist", ""))
        if not track_name:
            messagebox.showinfo(
                "Preview unavailable",
                "Sorry, we don't have enough info to search for this track."
            )
            return

        query = f"{track_name} {artist_name} audio"
        url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(query)

        messagebox.showinfo(
            "Preview not in dataset",
            "This dataset doesn't provide a 30-second preview link for this song.\n\n"
            "We'll open a YouTube search for it instead."
        )
        webbrowser.open(url)

    def run_recommendation(self):
        try:
            k = int(self.k_value.get())
            if k <= 0 or k > 50:
                raise ValueError("K must be between 1 and 50.")
        except ValueError as e:
            messagebox.showwarning("Input Error", f"Invalid number of songs: {e}")
            return

        mode = self.model_choice.get()
        recommendations = None
        error = None

        if mode == "Song-Based":
            seed_id = self.selected_track_id.get()
            if not seed_id:
                messagebox.showwarning("Missing song", "Please pick a starting song.")
                return

            track_name = self.model.df.loc[seed_id]["Track"]
            recommendations, error = self.model.get_content_based_recommendations(
                seed_id, k
            )
            self._update_results_text(
                f"Songs similar to: {track_name}",
                recommendations,
                "Song-based (audio similarity)",
            )
            self._update_details_panel_for_song_based(seed_id)

        elif mode == "People-Based":
            if not self.model.get_listener_names():
                messagebox.showwarning(
                    "No listeners", "Please add at least one listener with liked songs."
                )
                return

            build_error = self.model.build_user_item_matrix_from_mongo()
            if build_error:
                messagebox.showwarning("Listener data problem", build_error)
                return

            user_id = self.user_selector.get().strip()
            if not user_id:
                messagebox.showwarning(
                    "Missing listener", "Please choose a listener from the list."
                )
                return

            recommendations, error = self.model.simulate_collaborative_filtering(
                user_id, k
            )
            self._update_results_text(
                f"Personalized mix for {user_id}",
                recommendations,
                "People-based (similar listeners)",
            )
            self._update_details_panel_for_people_based(user_id)

        if error:
            messagebox.showerror("Suggestions Error", error)
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Error: {error}")
            self.results_text.config(state=tk.DISABLED)

    def run_ab_test(self):
        if not self.model.get_listener_names():
            messagebox.showwarning(
                "No listeners", "Please add at least one listener with liked songs."
            )
            return

        build_error = self.model.build_user_item_matrix_from_mongo()
        if build_error:
            messagebox.showwarning("Listener data problem", build_error)
            return

        test_user = self.test_user_var.get().strip()
        if not test_user:
            messagebox.showwarning(
                "Missing listener", "Please pick a listener for the comparison."
            )
            return

        user_likes = self.model.user_item_matrix.loc[test_user][
            self.model.user_item_matrix.loc[test_user] > 0
            ]
        if user_likes.empty:
            messagebox.showwarning(
                "Evaluation Error",
                f"{test_user} has no liked songs.",
            )
            return

        k = 10
        cbf_seed_track_id = user_likes.index[0]

        # Evaluate song-based
        song_based_score, song_error = self.model.evaluate_model(
            self.model.get_content_based_recommendations,
            test_user_id=test_user,
            test_track_id=cbf_seed_track_id,
            k=k,
        )

        # Evaluate people-based
        people_based_score, people_error = self.model.evaluate_model(
            self.model.simulate_collaborative_filtering,
            test_user_id=test_user,
            k=k,
        )

        # Clear text box
        self.eval_text.config(state=tk.NORMAL)
        self.eval_text.delete(1.0, tk.END)

        # Header
        self.eval_text.insert(tk.END, f"Listener: {test_user}\n\n")
        self.eval_text.insert(
            tk.END,
            "We compare two engines:\n"
            "A = Song-based (similar audio features)\n"
            "B = People-based (learn from similar listeners)\n\n"
        )
        self.eval_text.insert(tk.END, "Score range: 0 (weak) to 1 (strong)\n\n")

        # Song-based score
        if song_error:
            self.eval_text.insert(tk.END, f"A: Song-based score: N/A ({song_error})\n")
            song_based_score = 0.0
        else:
            self.eval_text.insert(
                tk.END, f"A: Song-based score: {song_based_score:.4f}\n"
            )

        # People-based score
        if people_error:
            self.eval_text.insert(
                tk.END, f"B: People-based score: N/A ({people_error})\n"
            )
            people_based_score = 0.0
        else:
            self.eval_text.insert(
                tk.END, f"B: People-based score: {people_based_score:.4f}\n"
            )

        # Summary
        self.eval_text.insert(tk.END, "\nSummary:\n")
        if song_based_score > people_based_score:
            self.eval_text.insert(
                tk.END,
                "For this listener, the song-based engine matches their taste slightly better.\n",
            )
        elif people_based_score > song_based_score:
            self.eval_text.insert(
                tk.END,
                "For this listener, the people-based engine matches their taste slightly better.\n",
            )
        else:
            self.eval_text.insert(
                tk.END,
                "Both engines perform about the same for this listener.\n",
            )


        holdout_result, holdout_err = self.model.holdout_evaluation(test_user, k)

        if not holdout_err:
            self.eval_text.insert(tk.END, "\nExtra Metrics (Holdout Test):\n")
            self.eval_text.insert(
                tk.END,
                f"Precision@10: {holdout_result['precision']:.4f}\n"
            )
            self.eval_text.insert(
                tk.END,
                f"Recall@10: {holdout_result['recall']:.4f}\n"
            )
            self.eval_text.insert(
                tk.END,
                f"nDCG@10: {holdout_result['ndcg']:.4f}\n"
            )
            self.eval_text.insert(
                tk.END,
                f"Held-out test track: {holdout_result['held_out_track']} "
                f"(Genre: {holdout_result['test_genre']})\n"
            )
        else:
            self.eval_text.insert(
                tk.END,
                f"\nExtra Metrics: N/A ({holdout_err})\n"
            )
        # --------------------------------------------------------------

        self.eval_text.config(state=tk.DISABLED)

        # Update score bar chart
        self.ax.clear()
        scores = [song_based_score, people_based_score]
        colors = [
            "#1DB954" if scores[0] >= scores[1] else "#90EE90",
            "#1E90FF" if scores[1] >= scores[0] else "#ADD8E6",
        ]

        self.ax.bar(["Song-based", "People-based"], scores, color=colors)
        self.ax.set_title("Recommendation quality (AP@10)", fontsize=12)
        self.ax.set_ylim(0, 1.0)
        self.ax.set_ylabel("Match score")

        for i, score in enumerate(scores):
            self.ax.text(i, score + 0.02, f"{score:.4f}", ha="center", fontsize=10)

        self.fig.tight_layout()
        self.canvas.draw()

        # Existing genre plot update
        self._update_genre_plot(test_user)

    def _update_genre_plot(self, user_id: str):
        if (
            self.model.user_item_matrix is None
            or user_id not in self.model.user_item_matrix.index
        ):
            self.ax_genres.clear()
            self.ax_genres.set_title("Listener's top liked genres", fontsize=12)
            self.ax_genres.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=self.ax_genres.transAxes,
            )
            self.fig_genres.tight_layout()
            self.canvas_genres.draw()
            return

        user_likes = self.model.user_item_matrix.loc[user_id]
        liked_ids = user_likes[user_likes > 0].index.tolist()

        self.ax_genres.clear()
        self.ax_genres.set_title("Listener's top liked genres", fontsize=12)

        if not liked_ids:
            self.ax_genres.text(
                0.5,
                0.5,
                "This listener has no liked songs yet.",
                ha="center",
                va="center",
                transform=self.ax_genres.transAxes,
            )
        else:
            genre_counts = (
                self.model.df.loc[liked_ids, "playlist_genre"]
                .value_counts()
                .head(8)
            )

            self.ax_genres.bar(range(len(genre_counts.index)), genre_counts.values)
            self.ax_genres.set_ylabel("Count")

            self.ax_genres.set_xticks(range(len(genre_counts.index)))
            self.ax_genres.set_xticklabels(
                genre_counts.index, rotation=30, ha="right"
            )

        self.fig_genres.tight_layout()
        self.canvas_genres.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = RecommenderApp(root)
    root.mainloop()