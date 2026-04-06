"""
Boston Housing Price Predictor — GUI Application
=================================================
A premium Tkinter GUI for the Boston Housing ML project.
Features: Data exploration, visualizations, model training,
price prediction, and dataset statistics.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import (
    train_test_split, ShuffleSplit, learning_curve, validation_curve, GridSearchCV
)
from sklearn.metrics import r2_score, make_scorer


# ─── COLOUR PALETTE ─────────────────────────────────────────────────────────
BG_DARK       = "#0f1117"
BG_CARD       = "#1a1d28"
BG_INPUT      = "#252836"
FG_PRIMARY    = "#e4e6eb"
FG_SECONDARY  = "#8b8fa3"
ACCENT        = "#00d2ff"
ACCENT_DARK   = "#0097b2"
SUCCESS       = "#00e676"
WARNING       = "#ffab40"
ERROR         = "#ff5252"
BORDER        = "#2e3248"
HIGHLIGHT     = "#323750"


# ─── PERFORMANCE METRIC ─────────────────────────────────────────────────────
def performance_metric(y_true, y_predict):
    """R² score between true and predicted values."""
    return r2_score(y_true, y_predict)


# ─── HELPER: resolve data path ──────────────────────────────────────────────
def _resolve_data_path():
    """Try to find housing.csv relative to this script."""
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, "..", "..", "Data", "housing.csv"),
        os.path.join(base, "housing.csv"),
        os.path.join(base, "Data", "housing.csv"),
    ]
    for p in candidates:
        norm = os.path.normpath(p)
        if os.path.isfile(norm):
            return norm
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════
class BostonHousingApp(tk.Tk):
    """Root window for the Boston Housing Price Predictor."""

    def __init__(self):
        super().__init__()
        self.title("🏠 Boston Housing Price Predictor")
        self.geometry("1280x820")
        self.minsize(1000, 680)
        self.configure(bg=BG_DARK)

        # ── State ────────────────────────────────────────────────────────
        self.data: pd.DataFrame | None = None
        self.features: pd.DataFrame | None = None
        self.prices: pd.Series | None = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.best_model: DecisionTreeRegressor | None = None

        # ── Style ────────────────────────────────────────────────────────
        self._apply_theme()

        # ── Header ───────────────────────────────────────────────────────
        self._build_header()

        # ── Notebook ─────────────────────────────────────────────────────
        self.notebook = ttk.Notebook(self, style="Dark.TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        self._tab_explore  = DataExplorerTab(self.notebook, self)
        self._tab_viz      = VisualizationsTab(self.notebook, self)
        self._tab_train    = TrainModelTab(self.notebook, self)
        self._tab_predict  = PredictTab(self.notebook, self)
        self._tab_stats    = StatsTab(self.notebook, self)

        self.notebook.add(self._tab_explore, text="  📊 Data Explorer  ")
        self.notebook.add(self._tab_viz,     text="  📈 Visualizations  ")
        self.notebook.add(self._tab_train,   text="  ⚙️ Train Model  ")
        self.notebook.add(self._tab_predict, text="  🔮 Predict  ")
        self.notebook.add(self._tab_stats,   text="  📋 Statistics  ")

        # ── Status bar ───────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Ready — load a dataset to begin.")
        status = tk.Label(
            self, textvariable=self.status_var,
            bg=BG_CARD, fg=FG_SECONDARY, anchor="w",
            font=("Segoe UI", 9), padx=14, pady=4
        )
        status.pack(fill=tk.X, side=tk.BOTTOM)

        # ── Auto-load ───────────────────────────────────────────────────
        path = _resolve_data_path()
        if path:
            self._load_data(path)

    # ── theming ──────────────────────────────────────────────────────────
    def _apply_theme(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        # Notebook
        style.configure("Dark.TNotebook", background=BG_DARK, borderwidth=0)
        style.configure("Dark.TNotebook.Tab",
                        background=BG_CARD, foreground=FG_SECONDARY,
                        padding=[16, 8], font=("Segoe UI Semibold", 10))
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", HIGHLIGHT)],
                  foreground=[("selected", ACCENT)])

        # Frame
        style.configure("Card.TFrame", background=BG_CARD)
        style.configure("Dark.TFrame", background=BG_DARK)

        # Label
        style.configure("Card.TLabel", background=BG_CARD, foreground=FG_PRIMARY,
                        font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=BG_CARD, foreground=ACCENT,
                        font=("Segoe UI Semibold", 13))
        style.configure("Big.TLabel", background=BG_CARD, foreground=SUCCESS,
                        font=("Segoe UI Bold", 22))

        # Button
        style.configure("Accent.TButton",
                        background=ACCENT_DARK, foreground="white",
                        font=("Segoe UI Semibold", 10), padding=[18, 8])
        style.map("Accent.TButton",
                  background=[("active", ACCENT)])

        # Entry
        style.configure("Dark.TEntry",
                        fieldbackground=BG_INPUT, foreground=FG_PRIMARY,
                        insertcolor=FG_PRIMARY,
                        font=("Segoe UI", 10))

        # LabelFrame
        style.configure("Card.TLabelframe", background=BG_CARD,
                        foreground=ACCENT, font=("Segoe UI Semibold", 11))
        style.configure("Card.TLabelframe.Label", background=BG_CARD,
                        foreground=ACCENT, font=("Segoe UI Semibold", 11))

        # Progressbar
        style.configure("Accent.Horizontal.TProgressbar",
                        troughcolor=BG_INPUT, background=ACCENT,
                        thickness=6)

        # Scale
        style.configure("Dark.Horizontal.TScale",
                        background=BG_CARD, troughcolor=BG_INPUT)

    # ── header ───────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self, bg=BG_DARK, height=56)
        hdr.pack(fill=tk.X, padx=12, pady=(10, 6))
        tk.Label(hdr, text="🏠  Boston Housing Price Predictor",
                 bg=BG_DARK, fg=ACCENT, font=("Segoe UI Bold", 18)).pack(side=tk.LEFT)
        btn = ttk.Button(hdr, text="📂 Load CSV", style="Accent.TButton",
                         command=self._browse_csv)
        btn.pack(side=tk.RIGHT)

    # ── data handling ────────────────────────────────────────────────────
    def _browse_csv(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self._load_data(path)

    def _load_data(self, path: str):
        try:
            self.data = pd.read_csv(path)
            self.prices = self.data["MEDV"]
            self.features = self.data.drop("MEDV", axis=1)
            self.status_var.set(f"Loaded {os.path.basename(path)}  —  "
                                f"{self.data.shape[0]} rows × {self.data.shape[1]} cols")
            # notify tabs
            self._tab_explore.on_data_loaded()
            self._tab_stats.on_data_loaded()
            self._tab_viz.on_data_loaded()
            self._tab_train.on_data_loaded()
            self._tab_predict.on_data_loaded()
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load CSV:\n{exc}")

    def set_status(self, msg: str):
        self.status_var.set(msg)


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 1 — Data Explorer
# ═══════════════════════════════════════════════════════════════════════════
class DataExplorerTab(ttk.Frame):
    def __init__(self, parent, app: BostonHousingApp):
        super().__init__(parent, style="Dark.TFrame")
        self.app = app
        self._build_ui()

    def _build_ui(self):
        # Top info cards row
        self.info_frame = tk.Frame(self, bg=BG_DARK)
        self.info_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

        # Table area
        table_frame = tk.Frame(self, bg=BG_CARD, bd=0, highlightthickness=1,
                               highlightbackground=BORDER)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        cols_label = tk.Label(table_frame, text="Dataset Preview (first 15 rows)",
                              bg=BG_CARD, fg=ACCENT, font=("Segoe UI Semibold", 11),
                              anchor="w")
        cols_label.pack(fill=tk.X, padx=10, pady=(8, 2))

        # Treeview
        self.tree = ttk.Treeview(table_frame, style="Dark.Treeview")
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=(0, 10))
        vsb.pack(side=tk.RIGHT, fill=tk.Y, pady=(0, 10))

        # style the treeview rows
        s = ttk.Style()
        s.configure("Dark.Treeview",
                    background=BG_INPUT, foreground=FG_PRIMARY, fieldbackground=BG_INPUT,
                    font=("Segoe UI", 10), rowheight=26)
        s.configure("Dark.Treeview.Heading",
                    background=HIGHLIGHT, foreground=ACCENT,
                    font=("Segoe UI Semibold", 10))
        s.map("Dark.Treeview", background=[("selected", ACCENT_DARK)])

        # Outlier section
        out_frame = tk.Frame(self, bg=BG_CARD, bd=0, highlightthickness=1,
                             highlightbackground=BORDER)
        out_frame.pack(fill=tk.X, padx=8, pady=(4, 8))
        self.outlier_label = tk.Label(out_frame, text="Outlier Detection (IQR on MEDV): —",
                                     bg=BG_CARD, fg=WARNING,
                                     font=("Segoe UI Semibold", 11), anchor="w")
        self.outlier_label.pack(fill=tk.X, padx=10, pady=8)

    # ── helpers ──────────────────────────────────────────────────────────
    def _make_card(self, parent, title, value):
        card = tk.Frame(parent, bg=BG_CARD, bd=0, highlightthickness=1,
                        highlightbackground=BORDER, padx=16, pady=8)
        card.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        tk.Label(card, text=title, bg=BG_CARD, fg=FG_SECONDARY,
                 font=("Segoe UI", 9)).pack(anchor="w")
        tk.Label(card, text=str(value), bg=BG_CARD, fg=FG_PRIMARY,
                 font=("Segoe UI Semibold", 14)).pack(anchor="w")

    # ── event ────────────────────────────────────────────────────────────
    def on_data_loaded(self):
        data = self.app.data
        if data is None:
            return

        # clear old cards
        for w in self.info_frame.winfo_children():
            w.destroy()
        self._make_card(self.info_frame, "Rows", data.shape[0])
        self._make_card(self.info_frame, "Columns", data.shape[1])
        self._make_card(self.info_frame, "Missing values", int(data.isnull().sum().sum()))
        self._make_card(self.info_frame, "Data types", ", ".join(data.dtypes.unique().astype(str)))

        # populate treeview
        self.tree.delete(*self.tree.get_children())
        cols = list(data.columns)
        self.tree["columns"] = cols
        self.tree["show"] = "headings"
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=120, anchor="center")
        for _, row in data.head(15).iterrows():
            self.tree.insert("", "end", values=[f"{v:,.2f}" if isinstance(v, float) else str(v) for v in row])

        # outliers
        Q1 = data["MEDV"].quantile(0.25)
        Q3 = data["MEDV"].quantile(0.75)
        IQR = Q3 - Q1
        lb, ub = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        n_out = int(((data["MEDV"] < lb) | (data["MEDV"] > ub)).sum())
        self.outlier_label.config(
            text=f"Outlier Detection (IQR on MEDV):  {n_out} potential outliers  "
                 f"(lower={lb:,.0f}, upper={ub:,.0f})"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 2 — Visualizations
# ═══════════════════════════════════════════════════════════════════════════
class VisualizationsTab(ttk.Frame):
    def __init__(self, parent, app: BostonHousingApp):
        super().__init__(parent, style="Dark.TFrame")
        self.app = app
        self._build_ui()

    def _build_ui(self):
        btn_bar = tk.Frame(self, bg=BG_DARK)
        btn_bar.pack(fill=tk.X, padx=8, pady=8)

        ttk.Button(btn_bar, text="📦 Boxplot (Outliers)", style="Accent.TButton",
                   command=self._show_boxplot).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_bar, text="📈 Learning Curves", style="Accent.TButton",
                   command=self._show_learning).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_bar, text="📉 Complexity Curve", style="Accent.TButton",
                   command=self._show_complexity).pack(side=tk.LEFT, padx=4)

        self.canvas_frame = tk.Frame(self, bg=BG_CARD, bd=0,
                                     highlightthickness=1, highlightbackground=BORDER)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self.placeholder = tk.Label(self.canvas_frame,
                                    text="Load data and click a button above to render a chart.",
                                    bg=BG_CARD, fg=FG_SECONDARY,
                                    font=("Segoe UI", 12))
        self.placeholder.pack(expand=True)

    def _clear_canvas(self):
        for w in self.canvas_frame.winfo_children():
            w.destroy()

    def _embed_figure(self, fig):
        self._clear_canvas()
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ── charts ───────────────────────────────────────────────────────────
    def _show_boxplot(self):
        if self.app.data is None:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        fig = Figure(figsize=(9, 4), facecolor=BG_CARD)
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG_INPUT)
        bp = ax.boxplot(self.app.data["MEDV"], vert=False, patch_artist=True,
                        boxprops=dict(facecolor=ACCENT_DARK, color=ACCENT),
                        medianprops=dict(color=SUCCESS, linewidth=2),
                        whiskerprops=dict(color=FG_SECONDARY),
                        capprops=dict(color=FG_SECONDARY),
                        flierprops=dict(marker="o", markerfacecolor=ERROR, markersize=5,
                                        markeredgecolor=ERROR))
        ax.set_title("Detection of Outliers in MEDV (House Prices)",
                     color=FG_PRIMARY, fontsize=12, fontweight="bold")
        ax.set_xlabel("MEDV ($)", color=FG_SECONDARY)
        ax.tick_params(colors=FG_SECONDARY)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        fig.tight_layout()
        self._embed_figure(fig)

    def _show_learning(self):
        if self.app.features is None:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        self.app.set_status("Generating learning curves — this may take a moment…")
        self.update_idletasks()

        X, y = self.app.features, self.app.prices
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        train_sizes = np.rint(np.linspace(1, X.shape[0] * 0.8 - 1, 9)).astype(int)
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), facecolor=BG_CARD)

        for k, depth in enumerate([1, 3, 6, 10]):
            reg = DecisionTreeRegressor(max_depth=depth)
            sizes, tr_scores, te_scores = learning_curve(
                reg, X, y, cv=cv, train_sizes=train_sizes, scoring="r2"
            )
            tr_mean, tr_std = np.mean(tr_scores, axis=1), np.std(tr_scores, axis=1)
            te_mean, te_std = np.mean(te_scores, axis=1), np.std(te_scores, axis=1)
            ax = axes[k // 2, k % 2]
            ax.set_facecolor(BG_INPUT)
            ax.plot(sizes, tr_mean, "o-", color="#ff6b6b", label="Training Score")
            ax.plot(sizes, te_mean, "o-", color=SUCCESS, label="Testing Score")
            ax.fill_between(sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color="#ff6b6b")
            ax.fill_between(sizes, te_mean - te_std, te_mean + te_std, alpha=0.15, color=SUCCESS)
            ax.set_title(f"max_depth = {depth}", color=FG_PRIMARY, fontsize=10)
            ax.set_xlabel("Training Points", color=FG_SECONDARY, fontsize=8)
            ax.set_ylabel("Score", color=FG_SECONDARY, fontsize=8)
            ax.set_xlim([0, X.shape[0] * 0.8])
            ax.set_ylim([-0.05, 1.05])
            ax.legend(loc="lower right", fontsize=7)
            ax.tick_params(colors=FG_SECONDARY, labelsize=7)
            for spine in ax.spines.values():
                spine.set_color(BORDER)

        fig.suptitle("Decision Tree Regressor — Learning Curves",
                     fontsize=14, color=ACCENT, fontweight="bold", y=1.01)
        fig.tight_layout()
        self._embed_figure(fig)
        self.app.set_status("Learning curves rendered.")

    def _show_complexity(self):
        if self.app.features is None:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        self.app.set_status("Generating complexity curve…")
        self.update_idletasks()

        X, y = self.app.features, self.app.prices
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        max_depth = np.arange(1, 11)
        tr_scores, te_scores = validation_curve(
            DecisionTreeRegressor(), X, y,
            param_name="max_depth", param_range=max_depth, cv=cv, scoring="r2"
        )
        tr_mean, tr_std = np.mean(tr_scores, axis=1), np.std(tr_scores, axis=1)
        te_mean, te_std = np.mean(te_scores, axis=1), np.std(te_scores, axis=1)

        fig = Figure(figsize=(8, 5), facecolor=BG_CARD)
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG_INPUT)
        ax.plot(max_depth, tr_mean, "o-", color="#ff6b6b", label="Training Score")
        ax.plot(max_depth, te_mean, "o-", color=SUCCESS, label="Validation Score")
        ax.fill_between(max_depth, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color="#ff6b6b")
        ax.fill_between(max_depth, te_mean - te_std, te_mean + te_std, alpha=0.15, color=SUCCESS)
        ax.set_title("Decision Tree Regressor — Complexity Performance",
                     color=FG_PRIMARY, fontsize=12, fontweight="bold")
        ax.set_xlabel("Maximum Depth", color=FG_SECONDARY)
        ax.set_ylabel("Score", color=FG_SECONDARY)
        ax.set_ylim([-0.05, 1.05])
        ax.legend(loc="lower right")
        ax.tick_params(colors=FG_SECONDARY)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        fig.tight_layout()
        self._embed_figure(fig)
        self.app.set_status("Complexity curve rendered.")

    def on_data_loaded(self):
        pass  # charts are generated on-demand


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 3 — Train Model
# ═══════════════════════════════════════════════════════════════════════════
class TrainModelTab(ttk.Frame):
    def __init__(self, parent, app: BostonHousingApp):
        super().__init__(parent, style="Dark.TFrame")
        self.app = app
        self._build_ui()

    def _build_ui(self):
        wrapper = tk.Frame(self, bg=BG_DARK)
        wrapper.pack(expand=True)

        card = tk.Frame(wrapper, bg=BG_CARD, bd=0,
                        highlightthickness=1, highlightbackground=BORDER,
                        padx=30, pady=24)
        card.pack(padx=20, pady=20)

        tk.Label(card, text="⚙️  Model Training Configuration",
                 bg=BG_CARD, fg=ACCENT, font=("Segoe UI Bold", 15)).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 18))

        # Test size
        tk.Label(card, text="Test Size (%):", bg=BG_CARD, fg=FG_PRIMARY,
                 font=("Segoe UI", 11)).grid(row=1, column=0, sticky="w", pady=6)
        self.test_size_var = tk.IntVar(value=20)
        tk.Spinbox(card, from_=10, to=40, increment=5,
                   textvariable=self.test_size_var, width=6,
                   bg=BG_INPUT, fg=FG_PRIMARY, font=("Segoe UI", 11),
                   buttonbackground=BG_INPUT, insertbackground=FG_PRIMARY,
                   highlightthickness=0, bd=1, relief="flat").grid(
            row=1, column=1, sticky="w", padx=(10, 0), pady=6)

        # Random state
        tk.Label(card, text="Random State:", bg=BG_CARD, fg=FG_PRIMARY,
                 font=("Segoe UI", 11)).grid(row=2, column=0, sticky="w", pady=6)
        self.random_state_var = tk.IntVar(value=1)
        tk.Spinbox(card, from_=0, to=100,
                   textvariable=self.random_state_var, width=6,
                   bg=BG_INPUT, fg=FG_PRIMARY, font=("Segoe UI", 11),
                   buttonbackground=BG_INPUT, insertbackground=FG_PRIMARY,
                   highlightthickness=0, bd=1, relief="flat").grid(
            row=2, column=1, sticky="w", padx=(10, 0), pady=6)

        # Max depth range
        tk.Label(card, text="Max Depth Range:", bg=BG_CARD, fg=FG_PRIMARY,
                 font=("Segoe UI", 11)).grid(row=3, column=0, sticky="w", pady=6)
        tk.Label(card, text="1 – 10", bg=BG_CARD, fg=FG_SECONDARY,
                 font=("Segoe UI", 11)).grid(row=3, column=1, sticky="w", padx=(10, 0), pady=6)

        # Cross-validation
        tk.Label(card, text="Cross-Validation:", bg=BG_CARD, fg=FG_PRIMARY,
                 font=("Segoe UI", 11)).grid(row=4, column=0, sticky="w", pady=6)
        tk.Label(card, text="ShuffleSplit (10 splits, 20% test)",
                 bg=BG_CARD, fg=FG_SECONDARY,
                 font=("Segoe UI", 11)).grid(row=4, column=1, sticky="w", padx=(10, 0), pady=6)

        # Progress bar
        self.progress = ttk.Progressbar(card, mode="indeterminate", length=360,
                                        style="Accent.Horizontal.TProgressbar")
        self.progress.grid(row=5, column=0, columnspan=2, pady=(18, 6), sticky="ew")

        # Train button
        self.train_btn = ttk.Button(card, text="🚀  Train Model", style="Accent.TButton",
                                    command=self._train)
        self.train_btn.grid(row=6, column=0, columnspan=2, pady=(6, 14))

        # Results
        self.result_frame = tk.Frame(card, bg=BG_CARD)
        self.result_frame.grid(row=7, column=0, columnspan=2, sticky="ew")

    def _train(self):
        if self.app.features is None:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        self.train_btn.config(state="disabled")
        self.progress.start(12)
        self.app.set_status("Training model with GridSearchCV — please wait…")
        threading.Thread(target=self._do_train, daemon=True).start()

    def _do_train(self):
        try:
            ts = self.test_size_var.get() / 100.0
            rs = self.random_state_var.get()
            X, y = self.app.features, self.app.prices
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=ts, random_state=rs)
            self.app.X_train, self.app.X_test = X_train, X_test
            self.app.y_train, self.app.y_test = y_train, y_test

            cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)
            reg = DecisionTreeRegressor()
            params = {"max_depth": list(range(1, 11))}
            scoring_fnc = make_scorer(performance_metric)
            grid = GridSearchCV(reg, params, scoring=scoring_fnc, cv=cv)
            grid.fit(X_train, y_train)
            best = grid.best_estimator_
            self.app.best_model = best

            best_depth = best.get_params()["max_depth"]
            train_r2 = performance_metric(y_train, best.predict(X_train))
            test_r2 = performance_metric(y_test, best.predict(X_test))

            self.after(0, lambda: self._show_results(best_depth, train_r2, test_r2))
        except Exception as exc:
            self.after(0, lambda: messagebox.showerror("Training Error", str(exc)))
        finally:
            self.after(0, self._training_done)

    def _training_done(self):
        self.progress.stop()
        self.train_btn.config(state="normal")

    def _show_results(self, depth, train_r2, test_r2):
        for w in self.result_frame.winfo_children():
            w.destroy()

        self.app.set_status(f"✅ Training complete — best max_depth = {depth}")

        items = [
            ("Best max_depth", str(depth), ACCENT),
            ("Train R²", f"{train_r2:.4f}", SUCCESS if train_r2 > 0.7 else WARNING),
            ("Test R²", f"{test_r2:.4f}", SUCCESS if test_r2 > 0.7 else WARNING),
        ]
        for i, (lbl, val, clr) in enumerate(items):
            f = tk.Frame(self.result_frame, bg=BG_INPUT, padx=18, pady=10,
                         highlightthickness=1, highlightbackground=BORDER)
            f.grid(row=0, column=i, padx=6, pady=6, sticky="nsew")
            tk.Label(f, text=lbl, bg=BG_INPUT, fg=FG_SECONDARY,
                     font=("Segoe UI", 9)).pack()
            tk.Label(f, text=val, bg=BG_INPUT, fg=clr,
                     font=("Segoe UI Bold", 18)).pack()
        self.result_frame.columnconfigure((0, 1, 2), weight=1)


    def on_data_loaded(self):
        # clear any old results
        for w in self.result_frame.winfo_children():
            w.destroy()
        self.app.best_model = None


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 4 — Predict
# ═══════════════════════════════════════════════════════════════════════════
class PredictTab(ttk.Frame):
    def __init__(self, parent, app: BostonHousingApp):
        super().__init__(parent, style="Dark.TFrame")
        self.app = app
        self.client_entries: list[tuple[tk.Entry, tk.Entry, tk.Entry]] = []
        self._build_ui()

    def _build_ui(self):
        wrapper = tk.Frame(self, bg=BG_DARK)
        wrapper.pack(fill=tk.BOTH, expand=True, padx=20, pady=12)

        # Left: input card
        left = tk.Frame(wrapper, bg=BG_CARD, bd=0,
                        highlightthickness=1, highlightbackground=BORDER,
                        padx=24, pady=18)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        tk.Label(left, text="🔮  Client Feature Input",
                 bg=BG_CARD, fg=ACCENT, font=("Segoe UI Bold", 14)).pack(anchor="w", pady=(0, 12))

        header = tk.Frame(left, bg=BG_CARD)
        header.pack(fill=tk.X, pady=(0, 4))
        for i, h in enumerate(["", "RM (rooms)", "LSTAT (%)", "PTRATIO"]):
            tk.Label(header, text=h, bg=BG_CARD, fg=FG_SECONDARY,
                     font=("Segoe UI Semibold", 9), width=14).grid(row=0, column=i, padx=2)

        self.rows_frame = tk.Frame(left, bg=BG_CARD)
        self.rows_frame.pack(fill=tk.X)

        # Pre-fill 3 client rows with example data
        defaults = [(5, 17, 15), (4, 32, 22), (8, 3, 12)]
        for idx, (rm, ls, pt) in enumerate(defaults):
            self._add_client_row(idx + 1, rm, ls, pt)

        btn_row = tk.Frame(left, bg=BG_CARD)
        btn_row.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(btn_row, text="➕ Add Client", style="Accent.TButton",
                   command=lambda: self._add_client_row(len(self.client_entries) + 1)).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="💰 Predict Prices", style="Accent.TButton",
                   command=self._predict).pack(side=tk.LEFT, padx=4)

        # Right: results
        right = tk.Frame(wrapper, bg=BG_CARD, bd=0,
                         highlightthickness=1, highlightbackground=BORDER,
                         padx=24, pady=18)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

        tk.Label(right, text="💰  Predicted Prices",
                 bg=BG_CARD, fg=ACCENT, font=("Segoe UI Bold", 14)).pack(anchor="w", pady=(0, 12))

        self.results_frame = tk.Frame(right, bg=BG_CARD)
        self.results_frame.pack(fill=tk.BOTH, expand=True)
        self._result_placeholder()

    def _add_client_row(self, num, rm="", ls="", pt=""):
        row = tk.Frame(self.rows_frame, bg=BG_CARD)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text=f"Client {num}", bg=BG_CARD, fg=FG_PRIMARY,
                 font=("Segoe UI", 10), width=14).grid(row=0, column=0, padx=2)
        entries = []
        for j, default in enumerate([rm, ls, pt]):
            e = tk.Entry(row, width=14, bg=BG_INPUT, fg=FG_PRIMARY,
                         font=("Segoe UI", 10), insertbackground=FG_PRIMARY,
                         bd=1, relief="flat", highlightthickness=1,
                         highlightbackground=BORDER, highlightcolor=ACCENT)
            e.grid(row=0, column=j + 1, padx=2)
            if default != "":
                e.insert(0, str(default))
            entries.append(e)
        self.client_entries.append(tuple(entries))

    def _result_placeholder(self):
        tk.Label(self.results_frame,
                 text="Train a model, then click\n'Predict Prices' to see results.",
                 bg=BG_CARD, fg=FG_SECONDARY, font=("Segoe UI", 11),
                 justify="center").pack(expand=True)

    def _predict(self):
        if self.app.best_model is None:
            messagebox.showwarning("No Model", "Please train a model first (Train Model tab).")
            return
        client_data = []
        for i, (e_rm, e_ls, e_pt) in enumerate(self.client_entries):
            try:
                rm = float(e_rm.get())
                ls = float(e_ls.get())
                pt = float(e_pt.get())
                client_data.append([rm, ls, pt])
            except ValueError:
                messagebox.showerror("Input Error", f"Client {i + 1}: enter valid numeric values.")
                return

        preds = self.app.best_model.predict(client_data)

        for w in self.results_frame.winfo_children():
            w.destroy()

        for i, price in enumerate(preds):
            f = tk.Frame(self.results_frame, bg=BG_INPUT, padx=16, pady=12,
                         highlightthickness=1, highlightbackground=BORDER)
            f.pack(fill=tk.X, pady=4)
            tk.Label(f, text=f"Client {i + 1}", bg=BG_INPUT, fg=FG_SECONDARY,
                     font=("Segoe UI", 10)).pack(side=tk.LEFT)
            tk.Label(f, text=f"${price:,.2f}", bg=BG_INPUT, fg=SUCCESS,
                     font=("Segoe UI Bold", 16)).pack(side=tk.RIGHT)

        self.app.set_status(f"Predicted prices for {len(preds)} client(s).")

    def on_data_loaded(self):
        for w in self.results_frame.winfo_children():
            w.destroy()
        self._result_placeholder()


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 5 — Statistics
# ═══════════════════════════════════════════════════════════════════════════
class StatsTab(ttk.Frame):
    def __init__(self, parent, app: BostonHousingApp):
        super().__init__(parent, style="Dark.TFrame")
        self.app = app
        self._build_ui()

    def _build_ui(self):
        self.grid_frame = tk.Frame(self, bg=BG_DARK)
        self.grid_frame.pack(expand=True)
        self.placeholder = tk.Label(self.grid_frame,
                                    text="Load a dataset to view price statistics.",
                                    bg=BG_DARK, fg=FG_SECONDARY,
                                    font=("Segoe UI", 12))
        self.placeholder.pack(expand=True)

    def on_data_loaded(self):
        prices = self.app.prices
        if prices is None:
            return

        for w in self.grid_frame.winfo_children():
            w.destroy()

        tk.Label(self.grid_frame, text="📋  MEDV Price Statistics",
                 bg=BG_DARK, fg=ACCENT, font=("Segoe UI Bold", 16)).grid(
            row=0, column=0, columnspan=3, pady=(0, 20))

        stats = [
            ("Minimum", f"${prices.min():,.2f}", "#42a5f5"),
            ("Maximum", f"${prices.max():,.2f}", "#ef5350"),
            ("Mean", f"${prices.mean():,.2f}", "#66bb6a"),
            ("Median", f"${prices.median():,.2f}", "#ab47bc"),
            ("Std Dev", f"${prices.std():,.2f}", "#ffa726"),
            ("Mode", f"${prices.mode().iloc[0]:,.2f}", "#26c6da"),
        ]

        for idx, (title, value, color) in enumerate(stats):
            r, c = divmod(idx, 3)
            card = tk.Frame(self.grid_frame, bg=BG_CARD, padx=28, pady=18,
                            highlightthickness=2, highlightbackground=color)
            card.grid(row=r + 1, column=c, padx=10, pady=10, sticky="nsew")
            tk.Label(card, text=title, bg=BG_CARD, fg=FG_SECONDARY,
                     font=("Segoe UI", 10)).pack()
            tk.Label(card, text=value, bg=BG_CARD, fg=color,
                     font=("Segoe UI Bold", 20)).pack(pady=(4, 0))

        for c in range(3):
            self.grid_frame.columnconfigure(c, weight=1)


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    app = BostonHousingApp()
    app.mainloop()
