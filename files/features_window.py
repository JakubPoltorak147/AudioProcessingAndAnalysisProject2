import tkinter as tk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from features import (
    compute_volume, compute_ste, compute_zcr, compute_sr,
    compute_autocorr_f0, compute_amdf_f0
)
from design import ColorScheme

def auto_frame_size(total_samples, max_frames=2000):

    frame_size = max(256, int(np.ceil(total_samples / max_frames)))
    return frame_size

def downsample_block(x, y, max_points=2000):

    n = len(x)
    if n <= max_points:
        return x, y
    block_size = int(np.ceil(n / max_points))
    # Podziel dane na bloki
    x_blocks = [x[i*block_size : (i+1)*block_size] for i in range(int(np.ceil(n / block_size)))]
    y_blocks = [y[i*block_size : (i+1)*block_size] for i in range(int(np.ceil(n / block_size)))]
    x_ds = np.array([np.mean(block) for block in x_blocks])
    y_ds = np.array([np.mean(block) for block in y_blocks])
    return x_ds, y_ds

class FeaturesWindow:
    def __init__(self, master, data, fs, frame_size, silence_threshold):
        self.top = tk.Toplevel(master)
        self.top.title("Wykresy cech sygnału")
        self.top.geometry("1000x800")

        # Pastelowe tło
        self.main_bg_color = ColorScheme.MAIN_BG
        self.top.configure(bg=self.main_bg_color)

        # Główna ramka
        self.main_frame = tk.Frame(self.top, bg=self.main_bg_color)
        self.main_frame.pack(fill="both", expand=True)

        self.data = data
        self.fs = fs

        # Automatyczne dobranie rozmiaru ramki – dla długich nagrań zwiększamy ją, aby liczba ramek nie była zbyt duża.
        candidate = auto_frame_size(len(data))
        self.frame_size = min(candidate, frame_size)  # wybieramy większą z tych wartości
        self.silence_threshold = silence_threshold

        # Dzielimy sygnał na ramki i obliczamy cechy
        self.frames, self.times = self.frame_signal(data, fs, self.frame_size)
        self.volume = np.array([compute_volume(f) for f in self.frames])
        self.ste = np.array([compute_ste(f) for f in self.frames])
        self.zcr = np.array([compute_zcr(f) for f in self.frames])
        self.sr = np.array([compute_sr(f) for f in self.frames])
        self.f0_autocorr = np.array([compute_autocorr_f0(f, fs) for f in self.frames])
        self.f0_amdf = np.array([compute_amdf_f0(f, fs) for f in self.frames])

        # Przechowujemy cechy
        self.features_info = {
            "Volume (RMS)": (
                self.volume,
                "Volume określa średnią głośność sygnału (RMS).",
                "#4DB6AC"
            ),
            "STE": (
                self.ste,
                "Short Time Energy – rozróżnianie fragmentów dźwięcznych/bezdźwięcznych.",
                "#81C784"
            ),
            "ZCR": (
                self.zcr,
                "Zero Crossing Rate – liczba przejść przez zero.",
                "#FFF176"
            ),
            "SR (Silent Ratio)": (
                self.sr,
                "1 oznacza ramkę sklasyfikowaną jako cisza.",
                "#FFD54F"
            ),
            "F0 (Autocorr)": (
                self.f0_autocorr,
                "Częstotliwość podstawowa - metoda autokorelacji.",
                "#BA68C8"
            ),
            "F0 (AMDF)": (
                self.f0_amdf,
                "Częstotliwość podstawowa - metoda AMDF.",
                "#FF8A65"
            ),
        }

        # Panel wyboru cech
        self.select_frame = tk.Frame(self.main_frame, bg=self.main_bg_color)
        self.select_frame.pack(side="top", fill="x", padx=10, pady=10)

        tk.Label(
            self.select_frame,
            text="Wybierz cechy do wyświetlenia:",
            bg=self.main_bg_color,
            fg="#1B5E20",
            font=("Helvetica", 11, "bold")
        ).pack(side="left", anchor="n")

        self.feature_vars = {}
        for feat_name in self.features_info.keys():
            var = tk.BooleanVar(value=True)
            cb = tk.Checkbutton(
                self.select_frame,
                text=feat_name,
                variable=var,
                bg=self.main_bg_color,
                highlightthickness=0,
                font=("Helvetica", 9)
            )
            cb.pack(side="left", padx=5)
            self.feature_vars[feat_name] = var

        self.draw_button = tk.Button(
            self.select_frame,
            text="Rysuj",
            command=self.draw_selected_features,
            bg="#B2DFDB",
            fg="#004D40",
            activebackground="#80CBC4",
            font=("Helvetica", 9, "bold")
        )
        self.draw_button.pack(side="left", padx=10)

        self.plot_frame = tk.Frame(self.main_frame, bg=self.main_bg_color)
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.fig = None
        self.canvas = None

        # Rysujemy wykresy przy starcie
        self.draw_selected_features()

    def frame_signal(self, data, fs, frame_size):
        total_samples = len(data)
        num_frames = int(np.ceil(total_samples / frame_size))
        frames = []
        times = []
        for i in range(num_frames):
            start = i * frame_size
            end = start + frame_size
            frame = data[start:end]
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)), mode='constant')
            frames.append(frame)
            times.append(start / fs)
        return frames, np.array(times)

    def draw_selected_features(self):
        selected_features = [name for name, var in self.feature_vars.items() if var.get()]

        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.fig:
            plt.close(self.fig)
            self.fig = None

        if not selected_features:
            info_label = tk.Label(
                self.plot_frame,
                text="Nie wybrano żadnych cech do wyświetlenia!",
                bg=self.main_bg_color,
                fg="#B71C1C",
                font=("Helvetica", 12, "bold")
            )
            info_label.pack()
            return

        self.fig = plt.Figure(figsize=(12, 6), dpi=100)
        self.fig.set_tight_layout(True)

        n_feats = len(selected_features)
        rows, cols = self.calc_subplot_grid(n_feats)

        # Dla każdej cechy agregujemy dane, by rysować mniej punktów
        for i, feat_name in enumerate(selected_features, start=1):
            ax = self.fig.add_subplot(rows, cols, i)
            data_array, description, color_line = self.features_info[feat_name]
            # Używamy funkcji downsample_block, by zredukować liczbę punktów
            x_plot, y_plot = downsample_block(self.times, data_array, max_points=2000)
            ax.plot(x_plot, y_plot, linewidth=1.0, color=color_line, rasterized=True)
            ax.set_title(feat_name, fontsize=10, fontweight="bold", color="#2E7D32")
            ax.set_xlabel("Czas [s]", fontsize=9)
            ax.set_ylabel(feat_name, fontsize=9)
            ax.grid(True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def calc_subplot_grid(self, n):
        if n == 1:
            return (1, 1)
        elif n == 2:
            return (1, 2)
        elif n == 3:
            return (1, 3)
        elif n == 4:
            return (2, 2)
        elif n <= 6:
            return (2, 3)
        else:
            return (2, 3)
