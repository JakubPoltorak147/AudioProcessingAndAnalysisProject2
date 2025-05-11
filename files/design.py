from tkinter import ttk


class ColorScheme:
    MAIN_BG = "#E8F5E9"
    FRAME_BG = "#C8E6C9"
    ACCENT = "#43A047"
    WAVEFORM_COLOR = "#1B5E20"
    SILENCE_COLOR = "#FFECB3"
    VOICED_COLOR = "#B3E5FC"
    UNVOICED_COLOR = "#FFCDD2"

    # Dodatkowe kolory dla analizy częstotliwościowej
    ORIGINAL_SIGNAL = "#1565C0"  # Niebieski
    WINDOWED_SIGNAL = "#D32F2F"  # Czerwony
    F0_PEAK_COLOR = "#F44336"  # Jasny czerwony
    SPECTROGRAM_CMAP = "viridis"  # Kolorystyka spektrogramu


def configure_style(style: ttk.Style):
    style.theme_use("clam")
    style.configure("App.TFrame", background=ColorScheme.MAIN_BG)
    style.configure("Controls.TFrame", background=ColorScheme.FRAME_BG)
    style.configure("TLabel",
                    background=ColorScheme.MAIN_BG,
                    font=("Helvetica", 10))
    style.configure("TitleLabel.TLabel",
                    background=ColorScheme.MAIN_BG,
                    font=("Helvetica", 11, "bold"),
                    foreground=ColorScheme.ACCENT)

    style.configure("TButton",
                    font=("Helvetica", 9, "bold"),
                    padding=5)
    style.map("TButton",
              foreground=[("active", "#212121"), ("disabled", "#999999")],
              background=[("active", "#A5D6A7"), ("!active", "#81C784")])

    # Style dla Notebook (zakładek)
    style.configure("TNotebook", background=ColorScheme.MAIN_BG)
    style.configure("TNotebook.Tab",
                    background=ColorScheme.FRAME_BG,
                    padding=[10, 2],
                    font=("Helvetica", 9))
    style.map("TNotebook.Tab",
              background=[("selected", ColorScheme.ACCENT), ("active", "#A5D6A7")],
              foreground=[("selected", "#FFFFFF"), ("active", "#212121")])

    # Style dla ramek w analizie częstotliwościowej
    style.configure("Freq.TLabelframe",
                    background=ColorScheme.MAIN_BG)
    style.configure("Freq.TLabelframe.Label",
                    background=ColorScheme.MAIN_BG,
                    foreground=ColorScheme.ACCENT,
                    font=("Helvetica", 11, "bold"))