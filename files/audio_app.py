import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import sounddevice as sd
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Patch
from scipy.io import wavfile
import sys
import warnings
from scipy.io.wavfile import WavFileWarning
import os

from design import ColorScheme, configure_style
from audio_processing import VoicedAudioProcessor
from features_window import FeaturesWindow
# Dodajemy import klas analizy częstotliwościowej
from frequency_analysis import FrequencyAnalysisWindow, CepstrumAnalysisWindow

warnings.simplefilter("ignore", WavFileWarning)


class AudioApp:
    def __init__(self, master):
        self.master = master

        # --- Konfigurujemy styl ---
        self.style = ttk.Style()
        configure_style(self.style)

        # Klasa do przetwarzania audio (analizy ciszy, dźwięczności itd.)
        self.processor = VoicedAudioProcessor()

        self.master.title("Aplikacja Audio")
        self.master.geometry("900x700")

        # Zmienne audio
        self.fs = None
        self.data = None
        self.total_samples = 0
        self.time_array = None
        self.current_index = 0
        self.filename = ""

        # Flagi sterowania
        self.playing = False
        self.paused = False

        # Strumień audio w sounddevice
        self.stream = None

        # Parametry analizy
        self.silence_threshold = 0.001
        self.frame_size = 256

        # Zmienna do wyboru trybu podświetlania
        self.highlight_mode = tk.StringVar(value="silence")  # domyślnie "silence"

        # Referencja do pionowej linii
        self.line = None

        # Bliting – statyczne tło wykresu
        self.background = None

        # Główna ramka
        self.main_frame = ttk.Frame(self.master, style="App.TFrame")
        self.main_frame.pack(fill="both", expand=True)

        self.create_widgets()

        # Wywołujemy pętlę do aktualizowania UI (pozycji odtwarzania itp.)
        self.ui_after = None
        self.update_ui()
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        # --- Górny panel z przyciskami ---
        self.top_frame = ttk.Frame(self.main_frame, style="Controls.TFrame")
        self.top_frame.pack(side="top", fill="x", padx=10, pady=10)

        self.load_button = ttk.Button(
            self.top_frame,
            text="Wczytaj plik WAV",
            command=self.load_file
        )
        self.load_button.grid(row=0, column=0, padx=5, pady=5)

        self.play_button = ttk.Button(
            self.top_frame,
            text="Odtwórz",
            command=self.play_audio,
            state="disabled"
        )
        self.play_button.grid(row=0, column=1, padx=5, pady=5)

        # Odtwarzanie od początku
        self.play_from_start_button = ttk.Button(
            self.top_frame,
            text="Odtwórz od początku",
            command=self.play_from_start,
            state="disabled"
        )
        self.play_from_start_button.grid(row=0, column=2, padx=5, pady=5)

        self.pause_button = ttk.Button(
            self.top_frame,
            text="Pauza",
            command=self.toggle_pause,
            state="disabled"
        )
        self.pause_button.grid(row=0, column=3, padx=5, pady=5)

        self.features_button = ttk.Button(
            self.top_frame,
            text="Wykresy cech",
            command=self.open_features_window,
            state="disabled"
        )
        self.features_button.grid(row=0, column=4, padx=5, pady=5)

        self.freq_analysis_button = ttk.Button(
            self.top_frame,
            text="Analiza częstotliwościowa",
            command=self.open_frequency_analysis,
            state="disabled"
        )
        self.freq_analysis_button.grid(row=0, column=5, padx=5, pady=5)

        self.cepstrum_button = ttk.Button(
            self.top_frame,
            text="Analiza F0 (cepstrum)",
            command=self.open_cepstrum_analysis,
            state="disabled"
        )
        self.cepstrum_button.grid(row=0, column=6, padx=5, pady=5)

        self.close_button = ttk.Button(
            self.top_frame,
            text="Zamknij",
            command=self.on_close
        )
        self.close_button.grid(row=0, column=7, padx=5, pady=5)

        # --- Sekcja info: nazwa pliku, czas, tryb ---
        info_frame = ttk.Frame(self.main_frame, style="App.TFrame")
        info_frame.pack(side="top", fill="x", padx=10, pady=(0, 5))

        self.file_label = ttk.Label(
            info_frame,
            text="Brak wczytanego pliku",
            style="TitleLabel.TLabel"
        )
        self.file_label.pack(side="top", anchor="w")

        self.time_label = ttk.Label(
            info_frame,
            text="Czas: 00:00"
        )
        self.time_label.pack(side="top", anchor="w", pady=5)

        # Etykieta z aktualnym trybem
        self.highlight_label = ttk.Label(
            info_frame,
            text="Aktualnie pokazujemy: CISZĘ"
        )
        self.highlight_label.pack(side="top", anchor="w", pady=5)

        # Suwak (inicjalizujemy po wczytaniu pliku)
        self.slider = None

        # Tekst z parametrami
        self.frame_params_text = tk.StringVar()
        self.params_label = ttk.Label(
            info_frame,
            textvariable=self.frame_params_text,
            justify="left"
        )
        self.params_label.pack(side="top", anchor="w", pady=5)

        # --- Opcje wyboru trybu podświetlania ---
        mode_frame = ttk.Frame(self.main_frame, style="Controls.TFrame")
        mode_frame.pack(side="top", fill="x", padx=10, pady=5)

        ttk.Label(
            mode_frame,
            text="Tryb podświetlania:",
            style="TitleLabel.TLabel"
        ).pack(side="left", padx=(5, 10))

        rb_silence = ttk.Radiobutton(
            mode_frame,
            text="Cisza",
            variable=self.highlight_mode,
            value="silence",
            command=self.update_highlight_mode
        )
        rb_silence.pack(side="left", padx=5)

        rb_voiced = ttk.Radiobutton(
            mode_frame,
            text="Dźwięczne/Bezdźwięczne",
            variable=self.highlight_mode,
            value="voiced_unvoiced",
            command=self.update_highlight_mode
        )
        rb_voiced.pack(side="left", padx=5)

        # --- Ramka z wykresem audio ---
        plot_frame = ttk.LabelFrame(
            self.main_frame,
            text="Przebieg czasowy sygnału",
            style="App.TFrame"
        )
        plot_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(8, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Podpięcie obsługi zdarzenia zmiany rozmiaru wykresu
        self.canvas.mpl_connect('resize_event', self.on_resize)

    # Dodajemy nowe metody do obsługi analizy częstotliwościowej
    def open_frequency_analysis(self):
        """Otwiera okno analizy częstotliwościowej."""
        if self.data is not None:
            FrequencyAnalysisWindow(self.master, self)
        else:
            messagebox.showerror("Błąd", "Najpierw wczytaj plik audio.")

    def open_cepstrum_analysis(self):
        """Otwiera okno analizy cepstralnej do estymacji częstotliwości podstawowej."""
        if self.data is not None:
            CepstrumAnalysisWindow(self.master, self)
        else:
            messagebox.showerror("Błąd", "Najpierw wczytaj plik audio.")

    def on_resize(self, event):
        self.background = None
        if self.line is not None:
            self.canvas.draw()
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def update_highlight_mode(self):
        mode = self.highlight_mode.get()
        if mode == "silence":
            self.highlight_label.config(text="Aktualnie pokazujemy: CISZĘ")
        else:
            self.highlight_label.config(text="Aktualnie pokazujemy: DŹWIĘCZNE / BEZDŹWIĘCZNE")

        if self.data is not None:
            self.draw_main_plot()

    def load_file(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if not filepath:
            return

        self.filename = filepath
        base_name = os.path.basename(filepath)
        self.file_label.config(text=f"Plik: {base_name}")

        try:
            self.fs, raw_data = wavfile.read(filepath)
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się wczytać pliku WAV:\n{e}")
            return

        if len(raw_data.shape) > 1:
            raw_data = raw_data[:, 0]

        raw_data = raw_data.astype(np.float32)
        peak = np.max(np.abs(raw_data))
        if peak > 1e-9:
            raw_data /= peak

        self.data = raw_data
        self.total_samples = len(self.data)
        duration = self.total_samples / self.fs if self.fs else 0.001
        self.time_array = np.linspace(0, duration, self.total_samples)
        self.current_index = 0

        # Rysujemy główny wykres
        self.draw_main_plot()
        self.calculate_and_display_frame_params()

        # Tworzymy / aktualizujemy suwak
        if not self.slider:
            self.slider = ttk.Scale(
                self.main_frame,
                from_=0,
                to=max(duration, 0.001),
                orient="horizontal",
                command=self.on_slider_move,
                length=600
            )
            self.slider.pack(pady=5)
        else:
            self.slider.config(to=duration)
            self.slider.set(0)

        # Włączamy przyciski
        self.play_button.state(["!disabled"])
        self.pause_button.state(["!disabled"])
        self.features_button.state(["!disabled"])
        self.play_from_start_button.state(["!disabled"])
        # Włączamy nowe przyciski analizy częstotliwościowej
        self.freq_analysis_button.state(["!disabled"])
        self.cepstrum_button.state(["!disabled"])

        # Zatrzymujemy poprzedni strumień (jeśli był)
        self.stop_audio()

        # Tworzymy nowy strumień audio
        self.stream = sd.OutputStream(
            samplerate=self.fs,
            blocksize=1024,
            channels=1,
            dtype='float32',
            callback=self.audio_callback
        )

    def draw_main_plot(self):
        self.ax.clear()
        self.ax.set_title("Przebieg czasowy sygnału", fontsize=11, color=ColorScheme.ACCENT)
        self.ax.set_xlabel("Czas [s]", fontsize=9)
        self.ax.set_ylabel("Amplituda", fontsize=9)
        self.ax.plot(self.time_array, self.data, linewidth=0.8, color=ColorScheme.WAVEFORM_COLOR)

        legend_patches = []
        mode = self.highlight_mode.get()

        if mode == "silence":
            silence_regions = self.processor.detect_silence(
                self.data, self.fs, self.frame_size, self.silence_threshold
            )
            for (start_idx, end_idx) in silence_regions:
                start_t = start_idx / self.fs
                end_t = end_idx / self.fs
                self.ax.axvspan(start_t, end_t, color=ColorScheme.SILENCE_COLOR, alpha=0.6)
            silence_patch = Patch(facecolor=ColorScheme.SILENCE_COLOR, alpha=0.6, label="Cisza")
            legend_patches.append(silence_patch)
        else:
            vu_regions = self.processor.detect_voiced_unvoiced(
                self.data, self.fs, self.frame_size
            )
            for (start_idx, end_idx, is_voiced) in vu_regions:
                start_t = start_idx / self.fs
                end_t = end_idx / self.fs
                if is_voiced:
                    self.ax.axvspan(start_t, end_t, color=ColorScheme.VOICED_COLOR, alpha=0.3)
                else:
                    self.ax.axvspan(start_t, end_t, color=ColorScheme.UNVOICED_COLOR, alpha=0.3)
            voiced_patch = Patch(facecolor=ColorScheme.VOICED_COLOR, alpha=0.3, label="Dźwięczne")
            unvoiced_patch = Patch(facecolor=ColorScheme.UNVOICED_COLOR, alpha=0.3, label="Bezdźwięczne")
            legend_patches.extend([voiced_patch, unvoiced_patch])

        if legend_patches:
            self.ax.legend(handles=legend_patches, loc="upper right", fontsize=8)

        self.canvas.draw()

        # Po narysowaniu wykresu usuwamy background,
        # który zostanie zaktualizowany przy starcie odtwarzania
        self.background = None

    def on_slider_move(self, value):
        if self.data is not None:
            new_index = int(float(value) * self.fs) if self.fs else 0
            self.current_index = np.clip(new_index, 0, self.total_samples)
            self.update_time_label(float(value))

            # Jeżeli linia istnieje, przesuwamy ją
            if self.line is not None and self.background is not None:
                self.canvas.restore_region(self.background)
                self.line.set_xdata([float(value), float(value)])
                self.ax.draw_artist(self.line)
                self.canvas.blit(self.ax.bbox)

    def update_time_label(self, current_time):
        minutes = int(current_time // 60)
        seconds = int(current_time % 60)
        self.time_label.config(text=f"Czas: {minutes:02d}:{seconds:02d}")

    def calculate_and_display_frame_params(self):
        frame_step = self.frame_size
        rms_values = []
        zcr_values = []
        for i in range(0, self.total_samples, frame_step):
            frame = self.data[i:i + frame_step]
            if len(frame) == 0:
                continue
            rms = np.sqrt(np.mean(frame ** 2))
            rms_values.append(rms)
            zero_crosses = np.count_nonzero(np.diff(np.sign(frame)))
            zcr = zero_crosses / len(frame) if len(frame) != 0 else 0
            zcr_values.append(zcr)
        avg_rms = np.mean(rms_values) if rms_values else 0
        avg_zcr = np.mean(zcr_values) if zcr_values else 0
        text = (
            f"Parametry nagrania (ramkowe):\n"
            f"  • Średni RMS (Volume): {avg_rms:.6f}\n"
            f"  • Średnie ZCR: {avg_zcr:.6f}\n"
            f"(Próg ciszy: {self.silence_threshold})"
        )
        self.frame_params_text.set(text)

    def audio_callback(self, outdata, frames, time_info, status):
        """Funkcja wywoływana przez sounddevice przy każdej porcji danych."""
        if status:
            # print("Status audio_callback:", status, flush=True)
            pass

        if (self.data is None) or (not self.playing) or (self.paused):
            outdata.fill(0)
            return

        end_index = self.current_index + frames
        if end_index > self.total_samples:
            end_index = self.total_samples

        chunk = self.data[self.current_index:end_index]
        out_len = len(chunk)
        if out_len < frames:
            outdata[:out_len, 0] = chunk
            outdata[out_len:, 0] = 0
            self.playing = False
            self.current_index = self.total_samples
        else:
            outdata[:, 0] = chunk
            self.current_index = end_index

    def play_audio(self):
        if self.data is None or not self.stream:
            return

        # Jeżeli linia nie istnieje, tworzymy ją
        if self.line is None:
            self.line = self.ax.axvline(x=0, color="#004D40", linewidth=2)
            # Najpierw pełne rysowanie wykresu
            self.canvas.draw()
            # Dopiero teraz pobieramy background
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        # Jeżeli dotarliśmy do końca – wróćmy na początek
        if self.current_index >= self.total_samples:
            self.current_index = 0
            if self.slider:
                self.slider.set(0)
            # Reset linii do 0
            if self.line:
                self.line.set_xdata([0, 0])
                self.canvas.draw()
                self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        self.playing = True
        self.paused = False
        self.pause_button.config(text="Pauza")

        # Start strumienia
        if not self.stream.active:
            self.stream.start()

    def play_from_start(self):
        if self.data is None:
            return
        self.current_index = 0
        if self.slider:
            self.slider.set(0)
        if self.line:
            self.line.set_xdata([0, 0])
            self.canvas.draw()
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.play_audio()

    def stop_audio(self):
        self.playing = False
        self.paused = False
        if self.stream and self.stream.active:
            self.stream.stop()

        # Usuwamy linię z osi, jeśli istnieje
        if self.line:
            self.line.remove()
            self.line = None

        # Odśwież wykres (linia znika)
        self.canvas.draw()
        self.background = None

    def toggle_pause(self):
        if self.data is None or not self.stream:
            return

        if not self.playing:
            self.play_audio()
            self.pause_button.config(text="Pauza")
            return

        if self.paused:
            self.paused = False
            self.pause_button.config(text="Pauza")
        else:
            self.paused = True
            self.pause_button.config(text="Wznów")

    def update_ui(self):
        if self.data is not None and self.playing and not self.paused:
            current_time = self.current_index / self.fs if self.fs else 0

            # Ustawiamy suwak
            if self.slider:
                self.slider.set(current_time)

            # Aktualizujemy etykietę z czasem
            self.update_time_label(current_time)

            # Przesuwamy linię (blitowanie)
            if self.line and self.background:
                self.canvas.restore_region(self.background)
                self.line.set_xdata([current_time, current_time])
                self.ax.draw_artist(self.line)
                self.canvas.blit(self.ax.bbox)

        # Wywołanie za 50 ms ponownie
        self.ui_after = self.master.after(50, self.update_ui)

    def open_features_window(self):
        if self.data is None:
            messagebox.showwarning("Brak danych", "Najpierw wczytaj plik WAV!")
            return
        FeaturesWindow(self.master, self.data, self.fs, self.frame_size, self.silence_threshold)

    def on_close(self):
        self.stop_audio()
        if self.ui_after is not None:
            self.master.after_cancel(self.ui_after)
            self.ui_after = None
        if self.stream:
            self.stream.close()
            self.stream = None
        self.master.destroy()
        sys.exit(0)