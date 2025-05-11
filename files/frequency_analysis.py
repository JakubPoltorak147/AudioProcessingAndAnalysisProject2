import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox

from design import ColorScheme
from frequency_features import (
    compute_volume_frequency, compute_frequency_centroid,
    compute_bandwidth, compute_band_energy_ratio,
    compute_spectral_flatness, compute_spectral_crest_factor
)
from windowing import apply_window, get_window_type_name
from cepstrum_analysis import compute_cepstrum, estimate_f0_from_cepstrum
from frequency_features_window import FrequencyFeaturesWindow



class FrequencyAnalysisWindow:

    def __init__(self, parent, audio_app):
        self.parent = parent
        self.audio_app = audio_app

        # Tworzymy nowe okno
        self.window = tk.Toplevel(parent)
        self.window.title("Analiza częstotliwościowa")
        self.window.geometry("1200x800")
        self.window.minsize(800, 600)

        # Tworzymy główną ramkę
        self.main_frame = ttk.Frame(self.window, style="App.TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tworzymy panel kontrolny
        self.create_control_panel()

        # Tworzymy obszar wykresów
        self.create_plot_area()

        # Inicjalizujemy zmienne
        self.frame_start = 0
        self.frame_length = 1024
        self.window_type = "rectangular"
        self.overlap = 0.5  # 50% nakładanie dla spektrogramu

        # Panel statystyk
        self.create_stats_panel()

        # Aktualizujemy wykresy
        self.update_plots()

    def create_control_panel(self):
        """Tworzy panel kontrolny z opcjami analizy częstotliwościowej."""
        control_frame = ttk.LabelFrame(self.main_frame, text="Panel Kontrolny", style="Freq.TLabelframe")
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Wybór ramki
        frame_frame = ttk.Frame(control_frame, style="Controls.TFrame")
        frame_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(frame_frame, text="Początek ramki:").grid(row=0, column=0, padx=5, pady=5)
        self.frame_start_var = tk.StringVar(value="0")
        frame_start_entry = ttk.Entry(frame_frame, textvariable=self.frame_start_var, width=10)
        frame_start_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(frame_frame, text="Długość ramki:").grid(row=0, column=2, padx=5, pady=5)
        self.frame_length_var = tk.StringVar(value="1024")
        frame_length_entry = ttk.Entry(frame_frame, textvariable=self.frame_length_var, width=10)
        frame_length_entry.grid(row=0, column=3, padx=5, pady=5)

        # Wybór funkcji okienkowej
        window_frame = ttk.Frame(control_frame, style="Controls.TFrame")
        window_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(window_frame, text="Funkcja okienkowa:").grid(row=0, column=0, padx=5, pady=5)
        self.window_var = tk.StringVar(value="rectangular")
        window_combo = ttk.Combobox(window_frame, textvariable=self.window_var, width=15)
        window_combo['values'] = ('rectangular', 'triangular', 'hamming', 'hann', 'blackman')
        window_combo.grid(row=0, column=1, padx=5, pady=5)

        # Opcje spektrogramu
        spec_frame = ttk.Frame(control_frame, style="Controls.TFrame")
        spec_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(spec_frame, text="Nakładanie (%):").grid(row=0, column=0, padx=5, pady=5)
        self.overlap_var = tk.StringVar(value="50")
        overlap_entry = ttk.Entry(spec_frame, textvariable=self.overlap_var, width=10)
        overlap_entry.grid(row=0, column=1, padx=5, pady=5)

        # Przycisk aktualizacji
        update_button = ttk.Button(control_frame, text="Aktualizuj wykresy", command=self.update_plots)
        update_button.pack(padx=5, pady=5)

        features_button = ttk.Button(
            control_frame,
            text="Wykresy parametrów częstotliwościowych",
            command=self.open_frequency_features
        )
        features_button.pack(padx=5, pady=5)

    def create_stats_panel(self):
        self.stats_frame = ttk.LabelFrame(self.main_frame, text="Parametry częstotliwościowe", style="Freq.TLabelframe")
        self.stats_frame.pack(fill=tk.X, padx=5, pady=5)

        self.stats_text = tk.Text(self.stats_frame, height=5, width=80, font=("Courier", 9))
        self.stats_text.pack(padx=5, pady=5, fill=tk.X)
        self.stats_text.config(state=tk.DISABLED)

    def update_stats(self):
        if hasattr(self.audio_app, 'data') and self.audio_app.data is not None:
            # Pobieramy dane sygnału
            signal = self.audio_app.data
            sample_rate = self.audio_app.fs

            # Wycinamy ramkę do analizy
            end_idx = min(self.frame_start + self.frame_length, len(signal))
            frame = signal[self.frame_start:end_idx]

            # Stosujemy funkcję okienkową
            windowed_frame = apply_window(frame, self.window_type)

            # Obliczamy FFT
            fft_window = np.fft.rfft(windowed_frame)
            freqs = np.fft.rfftfreq(len(windowed_frame), d=1 / sample_rate)

            # Obliczamy parametry
            volume = compute_volume_frequency(fft_window)
            centroid = compute_frequency_centroid(fft_window, freqs)
            bandwidth = compute_bandwidth(fft_window, freqs, centroid)

            # Pasma częstotliwości dla BER (zakładając częstotliwość próbkowania 22050 Hz lub wyższą)
            if sample_rate >= 11025:
                ersb1 = compute_band_energy_ratio(fft_window, freqs, 0, 630)
                ersb2 = compute_band_energy_ratio(fft_window, freqs, 630, 1720)
                ersb3 = compute_band_energy_ratio(fft_window, freqs, 1720, 4400)
            else:
                # Dostosuj pasma dla niższych częstotliwości próbkowania
                max_freq = sample_rate / 2
                ersb1 = compute_band_energy_ratio(fft_window, freqs, 0, min(630, max_freq))
                ersb2 = compute_band_energy_ratio(fft_window, freqs, min(630, max_freq), min(1720, max_freq))
                ersb3 = compute_band_energy_ratio(fft_window, freqs, min(1720, max_freq), min(4400, max_freq))

            sfm = compute_spectral_flatness(fft_window)
            scf = compute_spectral_crest_factor(fft_window)

            # Aktualizujemy pole tekstowe
            self.stats_text.config(state=tk.NORMAL)
            self.stats_text.delete(1.0, tk.END)
            stats_info = (
                f"Głośność (Volume): {volume:.6f}\n"
                f"Centroid częstotliwościowy (FC): {centroid:.2f} Hz\n"
                f"Szerokość pasma (BW): {bandwidth:.2f} Hz\n"
                f"Stosunek energii w pasmach (ERSB1, ERSB2, ERSB3): {ersb1:.4f}, {ersb2:.4f}, {ersb3:.4f}\n"
                f"Płaskość widma (SFM): {sfm:.6f}, Współczynnik szczytu widma (SCF): {scf:.6f}\n"
                f"Okno: {get_window_type_name(self.window_type)}, Długość ramki: {self.frame_length}, Pozycja: {self.frame_start}"
            )
            self.stats_text.insert(tk.END, stats_info)
            self.stats_text.config(state=tk.DISABLED)

    def create_plot_area(self):
        # Tworzymy notebook z zakładkami
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Zakładka dziedziny czasu
        self.time_frame = ttk.Frame(self.notebook, style="App.TFrame")
        self.notebook.add(self.time_frame, text="Dziedzina czasu")

        # Zakładka dziedziny częstotliwości
        self.freq_frame = ttk.Frame(self.notebook, style="App.TFrame")
        self.notebook.add(self.freq_frame, text="Dziedzina częstotliwości")

        # Zakładka spektrogramu
        self.spec_frame = ttk.Frame(self.notebook, style="App.TFrame")
        self.notebook.add(self.spec_frame, text="Spektrogram")

        # Tworzymy figury i płótna dla każdej zakładki
        self.time_fig = Figure(figsize=(10, 6), dpi=100)
        self.time_canvas = FigureCanvasTkAgg(self.time_fig, self.time_frame)
        self.time_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.freq_fig = Figure(figsize=(10, 6), dpi=100)
        self.freq_canvas = FigureCanvasTkAgg(self.freq_fig, self.freq_frame)
        self.freq_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.spec_fig = Figure(figsize=(10, 6), dpi=100)
        self.spec_canvas = FigureCanvasTkAgg(self.spec_fig, self.spec_frame)
        self.spec_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_plots(self):
        if hasattr(self.audio_app, 'data') and self.audio_app.data is not None:
            try:
                # Pobieramy parametry z UI
                self.frame_start = int(self.frame_start_var.get())
                self.frame_length = int(self.frame_length_var.get())
                self.window_type = self.window_var.get()
                self.overlap = float(self.overlap_var.get()) / 100.0

                # Aktualizujemy wykresy
                self.plot_time_domain()
                self.plot_frequency_domain()
                self.plot_spectrogram()

                # Aktualizujemy statystyki
                self.update_stats()
            except ValueError as e:
                print(f"Błąd aktualizacji wykresów: {e}")
                messagebox.showerror("Błąd", f"Błąd aktualizacji wykresów: {e}")

    def plot_time_domain(self):
        # Czyścimy poprzedni wykres
        self.time_fig.clear()
        ax = self.time_fig.add_subplot(111)

        # Pobieramy dane sygnału
        signal = self.audio_app.data
        sample_rate = self.audio_app.fs

        # Wycinamy ramkę do analizy
        end_idx = min(self.frame_start + self.frame_length, len(signal))
        frame = signal[self.frame_start:end_idx]

        # Stosujemy funkcję okienkową
        windowed_frame = apply_window(frame, self.window_type)

        # Tworzymy tablicę czasu
        time = np.arange(len(frame)) / sample_rate

        # Rysujemy oryginalną ramkę
        ax.plot(time, frame, '-', color=ColorScheme.ORIGINAL_SIGNAL, alpha=0.5, label='Oryginalny')

        # Rysujemy ramkę z oknem
        ax.plot(time, windowed_frame, '-', color=ColorScheme.WINDOWED_SIGNAL, label='Z oknem')

        # Ustawiamy etykiety i tytuł
        ax.set_xlabel('Czas (s)')
        ax.set_ylabel('Amplituda')
        window_name = get_window_type_name(self.window_type)
        ax.set_title(f'Sygnał w dziedzinie czasu z oknem {window_name}')
        ax.legend()
        ax.grid(True)

        # Aktualizujemy płótno
        self.time_canvas.draw()

    def plot_frequency_domain(self):
        # Czyścimy poprzedni wykres
        self.freq_fig.clear()
        ax = self.freq_fig.add_subplot(111)

        # Pobieramy dane sygnału
        signal = self.audio_app.data
        sample_rate = self.audio_app.fs

        # Wycinamy ramkę do analizy
        end_idx = min(self.frame_start + self.frame_length, len(signal))
        frame = signal[self.frame_start:end_idx]

        # Stosujemy funkcję okienkową
        windowed_frame = apply_window(frame, self.window_type)

        # Obliczamy FFT dla oryginalnej ramki
        fft_orig = np.fft.rfft(frame)
        freq_orig = np.fft.rfftfreq(len(frame), d=1 / sample_rate)

        # Obliczamy FFT dla ramki z oknem
        fft_window = np.fft.rfft(windowed_frame)
        freq_window = np.fft.rfftfreq(len(windowed_frame), d=1 / sample_rate)

        # Konwertujemy na amplitudę w dB
        magnitude_orig = 20 * np.log10(np.abs(fft_orig) + 1e-10)  # Dodajemy małą wartość, aby uniknąć log(0)
        magnitude_window = 20 * np.log10(np.abs(fft_window) + 1e-10)

        # Rysujemy oryginalne widmo
        ax.plot(freq_orig, magnitude_orig, '-', color=ColorScheme.ORIGINAL_SIGNAL, alpha=0.5, label='Oryginalny')

        # Rysujemy widmo z oknem
        ax.plot(freq_window, magnitude_window, '-', color=ColorScheme.WINDOWED_SIGNAL, label='Z oknem')

        # Ustawiamy etykiety i tytuł
        ax.set_xlabel('Częstotliwość (Hz)')
        ax.set_ylabel('Amplituda (dB)')
        window_name = get_window_type_name(self.window_type)
        ax.set_title(f'Widmo częstotliwościowe z oknem {window_name}')
        ax.legend()
        ax.grid(True)

        # Aktualizujemy płótno
        self.freq_canvas.draw()

    def plot_spectrogram(self):
        # Czyścimy poprzedni wykres
        self.spec_fig.clear()
        ax = self.spec_fig.add_subplot(111)

        # Pobieramy dane sygnału
        signal = self.audio_app.data
        sample_rate = self.audio_app.fs

        # Obliczamy spektrogram
        spec_data, freqs, times = self.compute_spectrogram(signal, sample_rate)

        # Rysujemy spektrogram
        im = ax.imshow(spec_data, aspect='auto', origin='lower',
                       extent=[times[0], times[-1], freqs[0], freqs[-1]],
                       cmap=ColorScheme.SPECTROGRAM_CMAP)
        #ax.set_ylim(0,4000)

        # Ustawiamy etykiety i tytuł
        ax.set_xlabel('Czas (s)')
        ax.set_ylabel('Częstotliwość (Hz)')
        window_name = get_window_type_name(self.window_type)
        ax.set_title(f'Spektrogram z oknem {window_name}')

        # Dodajemy pasek kolorów
        self.spec_fig.colorbar(im, ax=ax, label='Amplituda (dB)')

        # Aktualizujemy płótno
        self.spec_canvas.draw()

    def compute_spectrogram(self, signal, sample_rate):
        # Parametry
        window_length = self.frame_length
        hop_length = int(window_length * (1 - self.overlap))  # Bazując na nakładaniu

        # Obliczamy liczbę ramek
        num_frames = 1 + (len(signal) - window_length) // hop_length

        # Prealokujemy macierz spektrogramu
        spec = np.zeros((window_length // 2 + 1, num_frames))

        # Obliczamy spektrogram ramka po ramce
        for i in range(num_frames):
            # Wycinamy ramkę
            start = i * hop_length
            end = start + window_length
            frame = signal[start:end]

            # Stosujemy okno
            windowed_frame = apply_window(frame, self.window_type)

            # Obliczamy FFT
            fft_frame = np.fft.rfft(windowed_frame)

            # Konwertujemy na amplitudę w dB
            magnitude = 20 * np.log10(np.abs(fft_frame) + 1e-10)

            # Zapisujemy w macierzy spektrogramu
            spec[:, i] = magnitude

        # Obliczamy tablice częstotliwości i czasu
        freqs = np.fft.rfftfreq(window_length, d=1 / sample_rate)
        times = np.arange(0, num_frames * hop_length, hop_length) / sample_rate

        return spec, freqs, times

    def open_frequency_features(self):
        if hasattr(self.audio_app, 'data') and self.audio_app.data is not None:
            FrequencyFeaturesWindow(
                self.window,
                self.audio_app.data,
                self.audio_app.fs,
                frame_size=self.frame_length,
                window_type=self.window_type,
                overlap=self.overlap
            )
        else:
            messagebox.showerror("Błąd", "Brak danych audio.")


class CepstrumAnalysisWindow:

    def __init__(self, parent, audio_app):
        self.parent = parent
        self.audio_app = audio_app

        # Tworzymy nowe okno
        self.window = tk.Toplevel(parent)
        self.window.title("Analiza cepstralna")
        self.window.geometry("900x700")
        self.window.minsize(800, 600)

        # Tworzymy główną ramkę
        self.main_frame = ttk.Frame(self.window, style="App.TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tworzymy panel kontrolny
        self.create_control_panel()

        # Tworzymy obszar wykresów
        self.create_plot_area()

        # Inicjalizujemy zmienne
        self.frame_start = 0
        self.frame_length = 2048
        self.window_type = "hamming"
        self.min_f0 = 50  # Minimalna częstotliwość podstawowa do rozważenia
        self.max_f0 = 500  # Maksymalna częstotliwość podstawowa do rozważenia

        # Panel informacyjny F0
        self.create_f0_info_panel()

        # Aktualizujemy wykresy
        self.update_plots()

    def create_f0_info_panel(self):
        self.f0_info_frame = ttk.LabelFrame(self.main_frame, text="Informacje o F0", style="Freq.TLabelframe")
        self.f0_info_frame.pack(fill=tk.X, padx=5, pady=5)

        self.f0_info_text = tk.Text(self.f0_info_frame, height=3, width=80, font=("Courier", 9))
        self.f0_info_text.pack(padx=5, pady=5, fill=tk.X)
        self.f0_info_text.config(state=tk.DISABLED)

    def update_f0_info(self, f0_value):
        self.f0_info_text.config(state=tk.NORMAL)
        self.f0_info_text.delete(1.0, tk.END)

        # Dodajemy informacje o F0
        f0_info = (
            f"Częstotliwość podstawowa (F0): {f0_value:.2f} Hz\n"
            f"Okres podstawowy: {1000 / f0_value:.2f} ms\n"
            f"Okno: {get_window_type_name(self.window_type)}, Długość ramki: {self.frame_length}, Pozycja: {self.frame_start}"
        )
        self.f0_info_text.insert(tk.END, f0_info)
        self.f0_info_text.config(state=tk.DISABLED)

    def create_control_panel(self):
        control_frame = ttk.LabelFrame(self.main_frame, text="Panel Kontrolny", style="Freq.TLabelframe")
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Wybór ramki
        frame_frame = ttk.Frame(control_frame, style="Controls.TFrame")
        frame_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(frame_frame, text="Początek ramki:").grid(row=0, column=0, padx=5, pady=5)
        self.frame_start_var = tk.StringVar(value="0")
        frame_start_entry = ttk.Entry(frame_frame, textvariable=self.frame_start_var, width=10)
        frame_start_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(frame_frame, text="Długość ramki:").grid(row=0, column=2, padx=5, pady=5)
        self.frame_length_var = tk.StringVar(value="2048")
        frame_length_entry = ttk.Entry(frame_frame, textvariable=self.frame_length_var, width=10)
        frame_length_entry.grid(row=0, column=3, padx=5, pady=5)

        # Wybór funkcji okienkowej
        window_frame = ttk.Frame(control_frame, style="Controls.TFrame")
        window_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(window_frame, text="Funkcja okienkowa:").grid(row=0, column=0, padx=5, pady=5)
        self.window_var = tk.StringVar(value="hamming")
        window_combo = ttk.Combobox(window_frame, textvariable=self.window_var, width=15)
        window_combo['values'] = ('rectangular', 'triangular', 'hamming', 'hann', 'blackman')
        window_combo.grid(row=0, column=1, padx=5, pady=5)

        # Zakres F0
        f0_frame = ttk.Frame(control_frame, style="Controls.TFrame")
        f0_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(f0_frame, text="Min F0 (Hz):").grid(row=0, column=0, padx=5, pady=5)
        self.min_f0_var = tk.StringVar(value="50")
        min_f0_entry = ttk.Entry(f0_frame, textvariable=self.min_f0_var, width=10)
        min_f0_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(f0_frame, text="Max F0 (Hz):").grid(row=0, column=2, padx=5, pady=5)
        self.max_f0_var = tk.StringVar(value="500")
        max_f0_entry = ttk.Entry(f0_frame, textvariable=self.max_f0_var, width=10)
        max_f0_entry.grid(row=0, column=3, padx=5, pady=5)

        # Przycisk aktualizacji
        update_button = ttk.Button(control_frame, text="Aktualizuj wykresy", command=self.update_plots)
        update_button.pack(padx=5, pady=5)

        features_button = ttk.Button(
            control_frame,
            text="Wykresy parametrów częstotliwościowych",
            command=self.open_frequency_features
        )
        features_button.pack(padx=5, pady=5)

    def create_plot_area(self):
        # Tworzymy notebook z zakładkami
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Zakładka widma
        self.spectrum_frame = ttk.Frame(self.notebook, style="App.TFrame")
        self.notebook.add(self.spectrum_frame, text="Widmo logarytmiczne")

        # Zakładka cepstrum
        self.cepstrum_frame = ttk.Frame(self.notebook, style="App.TFrame")
        self.notebook.add(self.cepstrum_frame, text="Cepstrum")

        # Zakładka F0 w czasie
        self.f0_frame = ttk.Frame(self.notebook, style="App.TFrame")
        self.notebook.add(self.f0_frame, text="F0 w czasie")

        # Tworzymy figury i płótna dla każdej zakładki
        self.spectrum_fig = Figure(figsize=(10, 6), dpi=100)
        self.spectrum_canvas = FigureCanvasTkAgg(self.spectrum_fig, self.spectrum_frame)
        self.spectrum_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.cepstrum_fig = Figure(figsize=(10, 6), dpi=100)
        self.cepstrum_canvas = FigureCanvasTkAgg(self.cepstrum_fig, self.cepstrum_frame)
        self.cepstrum_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.f0_fig = Figure(figsize=(10, 6), dpi=100)
        self.f0_canvas = FigureCanvasTkAgg(self.f0_fig, self.f0_frame)
        self.f0_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_plots(self):
        if hasattr(self.audio_app, 'data') and self.audio_app.data is not None:
            try:
                # Pobieramy parametry z UI
                self.frame_start = int(self.frame_start_var.get())
                self.frame_length = int(self.frame_length_var.get())
                self.window_type = self.window_var.get()
                self.min_f0 = float(self.min_f0_var.get())
                self.max_f0 = float(self.max_f0_var.get())

                # Aktualizujemy wykresy
                self.plot_log_spectrum()
                f0_value = self.plot_cepstrum()
                self.plot_f0_over_time()

                # Aktualizujemy info o F0
                self.update_f0_info(f0_value)
            except ValueError as e:
                print(f"Błąd aktualizacji wykresów: {e}")
                messagebox.showerror("Błąd", f"Błąd aktualizacji wykresów: {e}")

    def plot_log_spectrum(self):
        # Czyścimy poprzedni wykres
        self.spectrum_fig.clear()
        ax = self.spectrum_fig.add_subplot(111)

        # Pobieramy dane sygnału
        signal = self.audio_app.data
        sample_rate = self.audio_app.fs

        # Wycinamy ramkę do analizy
        end_idx = min(self.frame_start + self.frame_length, len(signal))
        frame = signal[self.frame_start:end_idx]

        # Obliczamy cepstrum
        _, _, log_spectrum = compute_cepstrum(frame, sample_rate, self.window_type)

        # Obliczamy oś częstotliwości
        freq = np.fft.rfftfreq(len(frame), d=1 / sample_rate)

        # Rysujemy widmo logarytmiczne
        ax.plot(freq, log_spectrum, color=ColorScheme.ACCENT)

        # Ustawiamy etykiety i tytuł
        ax.set_xlabel('Częstotliwość (Hz)')
        ax.set_ylabel('Logarytm amplitudy')
        ax.set_title('Widmo logarytmiczne')
        ax.grid(True)

        # Aktualizujemy płótno
        self.spectrum_canvas.draw()

    def plot_cepstrum(self):
        # Czyścimy poprzedni wykres
        self.cepstrum_fig.clear()
        ax = self.cepstrum_fig.add_subplot(111)

        # Pobieramy dane sygnału
        signal = self.audio_app.data
        sample_rate = self.audio_app.fs

        # Wycinamy ramkę do analizy
        end_idx = min(self.frame_start + self.frame_length, len(signal))
        frame = signal[self.frame_start:end_idx]

        # Obliczamy cepstrum
        cepstrum, quefrency, _ = compute_cepstrum(frame, sample_rate, self.window_type)

        # Estymujemy F0
        f0, peak_idx = estimate_f0_from_cepstrum(cepstrum, quefrency, self.min_f0, self.max_f0)

        # Rysujemy cepstrum
        ax.plot(quefrency, cepstrum, color=ColorScheme.ACCENT)

        # Zaznaczamy szczyt F0
        ax.axvline(x=quefrency[peak_idx], color=ColorScheme.F0_PEAK_COLOR, linestyle='--',
                   label=f'F0: {f0:.1f} Hz')

        # Ustawiamy granice osi, aby skupić się na istotnym zakresie kwefrencji
        min_quefrency = 1 / self.max_f0
        max_quefrency = 1 / self.min_f0
        ax.set_xlim(min_quefrency, max_quefrency)

        # Ustawiamy etykiety i tytuł
        ax.set_xlabel('Kwefrencja (sekundy)')
        ax.set_ylabel('Amplituda')
        ax.set_title('Cepstrum z zaznaczonym pikiem F0')
        ax.legend()
        ax.grid(True)

        # Aktualizujemy płótno
        self.cepstrum_canvas.draw()

        return f0

    def plot_f0_over_time(self):
        # Czyścimy poprzedni wykres
        self.f0_fig.clear()
        ax = self.f0_fig.add_subplot(111)

        # Pobieramy dane sygnału
        signal = self.audio_app.data
        sample_rate = self.audio_app.fs

        # Parametry analizy
        frame_size = 2048  # Stały rozmiar ramki dla śledzenia F0
        hop_size = 512  # Przeskok między ramkami

        # Obliczamy liczbę ramek
        num_frames = 1 + (len(signal) - frame_size) // hop_size

        # Prealokujemy tablice
        f0_values = np.zeros(num_frames)
        time_values = np.arange(num_frames) * hop_size / sample_rate

        # Obliczamy F0 dla każdej ramki
        for i in range(num_frames):
            # Wycinamy ramkę
            start = i * hop_size
            end = start + frame_size
            curr_frame = signal[start:end]

            # Obliczamy cepstrum
            cepstrum, quefrency, _ = compute_cepstrum(curr_frame, sample_rate, self.window_type)

            # Estymujemy F0
            f0, _ = estimate_f0_from_cepstrum(cepstrum, quefrency, self.min_f0, self.max_f0)

            # Zapisujemy wartość F0
            f0_values[i] = f0

        # Rysujemy F0 w czasie
        ax.plot(time_values, f0_values, color=ColorScheme.ACCENT)

        # Ustawiamy etykiety i tytuł
        ax.set_xlabel('Czas (s)')
        ax.set_ylabel('Częstotliwość podstawowa (Hz)')
        ax.set_title('Zmiany F0 w czasie (metoda cepstralna)')
        ax.grid(True)

        # Ustawiamy granice osi y na podstawie oczekiwanego zakresu F0
        ax.set_ylim(self.min_f0, self.max_f0)

        # Aktualizujemy płótno
        self.f0_canvas.draw()

    def open_frequency_features(self):
        if hasattr(self.audio_app, 'data') and self.audio_app.data is not None:
            FrequencyFeaturesWindow(
                self.window,
                self.audio_app.data,
                self.audio_app.fs,
                frame_size=self.frame_length,
                window_type=self.window_type
            )
        else:
            messagebox.showerror("Błąd", "Brak danych audio.")