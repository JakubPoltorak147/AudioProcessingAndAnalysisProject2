import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from design import ColorScheme
from windowing import apply_window
from frequency_features import (
    compute_volume_frequency, compute_frequency_centroid,
    compute_bandwidth, compute_band_energy_ratio,
    compute_spectral_flatness, compute_spectral_crest_factor
)


class FrequencyFeaturesWindow:
    """
    Okno do wyświetlania wykresów parametrów dźwięku w dziedzinie częstotliwości.
    """

    def __init__(self, parent, audio_data, sample_rate, frame_size=256, window_type='hamming',
                 overlap=0.5, frame_step=None):
        # Dane wejściowe
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.window_type = window_type
        self.overlap = overlap
        self.frame_step = frame_step or int(frame_size * (1 - overlap))

        # Tworzymy okno
        self.window = tk.Toplevel(parent)
        self.window.title("Wykresy parametrów częstotliwościowych")
        self.window.geometry("1000x800")
        self.window.minsize(800, 600)

        # Główna ramka
        self.main_frame = ttk.Frame(self.window, style="App.TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Panel kontrolny
        self.create_control_panel()

        # Obszar wykresów
        self.create_plot_area()

        # Obliczamy parametry dla wszystkich ramek
        self.feature_data = {}
        self.compute_all_features()

        # Aktualizujemy wykresy
        self.update_plots()

    def create_control_panel(self):
        """Tworzy panel kontrolny z opcjami wykresów."""
        control_frame = ttk.LabelFrame(self.main_frame, text="Opcje wyświetlania", style="Freq.TLabelframe")
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Ramka wyboru parametrów
        params_frame = ttk.Frame(control_frame, style="Controls.TFrame")
        params_frame.pack(fill=tk.X, padx=5, pady=5)

        # Zmienne do wyboru parametrów
        self.vol_var = tk.BooleanVar(value=True)
        self.fc_var = tk.BooleanVar(value=True)
        self.bw_var = tk.BooleanVar(value=True)
        self.ersb_var = tk.BooleanVar(value=False)
        self.sfm_var = tk.BooleanVar(value=False)
        self.scf_var = tk.BooleanVar(value=False)

        # Checkboxy parametrów
        ttk.Checkbutton(params_frame, text="Volume (Głośność)", variable=self.vol_var,
                        command=self.update_plots).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Checkbutton(params_frame, text="Centroid częstotliwościowy (FC)", variable=self.fc_var,
                        command=self.update_plots).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Checkbutton(params_frame, text="Szerokość pasma (BW)", variable=self.bw_var,
                        command=self.update_plots).grid(row=0, column=2, padx=5, pady=5, sticky="w")
        ttk.Checkbutton(params_frame, text="Stosunki energii (ERSB)", variable=self.ersb_var,
                        command=self.update_plots).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Checkbutton(params_frame, text="Płaskość widma (SFM)", variable=self.sfm_var,
                        command=self.update_plots).grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ttk.Checkbutton(params_frame, text="Współczynnik szczytu (SCF)", variable=self.scf_var,
                        command=self.update_plots).grid(row=1, column=2, padx=5, pady=5, sticky="w")

    def create_plot_area(self):
        """Tworzy obszar wykresów."""
        # Ramka wykresu
        plot_frame = ttk.LabelFrame(self.main_frame, text="Parametry w dziedzinie częstotliwości",
                                    style="Freq.TLabelframe")
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tworzymy figurę i płótno
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def compute_all_features(self):
        """Oblicza wszystkie parametry częstotliwościowe dla ramek."""
        # Liczba ramek
        num_frames = 1 + (len(self.audio_data) - self.frame_size) // self.frame_step

        # Inicjalizacja tablic na wyniki
        self.feature_data['time'] = np.arange(num_frames) * self.frame_step / self.sample_rate
        self.feature_data['volume'] = np.zeros(num_frames)
        self.feature_data['fc'] = np.zeros(num_frames)
        self.feature_data['bw'] = np.zeros(num_frames)
        self.feature_data['ersb1'] = np.zeros(num_frames)
        self.feature_data['ersb2'] = np.zeros(num_frames)
        self.feature_data['ersb3'] = np.zeros(num_frames)
        self.feature_data['sfm'] = np.zeros(num_frames)
        self.feature_data['scf'] = np.zeros(num_frames)

        # Obliczamy parametry dla każdej ramki
        for i in range(num_frames):
            # Wycinamy ramkę
            start = i * self.frame_step
            end = start + self.frame_size
            frame = self.audio_data[start:end]

            # Stosujemy okno
            windowed_frame = apply_window(frame, self.window_type)

            # Obliczamy FFT
            fft_frame = np.fft.rfft(windowed_frame)
            freqs = np.fft.rfftfreq(len(windowed_frame), d=1 / self.sample_rate)

            # Obliczamy parametry
            self.feature_data['volume'][i] = compute_volume_frequency(fft_frame)
            self.feature_data['fc'][i] = compute_frequency_centroid(fft_frame, freqs)
            self.feature_data['bw'][i] = compute_bandwidth(fft_frame, freqs, self.feature_data['fc'][i])

            # Pasma częstotliwości dla BER (zakładając częstotliwość próbkowania 22050 Hz lub wyższą)
            if self.sample_rate >= 11025:
                self.feature_data['ersb1'][i] = compute_band_energy_ratio(fft_frame, freqs, 0, 630)
                self.feature_data['ersb2'][i] = compute_band_energy_ratio(fft_frame, freqs, 630, 1720)
                self.feature_data['ersb3'][i] = compute_band_energy_ratio(fft_frame, freqs, 1720, 4400)
            else:
                # Dostosuj pasma dla niższych częstotliwości próbkowania
                max_freq = self.sample_rate / 2
                self.feature_data['ersb1'][i] = compute_band_energy_ratio(fft_frame, freqs, 0, min(630, max_freq))
                self.feature_data['ersb2'][i] = compute_band_energy_ratio(fft_frame, freqs, min(630, max_freq),
                                                                          min(1720, max_freq))
                self.feature_data['ersb3'][i] = compute_band_energy_ratio(fft_frame, freqs, min(1720, max_freq),
                                                                          min(4400, max_freq))

            self.feature_data['sfm'][i] = compute_spectral_flatness(fft_frame)
            self.feature_data['scf'][i] = compute_spectral_crest_factor(fft_frame)

    def update_plots(self):
        """Aktualizuje wykresy na podstawie wybranych parametrów."""
        # Czyścimy figurę
        self.fig.clear()

        # Określamy liczbę wykresów
        num_plots = sum([
            self.vol_var.get(),
            self.fc_var.get(),
            self.bw_var.get(),
            self.ersb_var.get(),
            self.sfm_var.get(),
            self.scf_var.get()
        ])

        if num_plots == 0:
            # Brak wybranych parametrów
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Wybierz co najmniej jeden parametr do wyświetlenia",
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            self.canvas.draw()
            return

        # Tworzymy subploty
        plot_index = 1
        time = self.feature_data['time']

        # Volume
        if self.vol_var.get():
            ax = self.fig.add_subplot(num_plots, 1, plot_index)
            ax.plot(time, self.feature_data['volume'], color=ColorScheme.ACCENT)
            ax.set_ylabel('Głośność')
            ax.set_title('Volume (Głośność)' if plot_index == 1 else '')
            ax.grid(True)
            plot_index += 1

        # Centroid częstotliwościowy
        if self.fc_var.get():
            ax = self.fig.add_subplot(num_plots, 1, plot_index)
            ax.plot(time, self.feature_data['fc'], color=ColorScheme.ACCENT)
            ax.set_ylabel('FC (Hz)')
            ax.set_title('Centroid częstotliwościowy' if plot_index == 1 else '')
            ax.grid(True)
            plot_index += 1

        # Szerokość pasma
        if self.bw_var.get():
            ax = self.fig.add_subplot(num_plots, 1, plot_index)
            ax.plot(time, self.feature_data['bw'], color=ColorScheme.ACCENT)
            ax.set_ylabel('BW (Hz)')
            ax.set_title('Szerokość pasma' if plot_index == 1 else '')
            ax.grid(True)
            plot_index += 1

        # Stosunki energii w pasmach
        if self.ersb_var.get():
            ax = self.fig.add_subplot(num_plots, 1, plot_index)
            ax.plot(time, self.feature_data['ersb1'], label='ERSB1 (0-630 Hz)',
                    color=ColorScheme.ORIGINAL_SIGNAL)
            ax.plot(time, self.feature_data['ersb2'], label='ERSB2 (630-1720 Hz)',
                    color=ColorScheme.WINDOWED_SIGNAL)
            ax.plot(time, self.feature_data['ersb3'], label='ERSB3 (1720-4400 Hz)',
                    color=ColorScheme.F0_PEAK_COLOR)
            ax.set_ylabel('ERSB')
            ax.set_title('Stosunki energii w pasmach częstotliwości' if plot_index == 1 else '')
            ax.legend()
            ax.grid(True)
            plot_index += 1

        # Płaskość widma
        if self.sfm_var.get():
            ax = self.fig.add_subplot(num_plots, 1, plot_index)
            ax.plot(time, self.feature_data['sfm'], color=ColorScheme.ACCENT)
            ax.set_ylabel('SFM')
            ax.set_title('Płaskość widma (SFM)' if plot_index == 1 else '')
            ax.grid(True)
            plot_index += 1

        # Współczynnik szczytu widma
        if self.scf_var.get():
            ax = self.fig.add_subplot(num_plots, 1, plot_index)
            ax.plot(time, self.feature_data['scf'], color=ColorScheme.ACCENT)
            ax.set_ylabel('SCF')
            ax.set_title('Współczynnik szczytu widma (SCF)' if plot_index == 1 else '')
            ax.grid(True)
            plot_index += 1

        # Dodajemy wspólną etykietę osi X tylko dla ostatniego wykresu
        ax.set_xlabel('Czas (s)')

        # Dostosowujemy układ
        self.fig.tight_layout()
        self.canvas.draw()