import numpy as np
from windowing import apply_window


def compute_cepstrum(frame, sample_rate, window_type='hamming'):
    """
    Oblicza rzeczywiste cepstrum ramki sygnału.

    Args:
        frame: Ramka sygnału (tablica 1D).
        sample_rate: Częstotliwość próbkowania sygnału.
        window_type: Typ funkcji okienkowej.

    Returns:
        Trójka (cepstrum, kwefrencja, logarytmiczne widmo).
    """
    # Stosujemy okno
    windowed_frame = apply_window(frame, window_type)

    # Obliczamy FFT
    spectrum = np.fft.rfft(windowed_frame)

    # Obliczamy logarytmiczne widmo amplitudowe
    log_spectrum = np.log(np.abs(spectrum) + 1e-10)  # Dodajemy małą wartość, aby uniknąć log(0)

    # Obliczamy odwrotną FFT logarytmicznego widma amplitudowego (rzeczywiste cepstrum)
    cepstrum = np.fft.irfft(log_spectrum)

    # Obliczamy oś kwefrencji (czasu)
    quefrency = np.arange(len(cepstrum)) / sample_rate

    return cepstrum, quefrency, log_spectrum


def estimate_f0_from_cepstrum(cepstrum, quefrency, min_f0=50, max_f0=500):
    """
    Estymuje częstotliwość podstawową z cepstrum.

    Args:
        cepstrum: Cepstrum sygnału.
        quefrency: Oś kwefrencji (czasu) odpowiadająca cepstrum.
        min_f0: Minimalna częstotliwość podstawowa do rozważenia (Hz).
        max_f0: Maksymalna częstotliwość podstawowa do rozważenia (Hz).

    Returns:
        Para (estymowana częstotliwość podstawowa, indeks piku).
    """
    # Konwertujemy min/max F0 na zakres kwefrencji
    min_quefrency = 1 / max_f0
    max_quefrency = 1 / min_f0

    # Znajdujemy indeksy odpowiadające zakresowi kwefrencji
    min_idx = np.argmin(np.abs(quefrency - min_quefrency))
    max_idx = np.argmin(np.abs(quefrency - max_quefrency))

    # Unikamy problemu z pustym zakresem
    if min_idx >= max_idx:
        min_idx = max(0, min_idx - 1)
        max_idx = min(len(cepstrum) - 1, max_idx + 1)

    # Znajdujemy szczyt w zakresie kwefrencji
    peak_idx = min_idx + np.argmax(cepstrum[min_idx:max_idx])

    # Konwertujemy szczytową kwefrencję na częstotliwość podstawową
    f0 = 1 / quefrency[peak_idx]

    return f0, peak_idx