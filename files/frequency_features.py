import numpy as np


def compute_volume_frequency(spectrum):
    """
    Obliczanie głośności (Volume) w dziedzinie częstotliwości.

    Volume(n) = (1/N) * sum(S_n^2(k))

    Args:
        spectrum: Widmo częstotliwościowe ramki.

    Returns:
        Wartość głośności.
    """
    return np.mean(np.abs(spectrum) ** 2)


def compute_frequency_centroid(spectrum, freqs):
    """
    Obliczanie centroidu częstotliwościowego (FC).

    FC(n) = sum(ω*S_n(ω)) / sum(S_n(ω))

    Args:
        spectrum: Widmo częstotliwościowe ramki.
        freqs: Tablica częstotliwości odpowiadających binów widma.

    Returns:
        Wartość centroidu częstotliwościowego.
    """
    magnitude = np.abs(spectrum)
    return np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)


def compute_bandwidth(spectrum, freqs, centroid=None):
    """
    Obliczanie efektywnej szerokości pasma (BW).

    BW^2(n) = sum((ω-FC(n))^2 * S_n^2(ω)) / sum(S_n^2(ω))

    Args:
        spectrum: Widmo częstotliwościowe ramki.
        freqs: Tablica częstotliwości odpowiadających binów widma.
        centroid: Opcjonalnie prekalkulowany centroid częstotliwościowy.

    Returns:
        Wartość efektywnej szerokości pasma.
    """
    if centroid is None:
        centroid = compute_frequency_centroid(spectrum, freqs)

    magnitude_squared = np.abs(spectrum) ** 2
    return np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitude_squared) / (np.sum(magnitude_squared) + 1e-10))


def compute_band_energy(spectrum, freqs, f0, f1):
    """
    Obliczanie energii w paśmie częstotliwości (BE).

    BE(t) = integral(S_t^2(f)df) / integral(w(t)dt)

    Args:
        spectrum: Widmo częstotliwościowe ramki.
        freqs: Tablica częstotliwości odpowiadających binów widma.
        f0: Dolna granica pasma częstotliwości.
        f1: Górna granica pasma częstotliwości.

    Returns:
        Energia w określonym paśmie częstotliwości.
    """
    mask = (freqs >= f0) & (freqs <= f1)
    return np.sum(np.abs(spectrum[mask]) ** 2)


def compute_band_energy_ratio(spectrum, freqs, f0, f1):
    """
    Obliczanie stosunku energii w paśmie częstotliwości (BER/ERSB).

    ERSB_{[f0,f1]}(t) = BE_{[f0,f1]}(t) / Vol(t)

    Args:
        spectrum: Widmo częstotliwościowe ramki.
        freqs: Tablica częstotliwości odpowiadających binów widma.
        f0: Dolna granica pasma częstotliwości.
        f1: Górna granica pasma częstotliwości.

    Returns:
        Stosunek energii w określonym paśmie do całkowitej energii.
    """
    band_energy = compute_band_energy(spectrum, freqs, f0, f1)
    total_energy = compute_volume_frequency(spectrum)
    return band_energy / (total_energy + 1e-10)


def compute_spectral_flatness(spectrum):
    """
    Obliczanie płaskości widma (SFM).

    SFM(b,n) = [Product(S_n^2(i))]^(1/(ih(b)-il(b)+1)) / [(1/(ih(b)-il(b)+1)) * Sum(S_n^2(i))]

    Args:
        spectrum: Widmo częstotliwościowe ramki.

    Returns:
        Wartość płaskości widma (SFM).
    """
    magnitude_squared = np.abs(spectrum) ** 2
    if np.sum(magnitude_squared) <= 1e-10:
        return 1.0  # Zgodnie ze standardem MPEG7

    # Unikamy log(0) przez dodanie małej wartości
    magnitude_squared = magnitude_squared + 1e-10

    geometric_mean = np.exp(np.mean(np.log(magnitude_squared)))
    arithmetic_mean = np.mean(magnitude_squared)

    return geometric_mean / arithmetic_mean


def compute_spectral_crest_factor(spectrum):
    """
    Obliczanie współczynnika Crest Factor widma (SCF).

    SCF(b,n) = max_i(S_n^2(i)) / [(1/(ih(b)-il(b)+1)) * Sum(S_n^2(i))]

    Args:
        spectrum: Widmo częstotliwościowe ramki.

    Returns:
        Wartość współczynnika Crest Factor widma (SCF).
    """
    magnitude_squared = np.abs(spectrum) ** 2
    if np.sum(magnitude_squared) <= 1e-10:
        return 1.0

    max_value = np.max(magnitude_squared)
    mean_value = np.mean(magnitude_squared)

    return max_value / mean_value