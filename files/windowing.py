import numpy as np


def apply_window(frame, window_type):
    """
    Stosuje wybraną funkcję okienkową do ramki.

    Args:
        frame: Ramka sygnału (tablica 1D).
        window_type: Typ funkcji okienkowej ('rectangular', 'triangular', 'hamming', 'hann', 'blackman').

    Returns:
        Ramka po zastosowaniu funkcji okienkowej.
    """
    N = len(frame)

    if window_type == 'rectangular':
        window = np.ones(N)
    elif window_type == 'triangular':
        window = np.bartlett(N)
    elif window_type == 'hamming':
        window = np.hamming(N)
    elif window_type == 'hann':
        window = np.hanning(N)
    elif window_type == 'blackman':
        window = np.blackman(N)
    else:
        window = np.ones(N)  # Domyślnie prostokątne

    return frame * window


def get_window_type_name(window_type):
    """
    Zwraca polską nazwę funkcji okienkowej.

    Args:
        window_type: Nazwa funkcji okienkowej w kodzie.

    Returns:
        Polska nazwa funkcji okienkowej.
    """
    window_names = {
        'rectangular': 'Prostokątne',
        'triangular': 'Trójkątne (Bartlett)',
        'hamming': 'Hamminga',
        'hann': 'Hanna',
        'blackman': 'Blackmana'
    }

    return window_names.get(window_type, window_type.capitalize())