# AudioProcessingAndAnalysisProject2

Niniejsze repozytorium zawiera aplikację napisaną w Pythonie do analizy częstotliwościowej sygnałów dźwiękowych w formacie .wav. Aplikacja stanowi rozszerzenie [AudioProcessingAndAnalysisProject1](https://github.com/JakubPoltorak147/AudioProcessingAndAnalysisProject1), wzbogacając go o zaawansowane narzędzia do analizy w dziedzinie częstotliwości.

## Założenia i cele projektu

Projekt rozszerza poprzednią aplikację o następujące funkcjonalności:

1. **Implementacja analizy widmowej** - analiza sygnału w dziedzinie częstotliwości za pomocą Szybkiej Transformaty Fouriera (FFT)
2. **Zastosowanie różnych funkcji okienkowych** - okna prostokątne, trójkątne, Hamminga, Hanna i Blackmana oraz wizualizacja ich wpływu na widmo
3. **Implementacja spektrogramu** - wizualizacja zmian częstotliwości w czasie z konfigurowalnymi parametrami
4. **Analiza cepstralna** - wyznaczanie częstotliwości podstawowej (F0) przy użyciu cepstrum
5. **Parametryzacja w dziedzinie częstotliwości** - obliczanie parametrów takich jak centroid częstotliwościowy, szerokość pasma, płaskość widma i inne
6. **Wizualizacja parametrów na poziomie ramki** - obserwacja zmian parametrów częstotliwościowych w czasie

## Struktura repozytorium

```
AudioProcessingAndAnalysisProject2/
├── audio_files/
│   └── ... (opcjonalne pliki .wav używane w projekcie)
├── files/
│   ├── main.py                 # Główny punkt startowy aplikacji
│   ├── audio_app.py            # Moduł z klasą AudioApp (GUI, odtwarzanie, wykres przebiegu)
│   ├── audio_processing.py     # Klasy do przetwarzania audio (detekcja ciszy/dźwięczności)
│   ├── design.py               # Klasy i funkcje definiujące styl, kolory w GUI
│   ├── features.py             # Funkcje obliczające cechy sygnału (RMS, ZCR, STE, F0, itp.)
│   ├── features_window.py      # Moduł z klasą FeaturesWindow do wyświetlania wykresów cech
│   ├── frequency_analysis.py   # Moduł analizy częstotliwościowej z klasami FrequencyAnalysisWindow i CepstrumAnalysisWindow
│   ├── frequency_features.py   # Implementacja parametrów w dziedzinie częstotliwości
│   ├── frequency_features_window.py # Moduł wizualizacji parametrów częstotliwościowych
│   ├── windowing.py            # Implementacja funkcji okienkowych
│   └── cepstrum_analysis.py    # Funkcje do analizy cepstralnej
├── documentation/
│   ├── AiPD_dokumentacja_2_Jakub_Poltorak.pdf # Dokumentacja projektu
├── .gitignore
├── README.md                   # Niniejszy plik
└── requirements.txt            # Lista zależności (pip)
```

## Nowe moduły i klasy

### `frequency_analysis.py`
Moduł zawiera dwie główne klasy:
- **`FrequencyAnalysisWindow`** - okno GUI do analizy częstotliwościowej sygnału, umożliwiające wizualizację w dziedzinie czasu, analizę FFT oraz generowanie spektrogramu
- **`CepstrumAnalysisWindow`** - okno GUI do analizy cepstralnej, pozwalające na estymację i śledzenie częstotliwości podstawowej (F0) w sygnałach mowy

### `frequency_features_window.py`
- **`FrequencyFeaturesWindow`** - klasa implementująca interaktywne okno do wizualizacji parametrów częstotliwościowych sygnału w funkcji czasu (głośność, centroid częstotliwościowy, szerokość pasma, ERSB, płaskość widma, współczynnik szczytu)

### `frequency_features.py`
- Implementacja wszystkich parametrów w dziedzinie częstotliwości

### `windowing.py`
- Implementacja funkcji okienkowych (prostokątne, trójkątne, Hamminga, Hanna, Blackmana)
- Funkcje do aplikowania okien na ramki sygnału

### `cepstrum_analysis.py`
- Funkcje do obliczania cepstrum i estymacji częstotliwości podstawowej

## Analiza częstotliwościowa

### Transformata Fouriera
Aplikacja wykorzystuje Szybką Transformatę Fouriera (FFT) do analizy spektralnej sygnału. Dla sygnałów dyskretnych stosowana jest Dyskretna Transformata Fouriera (DFT)

### Funkcje okienkowe
Zaimplementowano następujące funkcje okienkowe:
- **Okno prostokątne** - najlepsza rozdzielczość częstotliwościowa, ale największy przeciek widma
- **Okno trójkątne (Bartletta)** - kompromis między rozdzielczością a przeciekiem widma
- **Okno Hamminga** - dobry balans między rozdzielczością a tłumieniem listków bocznych
- **Okno Hanna** - podobne do Hamminga, ale z lepszym tłumieniem dalszych listków bocznych
- **Okno Blackmana** - najlepsza redukcja przecieku widma, kosztem rozdzielczości częstotliwościowej

### Spektrogram
Spektrogram to dwuwymiarowa reprezentacja sygnału w dziedzinie czas-częstotliwość. Parametry konfiguracyjne:
- **Długość ramki** - wpływa na rozdzielczość częstotliwościową
- **Przesunięcie między ramkami (hop)** - wpływa na rozdzielczość czasową
- **Nakładanie się ramek (overlap)**
- **Funkcja okienkowa** - wpływa na redukcję przecieku widma

### Analiza cepstralna
Aplikacja implementuje następujące kroki:
1. Podział sygnału na ramki i zastosowanie funkcji okienkowej
2. Obliczenie FFT dla każdej ramki
3. Obliczenie logarytmu wartości bezwzględnej FFT (logarytmiczne widmo amplitudowe)
4. Obliczenie odwrotnej FFT logarytmicznego widma amplitudowego (cepstrum)
5. Znalezienie piku w cepstrum w zakresie odpowiadającym oczekiwanej częstotliwości podstawowej
6. Konwersja znalezionej kwefrencji na częstotliwość podstawową

## Parametry częstotliwościowe

Aplikacja umożliwia obliczanie i wizualizację następujących parametrów:

1. **Głośność (Volume)** - miara całkowitej energii spektralnej sygnału
2. **Centroid częstotliwościowy (FC)** - "środek ciężkości" widma, związany z postrzeganą "jasnością" dźwięku
3. **Szerokość pasma (BW)** - miara rozrzutu energii widma wokół centroidu częstotliwościowego
4. **Energia w pasmach (BE) i stosunek energii w pasmach (ERSB)** - miary energii w określonych zakresach częstotliwości:
   - ERSB1: 0-630 Hz
   - ERSB2: 630-1720 Hz
   - ERSB3: 1720-4400 Hz
   - ERSB4: 4400-Nyquist Hz
5. **Płaskość widma (SFM)** - miara regularności rozkładu energii w widmie
6. **Współczynnik szczytu widma (SCF)** - stosunek maksymalnej wartości widma do jego wartości średniej

## Instalacja i uruchomienie

### Klonowanie repozytorium
```bash
git clone https://github.com/JakubPoltorak147/AudioProcessingAndAnalysisProject2.git
cd AudioProcessingAndAnalysisProject2
```

### Tworzenie i aktywacja wirtualnego środowiska (opcjonalnie, zalecane)
```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### Instalacja zależności
```bash
pip install -r requirements.txt
```

### Uruchomienie aplikacji
```bash
cd files
python main.py
```

## Główne funkcjonalności

Po uruchomieniu aplikacji użytkownik może:

1. **Wczytać plik WAV** - wybór dowolnego pliku dźwiękowego w formacie WAV
2. **Odtwarzać dźwięk** - z opcjami odtwarzania, pauzy i zatrzymania
3. **Wizualizować przebieg czasowy** - z opcjonalnym zaznaczeniem fragmentów ciszy, dźwięcznych i bezdźwięcznych
4. **Analizować cechy sygnału** - dostęp do wykresów cech z poprzedniej wersji aplikacji
5. **Przeprowadzać analizę częstotliwościową** - nowe okno z FFT, spektrogramem i konfigurowalnymi parametrami
6. **Przeprowadzać analizę cepstralną** - okno do analizy częstotliwości podstawowej
7. **Wizualizować parametry częstotliwościowe** - okno do analizy parametrów w dziedzinie częstotliwości

## Wyniki eksperymentów

Dokumentacja projektu zawiera wyniki eksperymentów przeprowadzonych przy użyciu aplikacji, w tym:

1. **Analiza parametrów częstotliwościowych dla różnych głosów i głosek**
2. **Analiza tonu podstawowego w głosach męskich i żeńskich**
3. **Analiza formantów w samogłoskach**
4. **Porównanie różnych funkcji okienkowych i ich wpływu na widmo**
5. **Badanie różnic między samogłoskami a spółgłoskami w analizie widmowej**

Szczegółowe wyniki i wnioski z tych eksperymentów dostępne są w pliku dokumentacji.

## Autor

Projekt opracowany przez Jakuba Półtoraka (@JakubPoltorak147).

## Dokumentacja

Szczegółowa dokumentacja teoretyczna i techniczna znajduje się w pliku `documentation/AiPD_dokumentacja_2_Jakub_Poltorak.pdf`.
