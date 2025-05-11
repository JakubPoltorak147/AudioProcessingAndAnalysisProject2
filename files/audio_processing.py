from features import compute_volume, compute_zcr


class BaseAudioProcessor:

    def detect_silence(self, data, fs, frame_size, silence_threshold):
        total_samples = len(data)
        frame_step = frame_size
        silence_regions = []
        in_silence = False
        start_silence = 0
        for i in range(0, total_samples, frame_step):
            frame = data[i:i + frame_size]
            if len(frame) == 0:
                continue
            # Używamy compute_volume do obliczenia RMS
            rms = compute_volume(frame)
            if rms < silence_threshold:
                if not in_silence:
                    in_silence = True
                    start_silence = i
            else:
                if in_silence:
                    silence_regions.append((start_silence, i))
                    in_silence = False
        if in_silence:
            silence_regions.append((start_silence, total_samples))
        return silence_regions


class VoicedAudioProcessor(BaseAudioProcessor):

    def detect_voiced_unvoiced(self, data, fs, frame_size, vol_threshold=0.02, zcr_threshold=0.3,
                               silence_threshold=0.001):

        total_samples = len(data)
        frame_step = frame_size
        results = []
        current_state = None
        start_idx = 0

        for i in range(0, total_samples, frame_step):
            frame = data[i:i + frame_size]
            if len(frame) == 0:
                continue

            # Obliczamy RMS ramki za pomocą compute_volume
            rms = compute_volume(frame)
            # Jeśli ramka jest cicha, kończymy bieżący segment (jeśli istnieje)
            if rms < silence_threshold:
                if current_state is not None:
                    results.append((start_idx, i, current_state))
                    current_state = None
                continue

            # Obliczamy ZCR dla niecichych ramek
            zcr_val = compute_zcr(frame)
            # Klasyfikacja: dźwięczny, gdy RMS > vol_threshold i ZCR < zcr_threshold
            is_voiced = (rms > vol_threshold and zcr_val < zcr_threshold)
            if current_state is None:
                current_state = is_voiced
                start_idx = i
            elif is_voiced != current_state:
                results.append((start_idx, i, current_state))
                current_state = is_voiced
                start_idx = i

        if current_state is not None:
            results.append((start_idx, total_samples, current_state))
        return results
