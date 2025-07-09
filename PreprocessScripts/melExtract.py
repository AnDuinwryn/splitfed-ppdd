import librosa
import numpy as np
import os
import pickle

import dataSegment
#-----------------------------------------------------------------------------------------------------------------------------#
sr_default = 44100
nm_pkl = "mel_2.pkl"
#-----------------------------------------------------------------------------------------------------------------------------#
def extract_mel_spectrogram_features(audio, sr, len_window=1024, len_hop=256, n_mels=128):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=len_window, hop_length=len_hop, n_mels=n_mels)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_delta = librosa.feature.delta(log_mel_spectrogram)
    mel_delta_delta = librosa.feature.delta(log_mel_spectrogram, order=2)
    features = np.stack([log_mel_spectrogram, mel_delta, mel_delta_delta], axis=0)
    return features
#-----------------------------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    mel_dat = []
    with open(os.path.join(dataSegment.processed_base_dir, dataSegment.nm_pkl), 'rb') as f:
        sliced_dat = pickle.load(f)
    for idx, row in enumerate(sliced_dat):
        mel_dat.append({'Feature': extract_mel_spectrogram_features(audio=(row['Segment']), sr=sr_default), 'Label': row['Label']})
    with open(dataSegment.prepare_dir(dataSegment.processed_base_dir, nm_pkl), 'wb') as f:
        pickle.dump(mel_dat, f, protocol=pickle.HIGHEST_PROTOCOL)
