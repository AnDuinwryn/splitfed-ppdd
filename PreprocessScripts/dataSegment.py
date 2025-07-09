import numpy as np
import os
import pandas as pd
import librosa
from pyAudioAnalysis import audioSegmentation as aS
import pickle
#-----------------------------------------------------------------------------------------------------------------------------#
path_wav_info = os.path.join('Data', 'EENT', 'test_path_dx.csv')
nm_pkl = "sliced_audio_2.pkl"
processed_base_dir = os.path.join('Data', 'EENT', 'Processed')
sr_default = 44100
fade_pts = int(.25 * sr_default)
segment_pts = int(1.5 * sr_default)
#-----------------------------------------------------------------------------------------------------------------------------#
def prepare_dir(base_dir: str, nm_file: str):
    if not os.path.exists(base_dir):
        print(f"Base directory \"{base_dir}\" didn't exist, now being created")
        os.makedirs(base_dir)
    else: print(f"Base directory \"{base_dir}\" already exists")
    return os.path.join(base_dir, nm_file)
#-----------------------------------------------------------------------------------------------------------------------------#
def sr2decimalPlaces(sr):
    frame_time = 1 / sr
    return len(format(frame_time, '.10f').lstrip('0')[1:].rstrip('0'))
#-----------------------------------------------------------------------------------------------------------------------------#
def round_intv2s_formatout(intv_list, sr):
    intvs = np.asarray(intv_list)
    if intvs.ndim != 2 or intvs.shape[1] != 2: raise ValueError(f"Expected shape (n, 2), but got {intvs.shape}")
    decimal_place_bd = sr2decimalPlaces(sr)
    for intv in intvs:
        print(f"From {str(np.round(intv[0], decimal_place_bd)).rjust(decimal_place_bd+2)} to {str(np.round(intv[1], decimal_place_bd)).rjust(decimal_place_bd+2)} in seconds.")
    return
#-----------------------------------------------------------------------------------------------------------------------------#
def remove_silence(csv_path):                                                   # Return all sliced audio data
    dat_sliced = []
    df_wav = pd.read_csv(csv_path)
    for _, row in df_wav.iterrows():
        segment_stack = []
        row_label = row['Label']
        audio, sr = librosa.load(row['Path'], sr=sr_default)
        # assert sr == sr_default, f"`.wav` file sampling rate expected to be 44100, yet {row['Path']} violates this discipline"
        intvs = aS.silence_removal(signal=audio, sampling_rate=sr, st_win=0.025, st_step=0.0125, weight=0.2, plot=False)
        
        for st, ed in intvs:
            st_idx = int(st * sr); ed_idx = int(ed * sr)
            if ed_idx - st_idx >= 2 * sr:
                st_idx += fade_pts; ed_idx -= fade_pts
                for i in range(st_idx, ed_idx, segment_pts):
                    segment_ed_idx = min(i + segment_pts, ed_idx)
                    if segment_ed_idx - i >= segment_pts:
                        segment_stack.append(audio[i: segment_ed_idx])
        # Dump
        for segment in segment_stack: dat_sliced.append({'Segment': segment, 'Label': row_label})
    return dat_sliced
#-----------------------------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    dat_sliced = remove_silence(path_wav_info)
    with open(prepare_dir(processed_base_dir, nm_pkl), 'wb') as f:
        pickle.dump(dat_sliced, f, protocol=pickle.HIGHEST_PROTOCOL)