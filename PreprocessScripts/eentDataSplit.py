# Revised on 10/3/2025.
import os
import argparse
import warnings
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
#-----------------------------------------------------------------------------------------------------------------------------#
eent_dat_path = os.path.join('Data', 'EENT', 'vowel-ai')
#-----------------------------------------------------------------------------------------------------------------------------#
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['EENT', 'SVD'], help='Specification of dataset in usage.')
    return parser.parse_args()
#-----------------------------------------------------------------------------------------------------------------------------#
def handle_eent(refresh=False):
    df_xlsx = pd.read_excel(os.path.join('Data', 'EENT', 'EENT_subjects_share_decrypted.xlsx'), engine='openpyxl', sheet_name=None)                  # Fetch all tables in xlsx file as dataFrames
    subj_dat = df_xlsx[list(df_xlsx.keys())[0]]
    id_dx_maplist = dict(zip(subj_dat['Final Random ID'], subj_dat['Diagnosis']))
    wav_info_list = []
    match = []

    for path in [f for f in os.listdir(eent_dat_path) if f.endswith('.wav')]:                                                                         # Extracting `.wav` file infos in EENT dataset
        title_wav = os.path.splitext(path)[0].split('_')
        intvl = title_wav[2].split('-')
        wav_info_list.append({
            'Diagnosis': title_wav[0],
            'ID': title_wav[1],
            'Start': intvl[0],
            'End': intvl[1]
        })
    df_wav = pd.DataFrame(wav_info_list)

    for _, row in df_wav.iterrows():                                                                                                                  # Now checking existing pairs in `.wav` file titles' ID with `.xlsx` "Table 1"'s "Final Random ID" as benchmark
        wav_dx = row['Diagnosis']
        wav_id = row['ID']
        if wav_id in id_dx_maplist:                                                                                                                   # Some audio `.wav` file diagnosis seem to be not matching with those ones in `.xlsx` summary, though they are supposed to be indicating the same and were merely abbrevicated.
            if id_dx_maplist[wav_id] == wav_dx \
            or (id_dx_maplist[wav_id] == "UVFP" and wav_dx == "VFP") \
            or (id_dx_maplist[wav_id] == "Hypofunctional Phonation" and wav_dx == "Functional Dysphonia") \
            or (id_dx_maplist[wav_id] == "Polyps (Relapse)" and wav_dx == "Polyps"):
                match.append(True)
        else: 
            print(f"wav_dx & wav_id pair not valid.")
            match.append(False)
    print(f"Valid percentage of wav_dx & wav_id pair: {sum(match) / len(match)}, " + f"n_existing_pair in './Data/EENT_subjects_share_decrypted.xlsx': {sum(match)}.")

    # Now splitting the data: train data with a proportion of .8 and test of .2
    if refresh:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_idxs, test_idxs in splitter.split(df_wav, groups=df_wav['ID']):
            train_wav_info = df_wav.iloc[train_idxs].copy()
            test_wav_info = df_wav.iloc[test_idxs].copy()

        train_wav_info['Label'] = train_wav_info['Diagnosis'].apply(lambda x: x == "Normal")
        train_wav_info['Path'] = train_wav_info.apply(lambda row: os.path.join('Data', 'EENT', 'vowel-ai', f"{row['Diagnosis']}_{row['ID']}_{row['Start']}-{row['End']}.wav"), axis=1); 
        train_wav_info = train_wav_info[['Path', 'Label']]; train_wav_info.to_csv(os.path.join('Data', 'EENT', 'train_path_dx.csv'), index=False)
        test_wav_info['Label'] = test_wav_info['Diagnosis'].apply(lambda x: x == "Normal")
        test_wav_info['Path'] = test_wav_info.apply(lambda row: os.path.join('Data', 'EENT', 'vowel-ai', f"{row['Diagnosis']}_{row['ID']}_{row['Start']}-{row['End']}.wav"), axis=1); 
        test_wav_info = test_wav_info[['Path', 'Label']]; test_wav_info.to_csv(os.path.join('Data', 'EENT', 'test_path_dx.csv'), index=False)
    return
#-----------------------------------------------------------------------------------------------------------------------------#
if __name__ == '__main__':
    args = parse_args()
    if args.dataset == "EENT": handle_eent(refresh=True)
    elif args.dataset == "SVD": pass