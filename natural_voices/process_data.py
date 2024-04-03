#  SPDX-License-Identifier: MIT
#  © 2024 ETH Zurich, see AUTHORS.txt for details

import argparse
import os
from math import floor
from tqdm import tqdm
import text
import json
import librosa
import numpy as np
import torch
import torchaudio.functional as F
from torchaudio import save as save_audio
from pydub import AudioSegment


def load_wav_to_torch(full_path):
  data, sampling_rate = librosa.load(full_path)
  data = librosa.to_mono(data)
  # sampling_rate, data = read(full_path)
  return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def process_filelist(args, wav_filelist, json_filelist, split):

    filepaths_and_text = []

    total_time = 0

    for wav_path, json_path in tqdm(zip(wav_filelist, json_filelist), total=len(wav_list), desc=f'{total_time}'):
        audio, sr = load_wav_to_torch(wav_path)
        new_audio = F.resample(audio, sr, 16000).unsqueeze(0)
        total_time += new_audio.shape[1]/16000
        save_audio(wav_path, new_audio.cpu(), 16000)

        # Check language
        with open(json_path, "r") as json_file:
            trans = json.load(json_file)
            txt = trans["text"]
            cleaned_text = text._clean_text(txt, args.text_cleaners)
            filepaths_and_text.append([wav_path, cleaned_text])
        
        if args.time_limit is not None:
            if total_time >= args.time_limit * 60:
                break

    new_filelist = f'filelists/{args.dataset_name}_{split}_filelist.txt'
    with open(new_filelist, "w", encoding="utf-8") as f:
      f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])

    return


def split_filelist(file_list, split=(0.9, 0.1, 0.0)):
    train_split, val_split, test_split = split

    train_split_index = floor(len(file_list) * train_split)
    train_split = file_list[:train_split_index]

    val_split_index = train_split_index + floor(len(file_list) * val_split)
    val_split = file_list[train_split_index:val_split_index]


    test_split = file_list[val_split_index:]

    return train_split, val_split, test_split

if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="./")
    parser.add_argument("--datapath", type=str, default="/mnt/processed_data")
    parser.add_argument("--lang", type=str, default="deu")
    parser.add_argument("--time_limit", type=int, help='Time limit of the dataset in minutes', default=None)

    args = parser.parse_args()

    # Choose cleaners
    if args.lang == "eng":
        args.text_cleaners = ["english_cleaners2"]
        
    else: 
        args.text_cleaners = ["basic_cleaners"]

    splits = ['train', 'val', 'test']

    # get all files in the dataset
    wav_list = [os.path.join(args.datapath, args.dataset_name, 'split_audio', f) for f in sorted(os.listdir(os.path.join(args.datapath, args.dataset_name, 'split_audio')))]
    json_list = [os.path.join(args.datapath, args.dataset_name, 'transcription/audio_transcriptions/', f) for f in sorted(os.listdir(os.path.join(args.datapath, args.dataset_name, 'transcription/audio_transcriptions/')))]

    # split into train val test
    wav_filelists = split_filelist(wav_list, split=(0.9, 0.1, 0.0))
    json_filelists = split_filelist(json_list, split=(0.9, 0.1, 0.0))

    for split, wav_filelist, json_filelist in zip(splits, wav_filelists, json_filelists):
        # process each split
        process_filelist(args, wav_filelist, json_filelist, split)
