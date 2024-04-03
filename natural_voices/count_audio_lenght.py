#  SPDX-License-Identifier: MIT
#  © 2024 ETH Zurich, see AUTHORS.txt for details

from pydub import AudioSegment
import os

def get_total_audio_length(folder_path):
    total_length = 0

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp3") or filename.endswith(".wav"):
            audio = AudioSegment.from_file(os.path.join(folder_path, filename))
            total_length += len(audio)

    # Convert milliseconds to seconds
    total_length_seconds = total_length / 1000

    return total_length_seconds

# Example usage
base = '/mnt/processed_data'
folder_paths = ["Urs_Gredig", 
                "Cornelia_Boesch", 
                "Binga_Silberschmidt", 
                "Andrea_Vetsch", 
                "Arthur_Honegger", 
                "Florian_Inhauser", 
                ]

for f in folder_paths:
    folder_path = os.path.join(base, f, 'split_audio')
    total_length_seconds = get_total_audio_length(folder_path)
    print(f"Total audio length of {folder_path}: {int(round(total_length_seconds//60, 0))} minutes, {int(round(total_length_seconds%60, 0))} seconds")
