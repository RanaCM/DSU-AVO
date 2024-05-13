import os
import re
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import csv
import pandas as pd

from text import _clean_text

_square_brackets_re = re.compile(r"\[[\w\d\s]+\]")
_inv_square_brackets_re = re.compile(r"(.*?)\](.+?)\[(.*)")


def int_zfill(n):
    '''
    n : int wav #.

    Returns
    filled str with 3 digits.
    e.g., 12 -> '012'

    '''
    return str(n).zfill(3)


def get_sorted_items(items):
    # sort by key
    return sorted(items, key=lambda x:x[0])


def prepare_align(config):
    in_dir = config["path"]["corpus_path"] # Chem
    pipeline_dir = config["path"]["pipeline_path"] # pipeline
    wav_sub_dir = config["path"]["wav_sub_dir"] # pycrop
    text_path = config["path"]["data_splits_path"] # data_splits
    out_dir = config["path"]["raw_path"]
    out_sub_dir = "speakers"
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]

    os.makedirs(os.path.join(out_dir), exist_ok=True)
    filelist_fixed = open(f'{out_dir}/filelist.txt', 'w', encoding='utf-8')
    speaker_info = dict()
    spk = 'Chem'
    speaker_info[spk] = {'gender': 'C'} # 'C' stands for Chem
    
    text_dict = {}
    for split_file in os.listdir(os.path.join(in_dir, text_path)):
        with open(os.path.join(in_dir, text_path, split_file), "r") as f:
            for line in f.readlines():
                basename, raw_text = line.split("|")
                text_dict[basename] = raw_text 
    
    for ses in tqdm(next(os.walk(os.path.join(in_dir, pipeline_dir)))[1]):
        for num in os.listdir(os.path.join(in_dir, pipeline_dir, ses, wav_sub_dir)):

            num_path = os.path.join(in_dir, pipeline_dir, ses, wav_sub_dir, num)
            wav_path = os.path.join(num_path, "audio_16k_crop.wav")
            base_name = "{}-{}".format(ses, num)
            if not base_name in text_dict.keys():
                continue
            if not os.path.exists(wav_path):
                print("audio_16k_crop.wav not found: {}".format(wav_path))
                continue

            # Build Text
            text = text_dict[base_name]
            text = re.sub(_square_brackets_re, "", text)
            text = ' '.join(text.split())
            text = _clean_text(text, cleaners)

            os.makedirs(os.path.join(out_dir, out_sub_dir), exist_ok=True)
            wav, _ = librosa.load(wav_path, sampling_rate)
            wav = wav / max(abs(wav)) * max_wav_value
            wavfile.write(
                os.path.join(out_dir, out_sub_dir, "{}.wav".format(base_name)),
                sampling_rate,
                wav.astype(np.int16),
            )
            with open(
                os.path.join(out_dir, out_sub_dir, "{}.lab".format(base_name)),
                "w",
            ) as f1:
                f1.write(text)
            
            # Filelist
            filelist_fixed.write("|".join([base_name, text, spk]) + "\n")
    filelist_fixed.close()

    # Save Speaker Info
    with open(f'{out_dir}/speaker_info.txt', 'w', encoding='utf-8') as f:
        for spk, spk_info in get_sorted_items(speaker_info.items()):
            gender = spk_info['gender']
            f.write(f'{spk}|{gender}\n')
            
def prepare_align_avhubert(config):
    # breakpoint()
    in_dir = config["path"]["corpus_path"] # /data07/junchen/LRS3
    filelist_path = config["path"]["filelist_path"] # /data07/junchen/LRS3/file.list
    label_path = config["path"]["label_path"] # /data07/junchen/LRS3/label.list
    wav_sub_dir = config["path"]["wav_sub_dir"] # audio
    data_partition = config["path"]["data_partition"] # short-pretrain
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    out_dir = config["path"]["raw_path"]

    os.makedirs(os.path.join(out_dir), exist_ok=True)
    filelist_fixed = open(f'{out_dir}/filelist.txt', 'w', encoding='utf-8')
    speaker_info, speaker_done = dict(), set()
    
    full_filelist = []
    with open(filelist_path) as f:
        for line in f.readlines():
            # if not par in data_partition:
            #     continue
            full_filelist.append([line.strip().split("/")]) # [partition, spk, name]
    with open(label_path) as f:
        for i, line in enumerate(f.readlines()):
            full_filelist[i].append(line) # filelist[i]: [[par, spk, name], label]
    filelist = []
    for line in full_filelist:
        if line[0][0] in data_partition:
            filelist.append(line)
    
    for line in tqdm(filelist):

        # Build Text
        par = line[0][0]
        spk = line[0][1]
        name = line[0][2]
        text = line[1]
        base_name = "{}_{}".format(spk, name)

        if spk not in speaker_done:
            speaker_info[spk] = {
                'gender': 'C' # 'C' stands for Chem
            }
            speaker_done.add(spk)

        text = re.sub(_square_brackets_re, "", text)
        text = ' '.join(text.split())
        text = _clean_text(text, cleaners)

        with open(
            os.path.join(in_dir, wav_sub_dir, par, spk, "{}.lab".format(name)),
            "w",
        ) as f1:
            f1.write(text)
        
        # Filelist
        filelist_fixed.write("|".join([base_name, text, spk]) + "\n")
    filelist_fixed.close()

    # Save Speaker Info
    with open(f'{out_dir}/speaker_info.txt', 'w', encoding='utf-8') as f:
        for spk, spk_info in get_sorted_items(speaker_info.items()):
            gender = spk_info['gender']
            f.write(f'{spk}|{gender}\n')
