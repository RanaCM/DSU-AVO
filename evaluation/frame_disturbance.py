# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 21:57:15 2021

@author: Lu Junchen
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy
from dtw import dtw
from numpy.linalg import norm
import math
import os
import csv

num_mels=80
num_freq = 2049
frame_shift_ms = 10
frame_length_ms = 40
sr = 16000
ref_level_db = 20
dct_type = 2
norm_type = 'ortho'

n_fft = (num_freq - 1) * 2
hop_length = int(frame_shift_ms / 1000 * sr)
win_length = int(frame_length_ms / 1000 * sr)


def linear_to_mel(spectrogram):
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels=80)
    return np.dot(mel_basis, spectrogram)

def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

def frame_disturbance_mel(mel_a, mel_b, mel_x, n_mfcc=20):
    
    mfcc_a = scipy.fftpack.dct(mel_a, axis=0, type=dct_type, norm=norm_type)[:n_mfcc]
    mfcc_b = scipy.fftpack.dct(mel_b, axis=0, type=dct_type, norm=norm_type)[:n_mfcc]
    mfcc_x = scipy.fftpack.dct(mel_x, axis=0, type=dct_type, norm=norm_type)[:n_mfcc]
    
    _, _, _, path_a = dtw(mfcc_a.T, mfcc_x.T, dist=lambda x, y: norm(x - y, ord=1))
    _, _, _, path_b = dtw(mfcc_b.T, mfcc_x.T, dist=lambda x, y: norm(x - y, ord=1))

    # calculate frame disturbance
    n_x = mfcc_x.shape[1]
    
    fd_sum_a = 0
    n_frame_a = len(path_a[0])
    for i in range(0, n_frame_a):
        fd_sum_a += (path_a[0][i] - path_a[1][i]) ** 2
    fd_a = math.sqrt(fd_sum_a / n_frame_a)
    
    fd_sum_b = 0
    n_frame_b = len(path_b[0])
    for i in range(0, n_frame_b):
        fd_sum_b += (path_b[0][i] - path_b[1][i]) ** 2
    fd_b = math.sqrt(fd_sum_b / n_frame_b)
    
    return fd_a, fd_b


def calculate_frame_disturbance_wav(wavpath_a, wavpath_sto, wavpath_b, wavpath):
    
    sr = 16000
    wav_a, _ = librosa.load(wavpath_a, sr=sr)
    wav_sto, _ = librosa.load(wavpath_sto, sr=sr)
    wav_b, _ = librosa.load(wavpath_b, sr=sr)
    wav, _ = librosa.load(wavpath, sr=sr)

    D = librosa.stft(y=wav_a, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mel_a = amp_to_db(linear_to_mel(np.abs(D))) - ref_level_db
    D = librosa.stft(y=wav_sto, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mel_sto = amp_to_db(linear_to_mel(np.abs(D))) - ref_level_db
    D = librosa.stft(y=wav_b, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mel_b = amp_to_db(linear_to_mel(np.abs(D))) - ref_level_db
    D = librosa.stft(y=wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mel_x = amp_to_db(linear_to_mel(np.abs(D))) - ref_level_db
    
    fd_a, fd_b = frame_disturbance_mel(mel_a, mel_b, mel_x)
    fd_sto, fd_b_2 = frame_disturbance_mel(mel_sto, mel_b, mel_x)
    assert fd_b == fd_b_2

    return fd_a, fd_sto, fd_b


def evaluate_wav_all():
    
    dirpath_1 = '/path/to/dsu-avo/output_aligner/result/Chem_16k_crop_untrimmed_avhubertv_huberttoken_dec1_res_acc/11k/wav' # mel
    dirpath_2 = '/path/to/dsu-avo/output_aligner/result/Chem_16k_crop_untrimmed_avhubertv_huberttoken_dec1_res_acc/18k/wav' # unit
    dirpath_3 = '/path/to/dsu-avo/output_aligner/result/Chem_16k_crop_untrimmed_avhubertv_huberttoken_dec1_res_acc/20k/wav'
    dirpath_gt = '/data07/junchen/Chem/gt_test/audio' # ground truth / trimmed original audios from GRID
    csvname = 'fd_test_dsu11_dsu18_dsu20.csv'
    
    dir_1_inc = os.listdir(dirpath_1)
    dir_1 = []
    for filename in dir_1_inc:
        if filename.endswith('.wav'):
            dir_1.append(filename)

    dir_2_inc = os.listdir(dirpath_2)
    dir_2 = []
    for filename in dir_2_inc:
        if filename.endswith('.wav'):
            dir_2.append(filename)

    dir_3_inc = os.listdir(dirpath_3)
    dir_3 = []
    for filename in dir_3_inc:
        if filename.endswith('.wav'):
            dir_3.append(filename)

    dir_gt = os.listdir(dirpath_gt)

    dir_1.sort()
    dir_2.sort()
    dir_3.sort()
    dir_gt.sort()
    
    rows_file = []
    rows_1 = []
    rows_2 = []
    rows_3 = []
    
    for file_1, file_2, file_3, file_gt in zip(dir_1, dir_2, dir_3, dir_gt):
        
        # breakpoint()
        # assert file_1[:15] == file_3[:15]
        assert file_3[:15] in file_1
        assert file_2[:15] == file_3[:15]
        assert file_3[:15] == file_gt[:15]
        
        wavpath_1 = os.path.join(dirpath_1, file_1)
        wavpath_2 = os.path.join(dirpath_2, file_2)
        wavpath_3 = os.path.join(dirpath_3, file_3)
        wavpath_gt = os.path.join(dirpath_gt, file_gt)
    
        print(file_1)
        fd_1, fd_2, fd_3 = calculate_frame_disturbance_wav(wavpath_1, wavpath_2, wavpath_3, wavpath_gt, trim_wav=False, trim_top_db=15, plot_mel=False, plot_path=False)
        rows_file.append(file_1)
        rows_1.append(fd_1)
        rows_2.append(fd_2)
        rows_3.append(fd_3)
        
    with open(csvname, 'w', newline='') as csvfile:
        
        writer = csv.writer(csvfile, dialect='excel')
        writer.writerow(['name', 'avo', 'tts', 'dsu'])
        writer.writerows(zip(rows_file, rows_1, rows_2, rows_3))
    

if __name__ == '__main__':
    
    evaluate_wav_all()