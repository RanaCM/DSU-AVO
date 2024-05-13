import os
import random
import json

import tgt
import librosa
import numpy as np
from tqdm import tqdm

random.seed(1234)

class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.dataset = config["dataset"]
        self.corpus_path = config["path"]["corpus_path"]
        self.sub_dir = config["path"]["sub_dir"]
        self.speakers = dict()
        self.speakers = self.load_speaker_dict()
        self.raw_path = config["path"]["raw_path"]
        self.in_dir = os.path.join(config["path"]["raw_path"], self.sub_dir)
        self.in_dir_lip = config["path"]["in_dir_lip"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]
        self.frame_rate = config["preprocessing"]["video"]["frame_rate"]
        self.trim = config["preprocessing"]["trim"]
        self.use_data_splits = config["preprocessing"]["use_data_splits"]
        if self.use_data_splits:
            self.data_splits_path = config["path"]["data_splits_path"]

    def load_speaker_dict(self):
        spk_dir = os.path.join(self.config["path"]["raw_path"], 'speaker_info.txt')
        spk_dict = dict()
        with open(spk_dir, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()):
                spk_id = line.split("|")[0]
                spk_dict[spk_id] = i
        return spk_dict

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "lip")), exist_ok=True)

        print("Processing Data ...")
        out = list()

        speakers = self.speakers.copy()
        for i, wav_name in enumerate(tqdm(os.listdir(self.in_dir))): 
            if ".wav" not in wav_name:
                continue

            basename = wav_name.split(".")[0]
            tg_path = os.path.join(
                self.out_dir, "TextGrid", "{}.TextGrid".format(basename)
            )
            if os.path.exists(tg_path):
                info = self.process_utterance(basename)
                out.append(info)

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

    def process_utterance(self, basename):
        wav_path = os.path.join(self.in_dir, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, "{}.lab".format(basename))
        session = basename[:-4]
        lip_path = os.path.join(self.in_dir_lip, session, "{}.npy".format(basename[-3:]))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", "{}.TextGrid".format(basename)
        )
        speaker = "Chem"

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path, include_empty_intervals=True) 

        if self.trim:
            phone, duration, start, end = self.get_alignment(
                textgrid.get_tier_by_name("phones")
            )
            assert len(phone) == len(duration)
            text = "{" + " ".join(phone) + "}"
            if start >= end:
                return None

            # Read and trim wav files
            wav, sr = librosa.load(wav_path, sr=self.sampling_rate)
            if self.sampling_rate != sr:
                wav = librosa.resample(wav, sr, self.sampling_rate)
            wav = wav[
                int(self.sampling_rate * start) : int(self.sampling_rate * end)
            ].astype(np.float32)
            
            start_idx = int(self.frame_rate * start)
            end_idx = int(np.ceil(self.frame_rate * end))
            lip = np.load(lip_path)
            lip = lip[
                start_idx : min(end_idx, lip.shape[0]), :
            ].astype(np.float32) 
            
        else:
            phone, duration, start, end = self.get_alignment_no_trim(
                textgrid.get_tier_by_name("phones")
            )
            assert len(phone) == len(duration)
            text = "{" + " ".join(phone) + "}"
            if start >= end:
                return None

            # Read wav files
            wav, sr = librosa.load(wav_path, sr=self.sampling_rate)
            if self.sampling_rate != sr:
                wav = librosa.resample(wav, sr, self.sampling_rate)

            # trim tailing visual embedding
            end_idx = int(np.ceil(self.frame_rate * end))
            lip = np.load(lip_path)
            lip = lip[: min(end_idx, lip.shape[0]), :].astype(np.float32)

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        # Save files
        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)
        lip_filename = "{}-lip-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "lip", lip_filename), lip) # TODO: change lip to sync

        return (
            "|".join([basename, speaker, text, raw_text])
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text
            if p == "":
                p = "sil"

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def get_alignment_no_trim(self, tier):

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text
            if p == "":
                p = "sil"
            phones.append(p)
            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )
            end_time = e

        return phones, durations, start_time, end_time
    
    def generate_data_splits(self):
        if self.use_data_splits:
            # parse pre-assigned data splits
            print("Use pre-assgined data splits from {}: ".format(os.path.join(self.corpus_path, self.data_splits_path)))
            data_splits = {}
            for split_file in os.listdir(os.path.join(self.corpus_path, self.data_splits_path)):
                print(split_file)
                split_name = split_file.split(".")[0]
                assert split_name in ["train", "val", "test"]
                data_splits[split_name] = []
                with open(os.path.join(self.corpus_path, self.data_splits_path, split_file), "r") as f:
                    for line in f.readlines():
                        data_splits[split_name].append(line.split("|")[0])
            
            # parse auto-generated train.txt and val.txt in build_from_path
            data_splits_info = {}
            for split_name in data_splits.keys():
                data_splits_info[split_name] = []
            for split_file in ["train_ori.txt", "val_ori.txt"]:
                with open(os.path.join(self.out_dir, split_file), "r") as f:
                    for line in f.readlines():
                        basename = line.split("|")[0]
                        for split_name in data_splits.keys():
                            if basename in data_splits[split_name]:
                                data_splits_info[split_name].append(line)
                                data_splits[split_name].remove(basename)
            for split_name in data_splits.keys():
                with open(os.path.join(self.out_dir, "{}.txt".format(split_name)), "w", encoding="utf-8") as f:
                    f.writelines(data_splits_info[split_name])
                print(split_name)
                print(len(data_splits[split_name]))
