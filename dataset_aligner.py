import json
import math
import os
from tqdm import tqdm

import numpy as np
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.tools import pad_1D, pad_2D


class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, model_config, train_config, sort=False, drop_last=False
    ):

        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.max_seq_len = model_config["max_seq_len"]
        self.ignore_max_seq_len = model_config["ignore_max_seq_len"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.con_path = preprocess_config["path"]["con_path"]

        self.basename, self.speaker, self.text, self.raw_text, self.aux_data = self.process_meta(filename)
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last


    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        conname = basename
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        
        sync_path = os.path.join(
            self.preprocessed_path,
            "sync",
            "{}-{}-{}.npy".format(speaker, "sync", basename),
        )
        sync = np.load(sync_path)

        con_path = os.path.join(
            self.con_path,
            speaker,
            "{}.npy".format(conname),
        )
        con = np.load(con_path)

        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "sync": sync,
            "con": con,
            "duration": duration,
        }

        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            breakpoint()
            name = []
            speaker = []
            text = []
            raw_text = []
            aux_data = []
            for line in tqdm(f.readlines()):
                line_split = line.strip("\n").split("|")
                n, s, t, r = line_split[:4]
                if not self.ignore_max_seq_len: 
                    conname = n 
                    con_path = os.path.join(
                        self.con_path,
                        s,
                        "{}.npy".format(conname),
                    )
                    con = np.load(con_path)
                    if con.shape[0] > self.max_seq_len:
                        continue
                a = "|".join(line_split[4:])
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
                aux_data.append(a)
            return name, speaker, text, raw_text, aux_data

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        syncs = [data[idx]["sync"] for idx in idxs] 
        cons = [data[idx]["con"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        sync_lens = np.array([sync.shape[0] for sync in syncs])
        con_lens = np.array([con.shape[0] for con in cons])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        syncs = pad_2D(syncs)
        cons = pad_1D(cons).astype('int64') 
        durations = pad_1D(durations)
        

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            syncs, 
            sync_lens,
            max(sync_lens),
            cons,
            con_lens,
            max(con_lens),
            durations, 
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output



if __name__ == "__main__":
    # Test
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from utils.tools import to_device_aligner

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    preprocess_config = yaml.load(
        open("./config/Chem_16k_crop_untrimmed_avhubertv_huberttoken_dec1_res_acc/preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(
        open("./config/Chem_16k_crop_untrimmed_avhubertv_huberttoken_dec1_res_acc/model.yaml", "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("./config/Chem_16k_crop_untrimmed_avhubertv_huberttoken_dec1_res_acc/train.yaml", "r"), Loader=yaml.FullLoader
    )

    breakpoint()
    train_dataset = Dataset(
        "train.txt", preprocess_config, model_config, train_config, sort=True, drop_last=True
    )

    val_dataset = Dataset(
        "val.txt", preprocess_config, model_config, train_config, sort=False, drop_last=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"] * 4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["optimizer"]["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            to_device_aligner(batch, device)
            n_batch += 1
    print(
        "Training set  with size {} is composed of {} batches.".format(
            len(train_dataset), n_batch
        )
    )

    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            to_device_aligner(batch, device)
            n_batch += 1
    print(
        "Validation set  with size {} is composed of {} batches.".format(
            len(val_dataset), n_batch
        )
    )
