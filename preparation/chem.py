import re
import argparse
import yaml
import os
import shutil
import json
import librosa
import soundfile
from glob import glob
from tqdm import tqdm
from text import _clean_text, text_to_sequence
from g2p_en import G2p


def normalize_nonchar(text, inference=False):
    return re.sub(r"\{[^\w\s]?\}", "{sp}", text) if inference else\
            re.sub(r"[^\w\s|\']?", "", text)

def extract_nonen(preprocess_config):
    in_dir = preprocess_config["path"]["raw_path"]
    filelist = open(f'{in_dir}/nonen.txt', 'w', encoding='utf-8')

    count = 0
    nonen = set()
    print("Extract non english charactors...")
    with open(f'{in_dir}/filelist.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        total_count = len(lines)
        for line in tqdm(lines):
            wav = line.split('|')[0]
            text = line.split('|')[1]

            reg = re.compile("""[^ a-zA-Z~!.,?:`"'＂“‘’”’]+""")
            impurities = reg.findall(text)
            if len(impurities) == 0:
                count+=1
                continue
            norm = _clean_text(text, preprocess_config["preprocessing"]["text"]["text_cleaners"])
            impurities_str = ','.join(impurities)
            filelist.write(f'{norm}|{text}|{impurities_str}|{wav}\n')
            for imp in impurities:
                nonen.add(imp)
    filelist.close()
    print('Total {} non english charactors from {} lines'.format(len(nonen), total_count-count))
    print(sorted(list(nonen)))


def extract_lexicon(preprocess_config):
    """
    Extract lexicon and build grapheme-phoneme dictionary for MFA training
    """
    in_dir = preprocess_config["path"]["raw_path"]
    lexicon_path = preprocess_config["path"]["lexicon_path"]
    filelist = open(lexicon_path, 'a+', encoding='utf-8')

    # Load Lexicon Dictionary
    done = set()
    if os.path.isfile(lexicon_path):
        filelist.seek(0)
        for line in filelist.readlines():
            grapheme = line.split("\t")[0]
            done.add(grapheme)

    print("Extract lexicon...")
    g2p = G2p()
    for lab in tqdm(glob(f'{in_dir}/**/*.lab', recursive=True)):
        with open(lab, 'r', encoding='utf-8') as f:
            text = f.readline().strip("\n")
        text = normalize_nonchar(text)

        for grapheme in text.split(" "):
            if not grapheme in done:
                phoneme = " ".join(g2p(grapheme))
                filelist.write("{}\t{}\n".format(grapheme, phoneme))
                done.add(grapheme)
    filelist.close()

