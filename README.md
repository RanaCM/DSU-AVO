# High-Quality Automatic Voice Over with Accurate Alignment: Supervision through Self-Supervised Discrete Speech Units

This repository contains the source code and speech samples for the [paper](https://www.isca-speech.org/archive/interspeech_2023/lu23f_interspeech.html) accepted to INTERSPEECH 2023 by Junchen Lu, Berrak Sisman, Mingyang Zhang, Haizhou Li.

## Abstract

The goal of Automatic Voice Over (AVO) is to generate speech in sync with a silent video given its text script. Recent AVO frameworks built upon text-to-speech synthesis (TTS) have shown impressive results. However, the current AVO learning objective of acoustic feature reconstruction brings in indirect supervision for inter-modal alignment learning, thus limiting the synchronization performance and synthetic speech quality. To this end, we propose a novel AVO method leveraging the learning objective of self-supervised discrete speech unit prediction, which not only provides more direct supervision for the alignment learning, but also alleviates the mismatch between the text-video context and acoustic features. Experimental results show that our proposed method achieves remarkable lip-speech synchronization and high speech quality by outperforming baselines in both objective and subjective evaluations. 

## Speech Samples

Voice over video samples are available [here](https://ranacm.github.io/DSU-AVO/).

## Getting Started

### Prerequisites

* Python >= 3.6
* PyTorch v1.7
* Install dependencies
  ```bash
  git clone https://github.com/ranacm/DSU-AVO.git
  cd DSU-AVO
  pip install -r requirements.txt
  ```
### Data Preparation
- The full list of Chem dataset can be found at [Lip2Wav dataset](https://github.com/Rudrabha/Lip2Wav/tree/master/Dataset). Download and preprocess the dataset follow [Neural Dubber paper](https://arxiv.org/abs/2110.08243). For preprocessed data, please send an inquiry to the author at junchen@u.nus.edu.
- To quantize the speech audio data with HuBERT, change the first line of ```hubert_tokenizer/chem_manifest.txt``` to audio dataset path, put the pretrained model weights under ```hubert_tokenizer/pretrained_models``` and use the commands provided below. More details and pretrained quantizer models can be found at [GSLM code](https://github.com/pytorch/fairseq/tree/master/examples/textless_nlp/gslm).
  ```
  cd hubert_tokenizer
  python quantize_with_kmeans.py \
      --feature_type hubert \
      --kmeans_model_path ../hubert_tokenizer/pretrained_models/km.bin \
      --acoustic_model_path ../hubert_tokenizer/pretrained_models/hubert_base_ls960.pt \
      --layer 6 \
      --manifest_path ../hubert_tokenizer/chem_manifest.txt \
      --out_quantized_file_path Chem_hubert100.txt \
      --extension ".wav"
  ```
- For visual data preparation, follow the instructions described in [AV-HuBERT preparation](https://github.com/facebookresearch/av_hubert/tree/main/avhubert/preparation).

### Unit Vocoder
- To train the unit vocoder with Chem dataset follow the instructions described in [Speech Resynthesis](https://github.com/facebookresearch/speech-resynthesis). Unit vocoder config is provided at ```unit_vocoder/Chem_hubert100.json```. The pretrained model can be downloaded [here](https://drive.google.com/file/d/1-bqZlo3hsNjia8mYllBlHJKX-kZbVLA5/view?usp=share_link).

## Training
This repository is currently under construction. The source code for  experiments described in our paper will be made publicly available very soon. We appreciate your patience and interest in our work.

## Citation

If you find our work interesting, please consider citing our paper:
```
@inproceedings{lu23f_interspeech,
  author={Junchen Lu and Berrak Sisman and Mingyang Zhang and Haizhou Li},
  title={{High-Quality Automatic Voice Over with Accurate Alignment: Supervision through Self-Supervised Discrete Speech Units}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
  pages={5536--5540},
  doi={10.21437/Interspeech.2023-2179}
}
```

## References
- [ming024's FastSpeech2 implementation](https://github.com/ming024/FastSpeech2/tree/master)
- [Speech Resynthesis](https://github.com/facebookresearch/speech-resynthesis)
- [SyncNet](https://github.com/joonson/syncnet_python)