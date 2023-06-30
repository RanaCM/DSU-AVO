# High-Quality Automatic Voice Over with Accurate Alignment: Supervision through Self-Supervised Discrete Speech Units

This repository contains the source code and speech samples for the paper accepted to INTERSPEECH 2023 by Junchen Lu, Berrak Sisman, Mingyang Zhang, Haizhou Li: [paper on arxiv](https://arxiv.org/abs/2306.17005).

## Abstract

The goal of Automatic Voice Over (AVO) is to generate speech in sync with a silent video given its text script. Recent AVO frameworks built upon text-to-speech synthesis (TTS) have shown impressive results. However, the current AVO learning objective of acoustic feature reconstruction brings in indirect supervision for inter-modal alignment learning, thus limiting the synchronization performance and synthetic speech quality. To this end, we propose a novel AVO method leveraging the learning objective of self-supervised discrete speech unit prediction, which not only provides more direct supervision for the alignment learning, but also alleviates the mismatch between the text-video context and acoustic features. Experimental results show that our proposed method achieves remarkable lip-speech synchronization and high speech quality by outperforming baselines in both objective and subjective evaluations. 

## Speech Samples

Voice over video samples are available [here](https://ranacm.github.io/DSU-AVO/).

## Source Code

This repository is currently under construction. The source code for the experiments described in the paper will be made publicly available very soon. We appreciate your patience and interest in our work.

## Citation

If you find our work useful in your research, please consider citing our paper:
```
@article{lu2023highquality,
  title={High-Quality Automatic Voice Over with Accurate Alignment: Supervision through Self-Supervised Discrete Speech Units},
  author={Lu, Junchen and Sisman, Berrak and Zhang, Mingyang and Li, Haizhou},
  journal={arXiv preprint arXiv:2306.17005},
  year={2023}
}
```
