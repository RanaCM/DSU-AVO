cd /path/to/fairseq

python -m examples.speech_synthesis.evaluation.eval_asr --audio-header syn --text-header text --err-unit word --w2v-ckpt examples/speech_synthesis/evaluation/pretrained/wav2vec_big_960h.pt --w2v-dict-dir examples/speech_synthesis/evaluation/pretrained --raw-manifest /data07/junchen/Expressive-FastSpeech2/continuous-vp/output_aligner/result/Chem_16k_crop_untrimmed_avhubertv_huberttoken_dec1_res_acc/20k/eval.tsv --asr-dir /data07/junchen/Expressive-FastSpeech2/continuous-vp/output_aligner/result/Chem_16k_crop_untrimmed_avhubertv_huberttoken_dec1_res_acc/20k/asr
