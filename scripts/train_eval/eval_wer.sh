cd /path/to/fairseq

python -m examples.speech_synthesis.evaluation.eval_asr --audio-header syn --text-header text --err-unit word --w2v-ckpt examples/speech_synthesis/evaluation/pretrained/wav2vec_big_960h.pt --w2v-dict-dir examples/speech_synthesis/evaluation/pretrained --raw-manifest /path/to/dsu-avo/output_aligner/result/Chem_16k_crop_untrimmed_avhubertv_huberttoken_dec1_res_acc/20k/eval.tsv --asr-dir  /path/to/dsu-avo/output_aligner/result/Chem_16k_crop_untrimmed_avhubertv_huberttoken_dec1_res_acc/20k/asr
