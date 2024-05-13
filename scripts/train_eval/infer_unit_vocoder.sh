cd /path/to/speech-resynthesis

python inference.py \
--checkpoint_file checkpoints/chem_hubert100_nof0 \
-n -1 \
--input_code_file /path/to/dsu-avo/output_aligner/result/Chem_16k_crop_untrimmed_avhubertv_huberttoken_dec1_res_acc/20k/pred_tokens.txt \
--output_dir /path/to/dsu-avo/output_aligner/result/Chem_16k_crop_untrimmed_avhubertv_huberttoken_dec1_res_acc/20k/wav 
