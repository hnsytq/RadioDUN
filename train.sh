# Training RadioDUN under carsDPM. The number of samples is 9 and the transmitter position is available.
python main.py --num_epochs 100 --sample_num 9 --model_phase train \
 --gain_dirs carsDPM --num_block 3 --para_num 3 --dim 16 --output_dir ./carsDPM_s9/ \
 --data_dir ./data/RadioMapSeer_Indices_mix.pkl
