# Official Implementation of "RadioDUN: A Physics-Inspired Deep Unfolding Network for Radio Map Estimation".

## Experiment Execution

Executing the following shell script for training RadioDUN under carsDPM:

```
bash train.sh
```
Executing the following Python script for testing RadioDUN under the **transmitter-known** conditions:
```
python main.py --model_phase test --model_path ./checkpoints/carsDPM_transmitter-known.pt --batch_size 128
```
Similarly, executing the following Python script for testing RadioDUN under the **transmitter-unknown** conditions:
```
python main.py --model_phase test --model_path ./checkpoints/carsDPM_transmitters-unknown.pt --batch_size 128
```

## Checkpoint Download

The pretrained checkpoints can be found at https://pan.baidu.com/s/1OEaMQz1pJ16W7UsG72dHlA. The extract code is b6fn.
