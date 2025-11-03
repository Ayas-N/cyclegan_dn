set -ex
torchrun --nproc_per_node=2 train.py --dataroot ../lstain2norm --name lstain2norm --model cycle_gan --phase train --no_dropout --load_size 256 --crop_size 256 --use_wandb --batch_size 4 --debug_num_batches 1000 --eval_freq 2000 --val_phase test


