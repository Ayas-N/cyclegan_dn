set -ex
torchrun --nproc_per_node=2 train.py --dataroot ../lstained_cycle --name lstain_test --model cycle_gan --phase train --no_dropout --load_size 256 --crop_size 256 --use_wandb --batch_size 4
