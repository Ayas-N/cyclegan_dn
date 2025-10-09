set -ex
python test.py --dataroot ../datasets/train_png --name cycle_test --model cycle_gan --phase test --no_dropout --monkey
