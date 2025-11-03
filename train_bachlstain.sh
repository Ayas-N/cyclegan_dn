set -ex
DATAROOT="../lstainbach_cycle"
torchrun --nproc_per_node=2 train.py --dataroot "$DATAROOT" --name cycle_test --model cycle_gan --phase train --no_dropout --load_size 512 --crop_size 512 --use_wandb --batch_size 4 --debug_num_batches 1000 --eval_freq 2000 --val_phase test

python test.py --dataroot "$DATAROOT" --model cycle_gan --phase test --load_size 512 --crop_size 512 --name cycle_test --norm dense --direction AtoB --results bach_dense1
python test.py --dataroot "$DATAROOT" --model cycle_gan --phase train --load_size 512 --crop_size 512 --name cycle_test --norm dense --direction AtoB --results bach_dense2
python merge_images.py --src bach_dense1/cycle_test/test_latest/images bach_dense2/cycle_test/train_latest/images --dst ../datasets/lstainbach_cycle --suffix train test
cd ../colo_class
bash run_bach.sh
