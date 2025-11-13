set -ex
DATAROOT="../nct_cycle"
NAME="nct_cycle"
RESULT="nct"
# python train.py --dataroot "$DATAROOT" --name $NAME --model cycle_gan --phase train --no_dropout --load_size 256 --crop_size 256 --use_wandb --batch_size 4 --debug_num_batches 1000 --eval_freq 2000 --val_phase test

python test.py --dataroot "$DATAROOT" --model cycle_gan --phase test --load_size 256 --crop_size 256 --name $NAME --norm dense --direction AtoB --results $RESULT --gen_test
mv  ./$RESULT/$NAME/test_latest/images ../datasets/$NAME
cd ../colo_class
python test.py \
    --dataroot train_png \
    --testroot $NAME \
    --batch_size 4 \
    --num_workers 4 \
    --gpu_id 1 \
    --checkpoint train_png_best_no_norm_.pth 
