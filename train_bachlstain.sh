set -ex
DATAROOT="../lstainbach_cycle"
NAME="bach_srch_dst"

python test.py --dataroot "$DATAROOT" --model cycle_gan --phase test --load_size 512 --crop_size 512 --name $NAME --norm dense --direction AtoB --results bach_1 --gen_test
mv  ./bach_1/$NAME/test_latest/images ../datasets/lstaingan_dst_test
cd ../colo_class
python test.py \
    --dataroot train_folder \
    --testroot $NAME \
    --batch_size 4 \
    --num_workers 4 \
    --gpu_id 1 \
    --checkpoint lstain_bach_train_best_no_norm_.pth 
