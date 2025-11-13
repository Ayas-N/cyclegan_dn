python test.py --dataroot ../bach_cycle --model cycle_gan --phase test --load_size 512 --crop_size 512 --name cycle_test --norm dense --direction AtoB --results bach_dense1 --gen_test
mv bach_dense1/cycle_test/test_latest/images ../datasets/lstaingan_bach_test
cd ../colo_class
python test.py \
    --dataroot lstain_bach_train \
    --testroot  lstain_bach_test \
    --batch_size 4 \
    --num_workers 4 \
    --gpu_id 0 \
    --checkpoint lstain_bach_train_best_no_norm_.pth 
