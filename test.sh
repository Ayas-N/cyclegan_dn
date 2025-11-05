python test.py --dataroot ../lstained_cycle --model cycle_gan --phase test --load_size 256 --crop_size 256 --name lstain_test --norm instance --direction BtoA --results lstain_1
python split_nct_by_class.py --src lstain_1/lstain_test/test_latest/images/Unknown --dst ../datasets/lstained_nct_test
cd ../colo_class
python test.py \
    --dataroot lstain_train \
    --testroot lstained_nct_test \
    --batch_size 4 \
    --num_workers 4 \
    --gpu_id 0 \
    --checkpoint lstain_train_best_no_norm_.pth 

