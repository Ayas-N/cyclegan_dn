python test.py --dataroot ../lstained_cycle --model cycle_gan --phase test --load_size 256 --crop_size 256 --name lstain_test --norm instance --direction AtoB --results lstain_1 --epoch 30
python test.py --dataroot ../lstained_cycle --model cycle_gan --phase train --load_size 256 --crop_size 256 --name lstain_test --norm instance --direction AtoB --results lstain_2 --epoch 30
python merge_images.py --src lstain_1/lstain_test/test_latest/images lstain_2/lstain_test/train_latest/images --dst lstained_cycleed --suffix train test
python split_nct_by_class.py --src ./lstained_cycleed/Unknown --dst ../datasets/lstain_cycled30_train
