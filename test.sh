python test.py --dataroot ../lstain_cycle --model cycle_gan --phase test --load_size 512 --crop_size 512 --name lstain_test --norm dense --direction AtoB --results lstain_1
python test.py --dataroot ../lstain_cycle --model cycle_gan --phase train --load_size 512 --crop_size 512 --name lstain_test --norm dense --direction AtoB --results lstain_2
python merge_images.py --src lstain_1/lstain_test/test_latest/images lstain_1/lstain_test/train_latest/images --dst bach_cycle --suffix train test

