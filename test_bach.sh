python test.py --dataroot ../bach_cycle --model cycle_gan --phase test --load_size 512 --crop_size 512 --name cycle_test --norm dense --direction AtoB --results bach_dense1
python test.py --dataroot ../bach_cycle --model cycle_gan --phase train --load_size 512 --crop_size 512 --name cycle_test --norm dense --direction AtoB --results bach_dense2
python merge_images.py --src bach_dense1/cycle_test/test_latest/images bach_dense2/cycle_test/train_latest/images --dst bach_cycle --suffix train test

