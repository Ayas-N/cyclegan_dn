python test.py --dataroot ../lstain2norm --model cycle_gan --phase test --load_size 256 --crop_size 256 --name lstain2norm --norm instance --direction AtoB --results lstain2norm_1 --gen_test
mv ./lstain2norm_1/lstain2norm/test_latest/images ../datasets/lsnsrctodst_test  
