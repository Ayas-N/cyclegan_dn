python test.py --dataroot ../lstain2norm --model cycle_gan --phase test --load_size 256 --crop_size 256 --name lstain2norm --norm instance --direction AtoB --results lstain2norm_1
python test.py --dataroot ../lstain2norm --model cycle_gan --phase train --load_size 256 --crop_size 256 --name lstain2norm --norm instance --direction AtoB --results lstain2norm_2 
python merge_images.py --src lstain2norm_1/lstain2norm/test_latest/images lstain2norm_2/lstain2norm/train_latest/images --dst lstained2 --suffix train test
python split_nct_by_class.py --src ./lstained2/Unknown --dst ../datasets/lstained2norm

