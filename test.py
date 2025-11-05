"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

import os
from pathlib import Path
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch
import re
from metrics import FIDSSIMPSNR  

from models.dense_norm_core import (
    init_dense_norm,
    use_dense_norm,
    not_use_dense_norm,
)

CODE2DIR = {'b': 'Benign', 'is': 'Insitu', 'iv': 'Invasive', 'n': 'Normal'}

def class_dir_from_path(p: str) -> str:
    name = os.path.basename(p).lower()
    m = re.search(r'(?:^|_)(b|is|iv|n)\d+', name)  # matches b0002, is0002, iv0002, n0002
    return CODE2DIR.get(m.group(1), 'Unknown') if m else 'Unknown'

def _infer_grid_shape(make_dataset_fn, opt):
    tmp = make_dataset_fn(opt)
    max_y = max_x = -1
    for d in tmp:
        # prefer anchors provided by your dataset
        ya = int(d.get('y_anchor', 0))
        xa = int(d.get('x_anchor', 0))
        if ya > max_y: max_y = ya
        if xa > max_x: max_x = xa
    # (+1 because anchors are indices)
    return (max_y + 1 if max_y >= 0 else 1, max_x + 1 if max_x >= 0 else 1)

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options
    opt.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    _meter = FIDSSIMPSNR(device=opt.device, use_fid=False)  # PSNR+SSIM only
    _meter.reset()
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    grid_h, grid_w = _infer_grid_shape(create_dataset, opt)
    # init_dense_norm(model.netG_A, y_anchor_num=grid_h, x_anchor_num=grid_w)
    # init_dense_norm(model.netG_B, y_anchor_num=grid_h, x_anchor_num=grid_w) 
    # collect_dataset = create_dataset(opt)   # fresh iterator
    model.eval()
    # with torch.no_grad():
    #     for data in collect_dataset:
    #         model.set_input(data)       
    #         model.forward()
    #         _ = model.netG_A(model.real_A)  
    not_use_dense_norm(model.netG_A)
    not_use_dense_norm(model.netG_B)
    # use_dense_norm(model.netG_A, padding = 1)
    # use_dense_norm(model.netG_B, padding = 1)

    # create a website
    web_dir = Path(opt.results_dir) / opt.name / f"{opt.phase}_{opt.epoch}"  
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = Path(f"{web_dir}_iter{opt.load_iter}")
    print(f"creating web directory {web_dir}")
    webpage = html.HTML(web_dir, f"Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}")
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image paths
        # PSNR / SSIM on cycle reconstructions (unpaired-safe)
        if hasattr(model, 'real_A') and hasattr(model, 'rec_A'):
            _meter.update_cycle_A(model.real_A, model.rec_A)  # A -> B -> A
        if hasattr(model, 'real_B') and hasattr(model, 'rec_B'):
            _meter.update_cycle_B(model.real_B, model.rec_B)  # B -> A -> B

        if i % 5 == 0:  # save images to an HTML file
            print(f"processing ({i:04d})-th image... {img_path}")
        if opt.direction == "AtoB":
            visuals = {'fake_B': visuals['fake_B']}
            if opt.gen_test:
                visuals = {'fake_A': visuals['fake_A']}
            
        elif opt.direction == "BtoA":
            visuals = {'fake_A': visuals['fake_A']}
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    metrics = _meter.compute()  # {'ssim_cycle_A':..., 'psnr_cycle_A':..., ...}
    # print to console (rounded) and save alongside results
    print("[test metrics]", {k: round(v, 4) for k, v in metrics.items()})
    out = "metrics_test.txt"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    write_header = (not os.path.exists(out)) or (os.path.getsize(out) == 0)
    with open(os.path.join(web_dir, out), "a") as f:
        for k, v in sorted(metrics.items()):
            f.write(f"{k}: {v:.6f}\n")
    webpage.save()  # save the HTML
