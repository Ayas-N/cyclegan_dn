"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

import time
import copy, torch
from metrics import FIDSSIMPSNR
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import init_ddp, cleanup_ddp

@torch.no_grad()
def evaluate(model, dataset, max_batches=50, use_fid=True):
    opt = TrainOptions().parse()  
    meter = FIDSSIMPSNR(device='cuda', use_fid=use_fid)
    meter.reset()
    # Many dataset wrappers in this repo support .reset(); safe to guard
    if hasattr(dataset, 'reset'):
        dataset.reset()

    for i, data in enumerate(dataset):
        if i >= max_batches:
            break
        # 1) prepare inputs
        model.set_input(data)   # expects dict with 'A','B'
        # 2) forward in inference mode
        model.test()            # runs generators only, no grads
        # 3) grab tensors in [-1,1]
        real_A = model.real_A
        real_B = model.real_B
        fake_B = model.fake_B   # A -> B
        rec_A  = model.rec_A    # A -> B -> A
        fake_A = model.fake_A   # B -> A
        rec_B  = model.rec_B    # B -> A -> B

        # cycle structure metrics
        meter.update_cycle_A(real_A, rec_A)
        meter.update_cycle_B(real_B, rec_B)

        # FID distribution metrics (unpaired)
        if use_fid:
            meter.update_fid_A2B(real_B, fake_B)
            meter.update_fid_B2A(real_A, fake_A)

    return meter.compute()


if __name__ == "__main__":
    opt = TrainOptions().parse()  # get training options
    opt.device = init_ddp()
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print(f"The number of training images = {dataset_size}")
    val_opt = copy.deepcopy(opt)
    val_opt.phase = getattr(opt, 'val_phase', 'test')
    val_opt.serial_batches = True      
    val_opt.max_dataset_size = 100000  
    val_dataset = create_dataset(val_opt)  # same factory you use for train


    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()
        # Set epoch for DistributedSampler
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            if 0 <= opt.debug_num_batches <= i:
                break

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cuda'
                metrics = evaluate(model, val_dataset,
                                max_batches=opt.eval_batches,
                                use_fid=opt.eval_use_fid)
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, total_iters, save_result)

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(total_iters, losses)

            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print(f"saving the latest model (epoch {epoch}, total_iters {total_iters})")
                save_suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

            if opt.eval_freq > 0 and (total_iters % opt.eval_freq == 0):
                metrics = evaluate(model, val_dataset,
                                max_batches=opt.eval_batches,
                                use_fid=opt.eval_use_fid)
                # print or plot
                print(f"[eval @ iters={total_iters}] " +
                      
                    " ".join([f"{k}={v:.4f}" for k,v in metrics.items()]))
                visualizer.log_eval_metrics(total_iters, metrics, prefix="eval")


            iter_data_time = time.time()

        model.update_learning_rate()  # update learning rates at the end of every epoch

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print(f"saving the model at the end of epoch {epoch}, iters {total_iters}")
            model.save_networks("latest")
            model.save_networks(epoch)

        print(f"End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time:.0f} sec")

    cleanup_ddp()
