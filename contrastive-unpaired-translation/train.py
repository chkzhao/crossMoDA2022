import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import os

import torch.distributed as dist
import torch
import socket

def setup(rank, world_size, master_addr, master_port):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options

    opt.name = os.path.join(opt.dataroot.strip('../data/'), opt.model+str(opt.lambda_minimum)+'_' +str(opt.batch_size)+'_'+str(opt.crop_size)
                            +'_'+opt.direction+('_SN' if opt.SN else '')
                            +(('_theta_mix' if opt.theta_mix else '') + '_'+opt.netG + '_' +opt.netD))

    if opt.rank == 0:
        master_addr = socket.gethostbyname(socket.gethostname())
        master_file = open(opt.master_port + '_master_file.txt', 'w')
        master_file.write(master_addr)
        master_file.close()
        setup(opt.rank, opt.world_size, master_addr, opt.master_port)
    else:
        while not os.path.exists(opt.master_port + '_master_file.txt'):
            pass
        time.sleep(1)
        master_file = open(opt.master_port + '_master_file.txt', 'r')
        master_addr = master_file.readlines()[0]
        master_file.close()
        setup(opt.rank, opt.world_size, master_addr, opt.master_port)

    torch.manual_seed(2021 + opt.rank)

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.

    model = create_model(opt)      # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)

    if opt.rank == 0:
        visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
        opt.visualizer = visualizer
    total_iters = 0                # the total number of training iterations

    optimize_time = 0.1

    times = []
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        if opt.rank == 0:
            visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        dataset.set_epoch(epoch)
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)  # regular setup: load and print networks; create schedulers
                model.parallelize()
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            if opt.rank == 0:
                if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                    save_result = total_iters % opt.update_html_freq == 0
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                    if opt.display_id is None or opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    print(opt.name)  # it's useful to occasionally show the experiment name on console
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)

            iter_data_time = time.time()


        if opt.rank == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
        if opt.rank == 0:
            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                   # update learning rates at the end of every epoch.
