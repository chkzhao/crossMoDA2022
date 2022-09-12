"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

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
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
import torch
import numpy as np
import ntpath
import torch.distributed as dist
import torch
import socket
import time

def setup(rank, world_size, master_addr, master_port):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
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

    opt.name = os.path.join(opt.dataroot.strip('../data/'),
                            opt.model + str(opt.lambda_minimum) + '_' + str(opt.batch_size) + '_' + str(opt.crop_size)
                            + '_' + opt.direction + ('_SN' if opt.SN else '')
                            + (('_theta_mix' if opt.theta_mix else '') + '_' + opt.netG + '_' + opt.netD))
    # hard-code some parameters for test
    opt.num_threads = 10   # test code only supports num_threads = 1
    opt.batch_size = 20    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)      # create a model given opt.model and other options
    # create a webpage for viewing the results
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    with torch.no_grad():
        for i, data in enumerate(dataset):
            if i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)  # regular setup: load and print networks; create schedulers
                model.parallelize()
                if opt.eval:
                    model.eval()
            if i >= opt.num_test:  # only apply our model to opt.num_test images.
                break
            model.set_input(data)  # unpack data from data loader
            model.test()  # run inference
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()  # get image paths

            if i % 5 == 0:  # save images to an HTML file
                print('processing (%04d)-th image... %s' % (i, img_path))
            save_path = os.path.join(save_images(webpage, visuals, img_path, width=opt.display_winsize),'fake_B')
            # print(img_path[0])
            for k, path in enumerate(img_path):
                short_path = ntpath.basename(path)
                name = os.path.splitext(short_path)[0]
                np.save(os.path.join(save_path, name),
                        visuals['fake_B'].cpu().numpy()[k])

        webpage.save()  # save the HTML
