from abc import ABC
import torch.nn as nn
import torch
from .patchnce import PatchNCELoss
import numpy as np
import torch.nn.functional as F
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangself.opt.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', 'nonsaturating']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        bs = prediction.size(0)
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'nonsaturating':
            if target_is_real:
                loss = F.softplus(-prediction).view(bs, -1).mean(dim=1)
            else:
                loss = F.softplus(prediction).view(bs, -1).mean(dim=1)
        return loss

class CUTModel(ABC):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    def __init__(self, opt, models, optimizer_G, optimizer_D):

        self.opt = opt

        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        self.optimizers = []
        self.scaler = torch.cuda.amp.GradScaler()
        self.autocast = torch.cuda.amp.autocast


        # define networks (both generator and discriminator)
        if self.opt.transfer_seg:
            self.netG, self.netD, self.netF, self.net_seg, self.loss_function = models
        else:
            self.netG, self.netD, self.netF = models

        # define loss functions
        self.criterionGAN = GANLoss('lsgan').cuda()
        self.criterionNCE = []

        for nce_layer in self.nce_layers:
            self.criterionNCE.append(PatchNCELoss(opt).cuda())

        self.criterionIdt = torch.nn.L1Loss().cuda()
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)


    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.a
        """
        self.set_input(data)
        self.forward()                     # compute fake images: G(A)
        with self.autocast():
            self.loss_D = self.compute_D_loss()
        self.scaler.scale(self.loss_D).backward()
        self.compute_G_loss()  # calculate graidents for G
        if self.opt.lambda_NCE > 0.0:
            if self.opt.sync_bn:
                process_group = torch.distributed.new_group()
                self.netF = nn.SyncBatchNorm.convert_sync_batchnorm(self.netF, process_group)
            self.netF = DDP(self.netF, broadcast_buffers=True)
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=2e-4, betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer_F)


    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()

        with self.autocast():
            self.loss_D = self.compute_D_loss()
        self.scaler.scale(self.loss_D).backward()
        average_gradients(self.netD)
        self.scaler.step(self.optimizer_D)
        self.scaler.update()
        # self.loss_D.backward()
        # self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        if self.opt.transfer_seg:
            self.set_requires_grad(self.net_seg, False)
        self.optimizer_G.zero_grad()
        self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        average_gradients(self.netG)
        average_gradients(self.netF)
        self.scaler.step(self.optimizer_G)
        self.scaler.step(self.optimizer_F)
        # self.optimizer_G.step()
        # self.optimizer_F.step()
        self.scaler.update()

        if self.opt.transfer_seg:
            self.set_requires_grad(self.net_seg,True)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_A = input[0]
        self.real_B = input[1]


        if self.opt.transfer_seg:
            self.label_A = input[2]

        self.epoch = input[3]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        with self.autocast():
            self.fake_B = self.netG(self.real_A)

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        self.loss_fake_seg = torch.zeros(1).cuda()
        if self.opt.transfer_seg and (self.epoch>= self.opt.warm_transfer):
            with self.autocast():
                out_fake = self.net_seg(self.fake_B)
                self.loss_fake_seg = self.loss_function(out_fake, self.label_A)
            self.scaler.scale(self.loss_fake_seg).backward(retain_graph=True)
            # self.loss_fake_seg.backward(retain_graph=True)
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            with self.autocast():
                pred_fake = self.netD(fake)
                self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
            self.scaler.scale(self.loss_G_GAN).backward(retain_graph=True)
            # self.loss_G_GAN.backward(retain_graph=True)
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            with self.autocast():
                self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
            self.scaler.scale(self.loss_NCE).backward()
            # self.loss_NCE.backward()
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            with self.autocast():
                self.idt_B = self.netG(self.real_B)
                self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            # self.loss_NCE_Y.backward()
            self.scaler.scale(self.loss_NCE_Y).backward()
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, encode_only=True)

        feat_k = self.netG(src, encode_only=True)

        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)

        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers