import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class MinimumGAN2Model(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['Domain_D','Domain_G','D_A', 'G_A', 'idt_A', 'D_B', 'G_B', 'idt_B', 'minimum_reg']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        # if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
        #     visual_names_A.append('idt_B')
        #     visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout,
                                      opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt, minimum=True, SN=False)

        if self.isTrain:  # define discriminators
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias,
                                            self.gpu_ids, opt=opt,minimum=True)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert (opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.As = torch.cat([self.real_A]*2,dim=0)
        self.A_ds = torch.cat([torch.ones(self.real_A.size()[0]).cuda().long(),torch.zeros(self.real_A.size()[0]).cuda().long()])
        self.fake_Bs = self.netG(self.As, d=self.A_ds)  # G_A(A)
        self.fake_B, self.rec_A = self.fake_Bs[:self.real_A.size()[0]], self.fake_Bs[self.real_A.size()[0]:]
        # self.rec_A = self.netG(self.fake_B, d=torch.zeros(self.real_A.size()[0]).cuda().long())  # G_B(G_A(A))
        self.Bs = torch.cat([self.real_B] * 2, dim=0)
        self.B_ds = torch.cat([torch.zeros(self.real_B.size()[0]).cuda().long(), torch.ones(self.real_B.size()[0]).cuda().long()])
        self.fake_As = self.netG(self.Bs, d=self.B_ds)  # G_B(B)
        self.fake_A, self.rec_B = self.fake_As[:self.real_B.size()[0]], self.fake_As[self.real_B.size()[0]:]
        # self.rec_B = self.netG(self.fake_A, d=torch.ones(self.real_B.size()[0]).cuda().long())  # G_A(G_B(B))

    def backward_D_basic(self):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        ABs = torch.cat([self.real_A,self.fake_A.detach(),self.real_B,self.fake_B.detach()])
        pred, pred_d = self.netD(ABs)
        pred_real_A, pred_fake_A, pred_real_B, pred_fake_B = torch.chunk(pred,4,dim=0)
        pred_real = torch.cat([pred_real_A, pred_real_B])
        pred_fake = torch.cat([pred_fake_A, pred_fake_B])
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        d_s = torch.cat([torch.zeros(self.real_A.size()[0]).cuda().long(),torch.ones(self.real_B.size()[0]).cuda().long()])


        pred_d_A,_,pred_d_B,_ = torch.chunk(pred_d,4,dim=0)
        self.loss_Domain_D = torch.nn.functional.cross_entropy(torch.cat([pred_d_A, pred_d_B]), d_s)

        with torch.no_grad():
            self.loss_D_A = self.criterionGAN(pred_real_A, True) + self.criterionGAN(pred_fake_A, False)
            self.loss_D_B = self.criterionGAN(pred_real_B, True) + self.criterionGAN(pred_fake_B, False)
        # Combined loss and calculate gradients

    def backward_D(self):

        self.backward_D_basic()

        self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5 + self.loss_Domain_D

        self.loss_D.backward()


    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        self.loss_minimum_reg = 0
        for name, param in self.netG.named_parameters():
            if 'embed' in name:  # 'gaind' in name or 'biasd' in name: #
                self.loss_minimum_reg += torch.norm(param[0] - param[1], p=1)

        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            # self.rec_B = self.netG(self.real_B, d=torch.ones(self.real_B.shape[0]).cuda().long())
            self.loss_idt_B = self.criterionIdt(self.rec_B, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            # self.rec_A = self.netG(self.real_A, d=torch.zeros(self.real_A.shape[0]).cuda().long())
            self.loss_idt_A = self.criterionIdt(self.rec_A, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        pred_fake, pred_d = self.netD(torch.cat([self.fake_A,self.fake_B]))
        pred_fake_A, pred_fake_B = torch.chunk(pred_fake,2,dim=0)
        self.loss_G_A = self.criterionGAN(pred_fake_B, True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(pred_fake_A, True)

        d_s = torch.cat([torch.zeros(self.real_A.size()[0]).cuda().long(), torch.ones(self.real_B.size()[0]).cuda().long()])
        self.loss_Domain_G = torch.nn.functional.cross_entropy(pred_d, d_s)
        # Forward cycle loss || G_B(G_A(A)) - A||
        # self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # # Backward cycle loss || G_A(G_B(B)) - B||
        # self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_Domain_G + self.loss_G_A + self.loss_G_B + self.loss_idt_A + self.loss_idt_B + self.loss_minimum_reg * 0.01  # + self.loss_cycle_A + self.loss_cycle_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD],
                               False)  # Ds require no gradients when optimizing Gs  # set G_A and G_B's gradients to zero
        self.optimizer_G.zero_grad()
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
