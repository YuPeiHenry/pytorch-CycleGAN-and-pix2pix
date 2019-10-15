import torch
from .base_model import BaseModel
from . import networks
import numpy as np


class Classifier128(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', norm_G='adain', netG='unet_256', netD='', dataset_mode='categorical', load_size=512, input_nc=3, output_nc=3, downsample_mode='downsample', upsample_mode='upsample', preprocess='N.A.', no_flip=True, no_html=True)
        parser.add_argument('--class_csv', type=str, default='default.csv', help='')
        parser.add_argument('--n_class', type=int, default=2, help='')
        parser.add_argument('--n_aggressive', type=int, default=10, help='')
        parser.add_argument('--threshold_increase', type=float, default=0.2, help='')
        parser.add_argument('--max_threshold', type=float, default=0.8, help='')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'D', 'Class']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = []
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['U', 'G', 'D', 'Classifier']
        # define networks
        self.netU = networks.define_G(1, opt.input_nc, opt.ngf, opt.netG, opt.norm_G,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, downsample_mode=opt.downsample_mode, upsample_mode=opt.upsample_mode, upsample_method=opt.upsample_method)
        self.netG = networks.StyledGenerator128(opt.input_nc, opt.output_nc, opt.n_class, 'classify')
        self.netD = networks.StyledDiscriminator(opt.input_nc, opt.n_class, 'classify')
        self.netClassifier = networks.StyledDiscriminator(opt.input_nc, opt.n_class, 'classify')

        if self.isTrain:
            # define loss functions
            self.criterion = torch.nn.CrossEntropyLoss()
            self.threshold = 0
            self.softmax = torch.nn.Softmax(dim=1)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_netClassifier = torch.optim.Adam(self.netClassifier.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_netClassifier)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.full_image = input['image'].to(self.device)
        self.given_label = input['label']
        self.indices1 = np.array([i for i in range(self.given_label.size()[0])])
        self.indices2 = self.given_label.numpy()
        self.image_paths = input['path']

    def forward(self):
        return

    def backward_D(self):
        predictD = self.netD(self.image)
        softmax = self.softmax(predictD).detach().cpu().numpy()
        given_score = softmax[self.indices1, self.indices2]
        label = torch.Tensor([self.given_label[i] if given_score[i] >= self.threshold else self.opt.n_class - 1 for i in range(self.indices1)]).to(self.device)
        for i in range(len(given_score)):
            if given_score[i] > self.max_D[i]:
                self.max_D[i] = given_score[i]
                self.max_D_pos[i][0] = self.X_pos
                self.max_D_pos[i][1] = self.Y_pos
        self.loss_D = criterion(predictD, label)
        self.loss_D.backward()

    def backward_G(self):
        predictG = self.netG(self.image)
        softmax = self.softmax(predictG).detach().cpu().numpy()
        given_score = softmax[self.indices1, self.indices2]
        label = torch.Tensor([self.given_label[i] if given_score[i] >= self.threshold else self.opt.n_class - 1 for i in range(self.indices1)]).to(self.device)
        for i in range(len(given_score)):
            if given_score[i] > self.max_G[i]:
                self.max_G[i] = given_score[i]
                self.max_G_pos[i][0] = self.X_pos
                self.max_G_pos[i][1] = self.Y_pos
        self.loss_G = criterion(predictG, label)
        self.loss_G.backward()

    def backward_Class(self):
        predictClass = self.netClassifier(self.image)
        softmax = self.softmax(predictClass).detach().cpu().numpy()
        given_score = softmax[self.indices1, self.indices2]
        label = torch.Tensor([self.given_label[i] if given_score[i] >= self.threshold else self.opt.n_class - 1 for i in range(self.indices1)]).to(self.device)
        for i in range(len(given_score)):
            if given_score[i] > self.max_Class[i]:
                self.max_Class[i] = given_score[i]
                self.max_Class_pos[i][0] = self.X_pos
                self.max_Class_pos[i][1] = self.Y_pos
        self.loss_Class = criterion(predictClass, label)
        self.loss_Class.backward()

    def optimize_parameters(self):
        self.max_D = [0] * len(self.indices1)
        self.max_D_pos = [[0, 0]] * len(self.indices1)
        self.max_G = [0] * len(self.indices1)
        self.max_G_pos = [[0, 0]] * len(self.indices1)
        self.max_Class = [0] * len(self.indices1)
        self.max_Class_pos = [[0, 0]] * len(self.indices1)
        
        interval = self.opt.load_size
        half_interval = interval // 2
        startX = np.random.randint(interval + 2) - half_interval
        startY = np.random.randint(interval + 2) - half_interval
        for i in range(3):
            for j in range(3):
                X = startX + interval * i
                Y = startY + interval * i
                self.X_pos = i
                self.Y_pos = i
                self.image = self.full_image[:, :, X:X + interval, Y:Y + interval]
                self.run_optimizers()
            
        for i in range(2):
            for j in range(2):
                X = startX + half_interval + interval * i
                Y = startY + half_interval + interval * i
                self.X_pos = i + 0.5
                self.Y_pos = i + 0.5
                self.image = self.full_image[:, :, X:X + interval, Y:Y + interval]
                self.run_optimizers()

        for i in range(len(self.max_D)):
            if self.max_D[i] < self.threshold:
                self.optimizer_D.zero_grad()
                X = startX + int(interval * self.max_D_pos[0])
                Y = startY + int(interval * self.max_D_pos[1])
                image = self.full_image[i, :, X:X + interval, Y:Y + interval].unsqueeze(0)
                predictD = self.netD(image)
                label = torch.Tensor(self.given_label[i]).to(self.device).unsqueeze(0)
                self.loss_D = criterion(predictD, label) * 2
                self.loss_D.backward()
                self.optimizer_D.step()
            if self.max_G[i] < self.threshold:
                self.optimizer_G.zero_grad()
                X = startX + int(interval * self.max_G_pos[0])
                Y = startY + int(interval * self.max_G_pos[1])
                image = self.full_image[i, :, X:X + interval, Y:Y + interval].unsqueeze(0)
                predictG = self.netG(image)
                label = torch.Tensor(self.given_label[i]).to(self.device).unsqueeze(0)
                self.loss_G = criterion(predictG, label) * 2
                self.loss_G.backward()
                self.optimizer_G.step()
            if self.max_Class[i] < self.threshold:
                self.optimizer_Class.zero_grad()
                X = startX + int(interval * self.max_Class_pos[0])
                Y = startY + int(interval * self.max_Class_pos[1])
                image = self.full_image[i, :, X:X + interval, Y:Y + interval].unsqueeze(0)
                predictClass = self.netClass(image)
                label = torch.Tensor(self.given_label[i]).to(self.device).unsqueeze(0)
                self.loss_Class = criterion(predictClass, label) * 2
                self.loss_Class.backward()
                self.optimizer_Class.step()

    def run_optimizers(self):
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.optimizer_netClassifier.zero_grad()
        self.backward_Class()
        self.optimizer_netClassifier.step()

    def update_epoch_params(self, epoch):
        if epoch > self.opt.n_aggressive:
            self.threshold = max(self.opt.max_threshold, self.threshold + self.opt.threshold_increase * (epoch - self.opt.n_aggressive))
            
