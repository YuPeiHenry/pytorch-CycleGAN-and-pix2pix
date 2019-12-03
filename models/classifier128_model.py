import torch
from .base_model import BaseModel
from . import networks
import numpy as np


class Classifier128Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode='categorical', preprocess='resize', load_size=512, input_nc=3, output_nc=3, no_flip=True)
        parser.set_defaults(save_epoch_freq=10, display_id=0, niter=20, niter_decay=0, lr=0.00001)
        parser.add_argument('--n_class', type=int, default=2, help='')
        parser.add_argument('--n_aggressive', type=int, default=10, help='')
        parser.add_argument('--threshold_increase', type=float, default=0.2, help='')
        parser.add_argument('--max_threshold', type=float, default=0.8, help='')
        parser.add_argument('--class_csv', type=str, default='class.csv', help='')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'D', 'Class']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['full_image', 'labelsG', 'labelsD', 'labelsC']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G', 'D', 'Classifier']
        # define networks
        self.netG = networks.StyledGenerator128(opt.input_nc, opt.output_nc, opt.n_class, 'classify')
        self.netD = networks.StyledDiscriminator(opt.input_nc, opt.n_class, 'classify')
        self.netClassifier = networks.StyledDiscriminator(opt.input_nc, opt.n_class, 'classify')
        self.netG = networks.init_net(self.netG, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD = networks.init_net(self.netD, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netClassifier = networks.init_net(self.netClassifier, opt.init_type, opt.init_gain, self.gpu_ids)

        self.colors = [[1, -1, -1], [1, 0, -1], [1, 1, -1], [-1, 1, -1], [-1, 1, 1], [-1, -1, 1]]
        self.colors[self.opt.n_class - 1] = [0, 0, 0]


        if self.isTrain:
            # define loss functions
            #weights = torch.FloatTensor([1 if i != opt.n_class - 1 else 1 / opt.n_class for i in range(opt.n_class)]).to(self.device)
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

    def backward_D(self, allow_background=True):
        predictD = self.netD(self.image)
        softmax = self.softmax(predictD).detach().cpu().numpy()
        given_score = softmax[self.indices1, self.indices2]
        label = torch.LongTensor([self.given_label[i] if given_score[i] >= self.threshold or not allow_background else self.opt.n_class - 1 for i in range(len(self.indices1))]).to(self.device)
        for i in range(len(given_score)):
            if given_score[i] > self.max_D[i]:
                self.max_D[i] = given_score[i]
        self.loss_D = self.criterion(predictD, label)
        self.loss_D.backward()

    def backward_G(self, allow_background=True):
        predictG = self.netG(self.image)
        softmax = self.softmax(predictG).detach().cpu().numpy()
        given_score = softmax[self.indices1, self.indices2]
        label = torch.LongTensor([self.given_label[i] if given_score[i] >= self.threshold or not allow_background else self.opt.n_class - 1 for i in range(len(self.indices1))]).to(self.device)
        for i in range(len(given_score)):
            if given_score[i] > self.max_G[i]:
                self.max_G[i] = given_score[i]
        self.loss_G = self.criterion(predictG, label)
        self.loss_G.backward()

    def backward_Class(self, allow_background=True):
        predictClass = self.netClassifier(self.image)
        softmax = self.softmax(predictClass).detach().cpu().numpy()
        given_score = softmax[self.indices1, self.indices2]
        label = torch.LongTensor([self.given_label[i] if given_score[i] >= self.threshold or not allow_background else self.opt.n_class - 1 for i in range(len(self.indices1))]).to(self.device)
        for i in range(len(given_score)):
            if given_score[i] > self.max_Class[i]:
                self.max_Class[i] = given_score[i]
        self.loss_Class = self.criterion(predictClass, label)
        self.loss_Class.backward()

    def optimize_parameters(self, allow_background=True):
        self.max_D = [0] * len(self.indices1)
        self.max_G = [0] * len(self.indices1)
        self.max_Class = [0] * len(self.indices1)
        
        self.cumulative_loss_G = 0
        self.cumulative_loss_D = 0
        self.cumulative_loss_C = 0

        interval = self.opt.load_size // 4
        half_interval = interval // 2
        startX = np.random.randint(interval + 2) - half_interval
        startY = np.random.randint(interval + 2) - half_interval
        for i in range(3):
            for j in range(3):
                X = min(startX + interval * (i + 1), self.opt.load_size - interval)
                Y = min(startY + interval * (j + 1), self.opt.load_size - interval)
                self.X_pos = i
                self.Y_pos = j
                self.image = self.full_image[:, :, X:X + interval, Y:Y + interval]
                self.run_optimizers(allow_background)
                self.cumulative_loss_G += self.loss_G.detach()
                self.cumulative_loss_D += self.loss_D.detach()
                self.cumulative_loss_C += self.loss_Class.detach()
            
        for i in range(2):
            for j in range(2):
                X = min(startX + interval * (i + 1), self.opt.load_size - interval)
                Y = min(startY + interval * (j + 1), self.opt.load_size - interval)
                self.X_pos = i + 0.5
                self.Y_pos = j + 0.5
                self.image = self.full_image[:, :, X:X + interval, Y:Y + interval]
                self.run_optimizers(allow_background)
                self.cumulative_loss_G += self.loss_G.detach()
                self.cumulative_loss_D += self.loss_D.detach()
                self.cumulative_loss_C += self.loss_Class.detach()

        self.loss_G = self.cumulative_loss_G / 13
        self.loss_D = self.cumulative_loss_D / 13
        self.loss_Class = self.cumulative_loss_C / 13

        rerun = False
        for i in range(len(self.max_D)):
            rerun = rerun or self.max_D[i] < self.threshold or self.max_G[i] < self.threshold or self.max_Class[i] < self.threshold
        if rerun and allow_background:
            self.optimize_parameters(False)

    def run_optimizers(self, allow_background=True):
        self.optimizer_D.zero_grad()
        self.backward_D(allow_background)
        self.optimizer_D.step()
        self.optimizer_G.zero_grad()
        self.backward_G(allow_background)
        self.optimizer_G.step()
        self.optimizer_netClassifier.zero_grad()
        self.backward_Class(allow_background)
        self.optimizer_netClassifier.step()

    def update_epoch_params(self, epoch):
        if epoch > self.opt.n_aggressive:
            self.threshold = min(self.opt.max_threshold, self.opt.threshold_increase * (epoch - self.opt.n_aggressive))

    def get_color_for_label(self, labels):
        return [self.colors[label] for label in labels]

    def compute_visuals(self, dataset=None):
        if dataset is None:
            return
        self.labelsG = self.full_image.clone()
        self.labelsD = self.full_image.clone()
        self.labelsC = self.full_image.clone()

        interval = self.opt.load_size // 4
        half_interval = interval // 2
        cumulative_softmaxG = 0
        cumulative_softmaxD = 0
        cumulative_softmaxC = 0
        for i in range(4):
            for j in range(4):
                X = interval * i
                Y = interval * j
                self.image = self.full_image[:, :, X:X + interval, Y:Y + interval]
                softmaxG = self.softmax(self.netG(self.image)).detach()
                softmaxD = self.softmax(self.netD(self.image)).detach()
                softmaxC = self.softmax(self.netClassifier(self.image))
                predictG = torch.argmax(softmaxG, dim=1).cpu().numpy()
                predictD = torch.argmax(softmaxD, dim=1).cpu().numpy()
                predictC = torch.argmax(softmaxC, dim=1).cpu().numpy()
                self.labelsG[:, :, X:X + interval, Y:Y + interval] = torch.FloatTensor(self.get_color_for_label(predictG)).view(-1, 3, 1, 1).repeat(1, 1, interval, interval).to(self.device)
                self.labelsD[:, :, X:X + interval, Y:Y + interval] = torch.FloatTensor(self.get_color_for_label(predictD)).view(-1, 3, 1, 1).repeat(1, 1, interval, interval).to(self.device)
                self.labelsC[:, :, X:X + interval, Y:Y + interval] = torch.FloatTensor(self.get_color_for_label(predictC)).view(-1, 3, 1, 1).repeat(1, 1, interval, interval).to(self.device)
                cumulative_softmaxG += softmaxG.detach()
                cumulative_softmaxD += softmaxD.detach()
                cumulative_softmaxC += softmaxC.detach()
        print(cumulative_softmaxG[0, :])
        print(cumulative_softmaxD[0, :])
        print(cumulative_softmaxC[0, :])
