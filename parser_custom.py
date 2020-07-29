import os
from deepcoloring.options.train_options import TrainOptions
from deepcoloring.models import create_model
from deepcoloring.util.visualizer import save_images
from deepcoloring.util import html
import io
import string
import torch
import torchvision
import torchvision.transforms as transforms
import base64
import os
from deepcoloring.util import util
from IPython import embed
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor

class Parser_custom(object):
    def __init__(self):
        self.opt = None
        self.model = None
    def __new__(self):
        if not hasattr(self, 'instance'):
            self.instance = super(Parser_custom, self).__new__(self)
        return self.instance
    def Setup(self):
        self.opt = TrainOptions().parse()
        self.opt.load_model = True
        self.opt.nThreads = 1   # test code only supports nThreads = 1
        self.opt.batch_size = 1  # test code only supports batch_size = 1
        self.opt.display_id = -1  # no visdom display
        self.opt.phase = 'val'
        self.opt.dataroot = './dataset/ilsvrc2012/%s/' % self.opt.phase

        self.opt.serial_batches = True
        self.opt.aspect_ratio = 1.
        self.model = create_model(self.opt)
        self.model.setup(self.opt)
        self.model.eval()
    def GetInstance(self):
        return self.instance
    def GetOpt(self):
        return self.opt
    def GetModel(self):
        return self.model