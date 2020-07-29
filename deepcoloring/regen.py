import torch
import torch.nn as nn
import torch.nn.init as winit
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
import torch.utils.data as data
# from multiprocessing import cpu_count
import math
import numpy as np
import os,sys
# import argparse
from PIL import Image
import base64
import io
from torchvision.transforms import ToTensor
from deepcoloring.test import img2base64
from deepcoloring.util import util
# from Dataset import Dataset


def batch2poli(batch, deg):
        if deg > 3: raise ValueError('deg > 2 not implemented yet.')
        if deg == 1: return batch
        r,g,b = torch.unsqueeze(batch[:,0,:,:],1), torch.unsqueeze(batch[:,1,:,:],1), torch.unsqueeze(batch[:,2,:,:],1)
        # r + g + b
        ris = batch
        if deg > 1:
            # r^2 + g^2 + b^2 + bg + br + gr
            ris = torch.cat((ris,r.pow(2)),1) # r^2
            ris = torch.cat((ris,g.pow(2)),1) # g^2
            ris = torch.cat((ris,b.pow(2)),1) # b^2
            ris = torch.cat((ris,b*g),1) # bg
            ris = torch.cat((ris,b*r),1) # br
            ris = torch.cat((ris,g*r),1) # gr
        if deg > 2:
            # (r^3 + g^3 + b^3) + (gb^2 + rb^2) + (bg^2 + rg^2) + (br^2  + gr^2) + bgr
            ris = torch.cat((ris,r.pow(3)),1) # r^3
            ris = torch.cat((ris,g.pow(3)),1) # g^3
            ris = torch.cat((ris,b.pow(3)),1) # b^3
            ris = torch.cat((ris,g*b.pow(2)),1) # gb^2
            ris = torch.cat((ris,r*b.pow(2)),1) # rb^2
            ris = torch.cat((ris,b*g.pow(2)),1) # bg^2
            ris = torch.cat((ris,r*g.pow(2)),1) # rg^2
            ris = torch.cat((ris,b*r.pow(2)),1) # br^2
            ris = torch.cat((ris,g*r.pow(2)),1) # gr^2
            ris = torch.cat((ris,b*g*r),1) # bgr
        return ris
def tensor2base64(img):
    buffered = BytesIO()
    convert_image = tensor_to_image(image)
    convert_image.save(buffered, format="png")
    img_str = base64.b64encode(buffered.getvalue())
    img_str = str(img_str)[2:-1]
    # os.remove(os.path.join(file_path, file_name))

    return img_str
# import sys
# sys.path.append(sys.path.append('./content/gdrive/My Drive/orijang/filter_removal/'))
 

class Net(nn.Module):
	def __init__(self, img_dim=[256,256], patchSize=32, nc=200, nf=2000, deg_poly_in=3, deg_poly_out=3):
		super(Net, self).__init__()
		self.img_dim = img_dim
		self.patchSize = patchSize
		self.deg_poly_in = deg_poly_in
		self.deg_poly_out = deg_poly_out
		# calculate number of channels
		self.nch_in = 3
		if deg_poly_in > 1: self.nch_in = self.nch_in + 6
		if deg_poly_in > 2: self.nch_in = self.nch_in + 10
		if deg_poly_in > 3: raise ValueError('deg > 3 not implemented yet.')
		self.nch_out = 3
		if deg_poly_out > 1: self.nch_out = self.nch_out + 6
		if deg_poly_out > 2: self.nch_out = self.nch_out + 10
		if deg_poly_out > 3: raise ValueError('deg > 3 not implemented yet.')
		# calculate number of patches
		self.hpatches = int(math.floor(img_dim[0]/patchSize))
		self.wpatches = int(math.floor(img_dim[1]/patchSize))
		self.npatches = self.hpatches *self.wpatches
		# create layers
		self.b1 = nn.BatchNorm2d(self.nch_in)
		self.c1 = nn.Conv2d(self.nch_in, nc, kernel_size=3, stride=2, padding=0)
		self.b2 = nn.BatchNorm2d(nc)
		self.c2 = nn.Conv2d(nc, nc, kernel_size=3, stride=2, padding=0)
		self.b3 = nn.BatchNorm2d(nc)
		self.c3 = nn.Conv2d(nc, nc, kernel_size=3, stride=2, padding=0)
		self.b4 = nn.BatchNorm2d(nc)
		self.c4 = nn.Conv2d(nc, nc, kernel_size=3, stride=2, padding=0)
		self.b5 = nn.BatchNorm2d(nc)
		self.c5 = nn.Conv2d(nc, nc, kernel_size=3, stride=2, padding=0)

		self.l1 = nn.Linear(nc*7*7, nf)
		self.l2 = nn.Linear(nf, nf)
		self.l3 = nn.Linear(nf, self.npatches*(self.nch_out*3+3)) # 2000 -> 21504   1->21

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def forward(self,x):
		# create poly input
		img = batch2poli(x, self.deg_poly_out)

		# convert net input to poly
		x = batch2poli(x, self.deg_poly_in)
		# convert poly input as array b x #px x 3 x 1
		img = img.view(-1,self.nch_out, self.img_dim[0]*self.img_dim[1],1)
		img = img.clone().permute(0,2,1,3)
		# calculate filters
		x = F.relu(self.c1(self.b1(x)))
		x = F.relu(self.c2(self.b2(x)))
		x = F.relu(self.c3(self.b3(x)))
		x = F.relu(self.c4(self.b4(x)))
		x = F.relu(self.c5(self.b5(x)))
		x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = self.l3(x)
		# x = x.view(-1, 9, self.hpatches, self.wpatches)
		x = x.view(-1, self.nch_out*3+3, self.hpatches, self.wpatches)
		# upsample
		x = F.upsample(x,scale_factor=self.patchSize,  mode='bilinear') # (36L, 18L+3, 256L, 256L)
		# unroll
		#x = x.view(-1,9,self.img_dim[0]*self.img_dim[1])
		x = x.view(-1,self.nch_out*3+3,self.img_dim[0]*self.img_dim[1]) # (36L, 18L+3, 65536L)
		# swap axes
		x = x.permute(0,2,1) # (36L, 65536L, 18L+3)
		# expand 3xnch
		x = x.contiguous().view(-1,x.size(1),3,self.nch_out+1) # (36L, 65536L, 3L, 6L+1)
		# add white channels to image
		w = Variable( torch.ones(img.size(0),img.size(1),1,img.size(3)) ).cuda()
		img = torch.cat((img,w),2)
		# prepare output variable
		ris = Variable(torch.zeros(img.size(0),img.size(1),3,img.size(3))).cuda()
		# multiply pixels for filters
		for bn in range(x.size(0)):
			ris[bn,:,:,:] = torch.bmm(x[bn,:,:,:].clone(),img[bn,:,:,:].clone())
		# convert images back to original shape
		ris = ris.permute(0,2,1,3)
		ris = ris.contiguous()
		ris = ris.view(-1,3, self.img_dim[0], self.img_dim[1])
		return ris

    



def test(image,net,ori_w,ori_h):
        
    # load checkpoint
        net.load_state_dict(torch.load('./deepcoloring/models/checkpoint2.pth')['state_dict'])
        # set network in test mode
        net.train(False)
        image = Variable(image, requires_grad=False)
        image.cuda()
        output = net(image)
        tensor2img = util.tensor2im(output)

        tensor2img=Image.fromarray(tensor2img)
        tensor2img=tensor2img.resize((ori_w,ori_h))
        tensor2img=img2base64(tensor2img)
        # save images

        # utils.save_image(output, './result/after_remove.jpg', nrow=1, padding=0)
        # utils.save_image(output, './deepcoloring/result/after_remove2_bohemian.png')

        return "data:image/jpeg;base64,"+tensor2img

