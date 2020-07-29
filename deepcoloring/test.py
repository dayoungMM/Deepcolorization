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
from io import BytesIO

# 텐서를 이미지로 바꾸는 함수
def tensor_to_image(tensor):
  tensor = tensor*255
  print("안녕 : ", tensor)
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return Image.fromarray(tensor)

def tensor2base64(img):
    buffered = BytesIO()
    convert_image = tensor_to_image(img)
    convert_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    img_str = str(img_str)[2:-1]
    # os.remove(os.path.join(file_path, file_name))

    return img_str

def img2base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    img_str = str(img_str)[2:-1]
    # os.remove(os.path.join(file_path, file_name))

    return img_str

def colorization(image_base64_encoded, xy_location):
  print("컬러라이제이션")
  sample_p = .03125
  to_visualize = ['gray', 'hint', 'hint_ab', 'fake_entr', 'real', 'fake_reg', 'real_ab', 'fake_ab_reg', ]
  S = 1

  opt = TrainOptions().parse()
  opt.load_model = True
  opt.nThreads = 1   # test code only supports nThreads = 1
  opt.batch_size = 1  # test code only supports batch_size = 1
  opt.display_id = -1  # no visdom display
  opt.phase = 'val'
  opt.dataroot = './dataset/ilsvrc2012/%s/' % opt.phase
  # opt.dataroot = '.\dataset\ilsvrc2012\%s\\' % opt.phase
  opt.serial_batches = True
  opt.aspect_ratio = 1.

  model = create_model(opt)
  model.setup(opt)
  model.eval()

  # create website
  web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
  webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

  # statistics
  psnrs = np.zeros((opt.how_many, S))
  entrs = np.zeros((opt.how_many, S))


  # #### base64를 tensor로 변환 #####
  base64_decoded = base64.b64decode(image_base64_encoded)
  image = Image.open(io.BytesIO(base64_decoded))
  ori_w, ori_h  = image.size
  image  = image.resize((256,256))
  image = ToTensor()(image).unsqueeze(0) 
  
  ## image를 tensor로 변환 ##
  # image = Image.open(base64_decoded)
  # image = ToTensor()(image).unsqueeze(0) 

  
  data_raw = [image, torch.Tensor([0])]

  data_raw[0] = data_raw[0].cuda()
  data_raw[0] = util.crop_mult(data_raw[0], mult=8)

  colorized_img = []
  # with no points
  for i in range(10):
  # for (pp, sample_p) in enumerate(sample_ps):
    # img_path = [string.replace('%08d_%.3f' % (i, sample_p), '.', 'p')]
    img_path = [('%08d_%.3f' % (i, sample_p)).replace('.', 'p')]
    data = util.get_colorization_data(data_raw, opt, ab_thresh=0., p=sample_p)
    user_hint = xy_location
    data = util.add_color_patches_rand_gt_user(data, user_hint, opt, p=sample_p)

    model.set_input(data)
    model.test(True)  # True means that losses will be computed
    visuals = util.get_subset_dict(model.get_current_visuals(), to_visualize)


    psnrs[i, 0] = util.calculate_psnr_np(util.tensor2im(visuals['real']), util.tensor2im(visuals['fake_reg']))
    entrs[i, 0] = model.get_current_losses()['G_entr']

    tensor2img = util.tensor2im(visuals['fake_reg'])

    tensor2img=Image.fromarray(tensor2img)
    # print(type(tensor2img))
    tensor2img=tensor2img.resize((ori_w,ori_h))
    tensor2img=img2base64(tensor2img)

    colorized_img.append("data:image/jpeg;base64,"+tensor2img)

  print("colorized_img 끝났다")
  return colorized_img