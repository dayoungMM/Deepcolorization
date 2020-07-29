from django.shortcuts import render
from django.http import HttpResponse
from django.urls import path
import numpy as np
from deepcoloring.test import *
from PIL import Image
from .forms import ImageForm
from deepcoloring.regen import *



def index(request):
    return render(request, 'deepcoloring/index.html')

def coloring(request):
    if request.method == "POST":
        uploaded_img = request.FILES['uploaded_img'] # 사진 받음
    
        model = Net().cuda()
        image = Image.open(uploaded_img)
        ori_w, ori_h  = image.size
        image = image.resize((256,256))
        image = ToTensor()(image).unsqueeze(0).cuda() 

        after_remove=test(image,model,ori_w, ori_h)

        return render(request, 'deepcoloring/removed.html', {"after_remove" : after_remove})
    else:
        return render(request, 'deepcoloring/coloring.html', {})



def palettepage(request):
    return render(request, 'deepcoloring/palettepage.html', {})

def filterON(request):
    if request.method == "POST":
        try:
            removed = request.POST['removed']
            return render(request, 'deepcoloring/filterON.html', {'removed' : removed})
        except :
            pal_img = request.POST.get('pal_img','') # 사진 받음
            pal_img=pal_img[23:]
            # 클릭 좌표 string로 받아서 list로 만들기
            xy_str = request.POST.get('xy_str')
            xy_location=xy_str.split("/")[:-1]
            if len(xy_location) > 0 :
                xy_location = np.array(xy_location).reshape(len(xy_location)//5,-1)

            # colorization
            colorized_img = colorization(pal_img, xy_location)

            return render(request, 'deepcoloring/colorized.html', {"colorized_img" : colorized_img})
    else:
        return render(request, 'deepcoloring/filterON.html', {'removed' : ''})