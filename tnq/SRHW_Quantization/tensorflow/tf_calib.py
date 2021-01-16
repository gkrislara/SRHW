import PIL
from PIL import Image
import os
import numpy as np
import glob

calib_image_dir="/workspace/SRHW/calib/deploy/LR/"
batch_size=1
image_files=[f for f in glob.glob(calib_image_dir+"*.png")]
print(len(image_files))
def calib_input(iter):
    images=[]
    for index in range(0,batch_size):
        curimg=image_files[iter*batch_size+index]
        img=Image.open(curimg)
        img=img.convert('YCbCr')
        img,_,_=img.split()
        img_array=np.asarray(img,dtype=np.float32)
        if img_array.shape[0]>img_array.shape[1]:
            img_array=img_array.transpose(1,0)
        img_array=img_array[:,:,np.newaxis]
        img_array/=255.0
        images.append(img_array.tolist())
    return {'LR':images}
         
