import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.utils.data as data
import os
import torch
def input_transform(size,scale=2):
  return transforms.Compose([
                    transforms.Resize((size[0]//scale,size[1]//scale),Image.BICUBIC),
                    transforms.ToTensor(),           
  ])
def target_transform():
  return transforms.ToTensor()

def load_img(path):
  img=Image.open(path)
  yuv=img.copy()
  yuv=yuv.convert('YCbCr')
  y,u,v=yuv.split()
  img=np.asarray(img)
  return y,u,v,img

#inherit dataset for collective dataset
class SRDataset(data.Dataset):
  def __init__(self,root_dir,input_transform=None,target_transform=None,
               fetch="train"):
    self.root_dir=root_dir
    self.input_transform=input_transform
    self.target_transform=target_transform
    self.fetch=fetch
    self.image_file_names=os.listdir(os.path.join(self.root_dir,
                                                  self.fetch.lower(),"HR"))

  def __len__(self):
    return len(self.image_file_names)
  
  def __getitem__(self,idx):
    if torch.is_tensor(idx):
      idx=idx.tolist()
    
    self.image_file_names.sort()
    HR,u,v,HRimg=load_img(os.path.join(self.root_dir,self.fetch.lower(),
                                       "HR",self.image_file_names[idx]))
    LR=HR.copy()
    if self.input_transform:
      LR=self.input_transform(LR)
      u=self.input_transform(u)
      v=self.input_transform(v)
    if self.target_transform:
      HR=self.target_transform(HR)
    sample={'LR':LR,'HR':HR}
   
    if self.fetch=='val' or self.fetch=='deploy' :
      return sample,u,v,HRimg
    else:
      return sample
