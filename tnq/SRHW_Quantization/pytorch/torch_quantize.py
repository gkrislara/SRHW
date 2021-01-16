import random
import math
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm.notebook import tqdm
import time
import PIL
from PIL import Image
import torch.utils.data as data
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from models import SRHW
from pytorch_nndct.apis import torch_quantizer, dump_xmodel
from datasets import SRDataset,input_transform,target_transform
import argparse
import gc

torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)  
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:",device)
print("Torch Version:",torch.__version__)
print("Torch Vision Version:",torchvision.__version__)

#argument parser
parser =argparse.ArgumentParser()
parser.add_argument('--data_dir', default ='/workspace/tnq/SRHW_Quantization/calib/',
        help='Data set directory, when quant_mode=1, it is for calibration, while quant_mode=2 it is for evaluation')
parser.add_argument('--model_dir',default ='/workspace/tnq/SRHW_Quantization/ckpt/',
        help='Model directory')
parser.add_argument('--quant_mode', 
    default='calib', 
    choices=['float', 'calib', 'test'], 
    help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
parser.add_argument('--model_name',default='SRHW',
        help='specify class name of the model')
parser.add_argument('--ckpt',default='checkpoint',
        help='specify checkpoint name')
parser.add_argument('--batch_size',default=1,type=int,
        help='batch_size for quantised model')
parser.add_argument('--alpha',default=1,type=float,
        help='Priority for Conv O/P wrt Bicubic Interpolation')
parser.add_argument('--deploy',
    dest='deploy',
    action='store_true',
    help='export xmodel for deployment')
parser.add_argument('--fast_finetune',
    dest='fast_finetune',
    action='store_true',
    help='fast finetune model before calibration')
parser.add_argument('--bit_width',default=8,type=int,
        help='Quantization Bit Width')

args,_=parser.parse_known_args()

#concatenate pixelshuffle
class ModelwPS(nn.Module):
    def __init__(self,model,upscale=2):
        super(ModelwPS,self).__init__()
        self.model=model
        self.PS=nn.PixelShuffle(upscale)

    def forward(self,x):
        x=self.model(x)
        x=self.PS(x)
        return(x)


#data loader train and val
def load_data(dirc,input_transform,target_transform,fetch="train",
             batch_size=1,shuffle=True,num_workers=0):
    dataset=SRDataset(root_dir=dirc, input_transform=input_transform((2160,3840),2), #TODO: Dynamic size setting
                    target_transform=target_transform(),fetch=fetch)
    loader=data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,
                         num_workers=num_workers)
    print("---------------------- Dataset Loaded ----------------------")
    return loader,dataset

#psnr and ssim
def psnr(SR,HR):
  diff=np.subtract(HR,SR)
  mse=np.mean(np.power(diff,2))
  snr= -10*math.log10(mse)
  return snr

def validate_batch(LR,HR,model,criterion,ssim_mc):
    alpha=args.alpha
    tensor=transforms.ToTensor()
    pil=transforms.ToPILImage()
    scale=transforms.Resize((2160,3840),PIL.Image.BICUBIC) #TODO: dynamic resolution
    model.eval()
    with torch.no_grad():
        LR,HR=LR.float(),HR.float()
        gLR,gHR=LR.to(device),HR.to(device)
        gSR=model(gLR)
        loss=criterion(gHR,gSR)
        SR=gSR.to('cpu')
        SR,HR,LR=SR.squeeze(1),HR.squeeze(1),LR.squeeze(1)
        out_SR_y,img_HR_y=SR.detach().numpy(),HR.detach().numpy()
        del gSR
        del gLR
        del gHR
        #out_img_y,out_HR_y=np.asarray(SR),np.asarray(HR)
        pil=transforms.ToPILImage()
        LR=pil(LR)
        out_HR_y=scale(LR)
        out_HR_y=tensor(out_HR_y)
        out_HR_y=np.asarray(out_HR_y)
        out_y= out_SR_y*alpha + out_HR_y*(1-alpha)
        snr=psnr(out_y,img_HR_y)
        out_y,img_HR_y=out_y.transpose(1,2,0),img_HR_y.transpose(1,2,0)
        stsim=ssim(out_y,img_HR_y,multichannel=ssim_mc)
        del LR
        del HR
        del out_SR_y
        del out_HR_y
        del img_HR_y
        del out_y
        del SR
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return loss,snr,stsim


#evaluation - forward and report psnr ssim
def evaluate(model,val_loader,criterion,ssim_mc):
    modelps=ModelwPS(model)
    print("---------------------- Eval Started ----------------------")
    loss=0
    total=0
    apsnr=0
    assim=0
    for i,(sample,_,_,_) in enumerate(val_loader):
        v_loss,psnr,ssim=validate_batch(sample['LR'],sample['HR'],modelps,criterion,ssim_mc)
        apsnr+=psnr
        assim+=ssim
        loss+=v_loss.item()
        total+=len(sample['LR'])
        print("Data:{0} Test Complete".format(i))
    print('.')
    return apsnr/total,assim/total,loss/total


#trace_test
def trace(model):
    print("---------------------- Trace Started ----------------------")
    #try trace
    model.eval()
    inp =torch.rand(1,1,1080,1920)
    inp=inp.to(device)
    op=torch.jit.trace(model,inp)
    if isinstance(op,torch.jit.ScriptModule):
        print("---------------------- End of trace ----------------------")
        torch.cuda.empty_cache()
        return True
    else:
        print("---------------------- Trace failed ----------------------")
        return False


#quantization - quantise and evaluate
def quantization(model,dirc,quant_mode=1,val=True):
    batch_size=args.batch_size
    ssim_mc=True
    deploy=args.deploy
    bitwidth=args.bit_width
    finetune=args.fast_finetune

    if quant_mode != 'test' and deploy:
        deploy = False
        print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
    if deploy and (batch_size != 1):
        print(r'Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!')
        batch_size = 1
        

        
    val_loader,_=load_data(dirc,input_transform,target_transform,fetch='deploy',batch_size=batch_size)
    inp= torch.randn([batch_size,1,1080,1920])
    if quant_mode == "float":
        quant_model=model
    else:
        quantizer=torch_quantizer(quant_mode,model,(inp),bitwidth=bitwidth,device=device)
        quant_model=quantizer.quant_model
        quant_model=quant_model.to(device)

    #print(quant_mode,':',type(quant_model))
    criterion=nn.L1Loss()
    
    if finetune == True:
        if quant_mode == 'calib':
            quantizer.fast_finetune(evaluate, (quant_model.to(device), val_loader, criterion,ssim_mc))
        elif quant_mode == 'test':
            quantizer.load_ft_param()
    
    if val:
        avg_psnr,avg_ssim,loss=evaluate(quant_model,val_loader,criterion,ssim_mc)
        print("Loss for Validation set:",loss)
        print("Avg_PSNR for Model is:",avg_psnr)
        print("Avg_SSIM for Model is:",avg_ssim)
        print("---------------------- Eval Complete ----------------------")
    
    if quant_mode == 'calib':
        quantizer.export_quant_config()
    if quant_mode == 'test' and deploy:
        quantizer.export_xmodel(deploy_check=True)
        #dump_xmodel()
        print("---------------------- Model Quantized ----------------------")


#main
if __name__ == '__main__':
    model_name=args.model_name
    ckpt=args.ckpt
    file_path=os.path.join(args.model_dir,ckpt+'.pt')   
    feature_test=' '
    if args.quant_mode != 'float':
        feature_test=' quantization'
        args.optimize=1
        feature_test+=' with optimization'
    else:
        feature_test=' float model evaluation'
   
    print("Model dir:",args.model_dir)
    print("Data dir:",args.data_dir)
    print("Quant_mode:",args.quant_mode)
    print("Batch_size:",args.batch_size)
    print("Model Name:",args.model_name)
    print("Checkpoint name:",args.ckpt)
    print("alpha:",args.alpha)

    title = model_name + feature_test
    torch.set_grad_enabled(False)



    if model_name == 'SRHW':
        model=SRHW(quant=True).to(device)
        model.load_state_dict(torch.load(file_path))
    else:
        model=None

    if not model== None:
        print("---------------------- Model Loaded  ----------------------")
        

        #perform jit trace test
        res=trace(model=model)

        #quantise
        if res:
            print("---------------------- Start {} Quantization ----------------------"
                    .format(model_name))
            quantization(model=model,
                    dirc=args.data_dir,quant_mode=args.quant_mode,val=True)

            print("---------------------- End of {} Quantization ----------------------"
                    .format(model_name))
            
            #TODO: Ask for image save and save a random image for each alpha and quant mode when evaluation is complete

