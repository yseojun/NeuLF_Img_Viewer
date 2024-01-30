# Copyright (C) 2023 OPPO. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import os
import logging
from datetime import datetime
import time

sys.path.insert(0, os.path.abspath('./'))


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
import argparse
import cv2
from torch.utils.tensorboard import SummaryWriter



from src.model import Nerf4D_relu_ps
import torchvision
import glob
import torch.optim as optim
from utils import rm_folder,AverageMeter,rm_folder_keep



import imageio
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import lpips
import psutil

def calculate_angle1(x, y):
    angle = np.arctan2(x, y)
    angle = np.where(angle < 0, angle + 2*np.pi, angle)
    return (((2*np.pi - angle) - np.pi)*2)/np.pi
def calculate_angle2(x, y):
    angle = np.arctan2(x, y)
    angle = np.where(angle < 0, angle + 2*np.pi, angle)
    return ((angle - np.pi)*2)/np.pi


def direction_to_euler(vec):
    x, y, z = vec[:, 0], vec[:, 1], vec[:, 2]
    theta = calculate_angle1(y,z)
    phi = calculate_angle1(x,z)

    return theta, phi


        

def get_rotaion_matirx(theta):

    rotation_matrix = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
    ])
    return rotation_matrix.T


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name',type=str, default = 'Ollie_d8_w256',help = 'exp name')
parser.add_argument('--data_dir',type=str, 
                    default = 'dataset/Ollie/',help='data folder name')
parser.add_argument('--batch_size',type=int, default = 8192,help='normalize input')
parser.add_argument('--test_freq',type=int,default=10,help='test frequency')
parser.add_argument('--save_checkpoints',type=int,default=10,help='checkpoint frequency')
parser.add_argument('--whole_epoch',type=int,default=50,help='checkpoint frequency')
parser.add_argument('--gpuid',type=str, default = '0',help='data folder name')
parser.add_argument("--factor", type=int, default=1, help='downsample factor for LLFF images')
parser.add_argument('--img_scale',type=int, default= 1, help= "devide the image by factor of image scale")
# parser.add_argument('--norm_fac',type=float, default=1, help= "normalize the data uvst")
# parser.add_argument('--st_norm_fac',type=float, default=1, help= "normalize the data uvst")
parser.add_argument('--work_num',type=int, default= 15, help= "normalize the data uvst")
parser.add_argument('--lr_pluser',type=int, default = 100,help = 'scale for dir')
parser.add_argument('--lr',type=float,default=5e-04,help='learning rate')
parser.add_argument('--loadcheckpoints', action='store_true', default = False)
parser.add_argument('--st_depth',type=int, default= 0, help= "st depth")
parser.add_argument('--uv_depth',type=int, default= 0.0, help= "uv depth")
parser.add_argument('--rep',type=int, default=1)
parser.add_argument('--mlp_depth', type=int, default = 4)
parser.add_argument('--mlp_width', type=int, default = 64)
#imlab

parser.add_argument('--renderpose_only', action='store_true' ,default = False)
parser.add_argument('--timecheck', action='store_true' ,default = False)
parser.add_argument('--test_render', action='store_true' ,default = False)


class train():
    def __init__(self,args):
        print("1102 333")
        process = psutil.Process(os.getpid())
         # set gpu id
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
        print('>>> Using GPU: {}'.format(args.gpuid))
        print('start : ',process.memory_info().rss / (1024 ** 2), "MB")

       
        self.timecheck = True


        # data root
        data_root = args.data_dir
        data_img = os.path.join(args.data_dir,'images_{}'.format(args.factor)) 

        self.model = Nerf4D_relu_ps(D=args.mlp_depth,W=args.mlp_width,input_ch=args.mlp_width)
        for name, module in self.model.named_children():
                print(name, module)
       
        

        self.exp = 'Exp_'+args.exp_name

        # tensorboard writer
        self.summary_writer = SummaryWriter("src/tensorboard/"+self.exp)

        # save img
        self.save_img = True
        self.img_folder_train = 'result/'+self.exp+'/train/'
        self.img_folder_test = 'result/'+self.exp+'/test/'
        self.checkpoints = 'result/'+self.exp+'/checkpoints/'
        self.img_check_time = 'result/' + self.exp + '/timecheck'

        # make folder
        rm_folder_keep(self.img_folder_train)
        rm_folder_keep(self.img_folder_test)
        rm_folder_keep(self.checkpoints)

        handlers = [logging.StreamHandler()]
        dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M")
        handlers.append(logging.FileHandler('result/'+self.exp+f'/{dt_string}.log', mode='w'))
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)-5s %(message)s',
            datefmt='%m-%d %H:%M:%S', handlers=handlers,
        )

        # load checkpoints
        if(args.loadcheckpoints):
            self.load_check_points()

        self.model = self.model.cuda()

        # height and width
        image_paths = glob.glob(f"{data_img}/*.png")
        sample_img = cv2.imread(image_paths[0])
        self.h = int(sample_img.shape[0]/args.img_scale)
        self.w = int(sample_img.shape[1]/args.img_scale)

        self.img_scale = args.img_scale

        self.step_count = 0

        # load nelf data
        print(f"Start loading...")
        split='train'
        # center
        # center
        self.uvst_whole   = np.load(f"{data_root}/uvst{split}.npy")
        self.uvst_whole[:,3] = (self.uvst_whole[:,3] - self.uvst_whole[:,3].min())/(self.uvst_whole[:,3].max() - self.uvst_whole[:,3].min())
        self.uvst_whole[:,5] = (self.uvst_whole[:,5] - self.uvst_whole[:,5].min())/(self.uvst_whole[:,5].max() - self.uvst_whole[:,5].min())
        x_min = self.uvst_whole[:,3].min()
        x_max = self.uvst_whole[:,3].max()
        z_min = self.uvst_whole[:,5].min()
        z_max = self.uvst_whole[:,5].max()
        print("Stop loading...")
        self.uvst_whole_len = self.uvst_whole.shape[0]

        self.uvst_whole_gpu    = torch.tensor(self.uvst_whole).float()
        self.uvst_whole = None
      
        self.color_whole   = np.load(f"{data_root}/rgb{split}.npy")
        self.color_whole_gpu      = torch.tensor(self.color_whole).float()
        self.color_whole = None

     
        self.start, self.end = [], []
        s = 0
        while s < self.uvst_whole_len:
            self.start.append(s)
            s += args.batch_size
            self.end.append(min(s, self.uvst_whole_len))
       
        split='val'
        self.uvst_whole_val   = np.load(f"{data_root}/uvst{split}.npy")
        self.uvst_whole_val[:,3] = (self.uvst_whole_val[:,3] - x_min)/(x_max - x_min)
        self.uvst_whole_val[:,5] = (self.uvst_whole_val[:,5] - z_min)/(z_max - z_min)

        
        self.color_whole_val   = np.load(f"{data_root}/rgb{split}.npy")
        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(0.9, 0.999))
       
        self.vis_step = 1
        
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995) 

        self.epoch_num = args.whole_epoch
    
        # mpips eval
        self.lpips = lpips.LPIPS(net='vgg') 
        

    def train_step(self,args):
        for epoch in range(0,self.epoch_num):

            if(args.loadcheckpoints):
                epoch += self.iter
            if(args.timecheck):

                self.gen_line_gif_llff_path("roty=0_round",0) 
                self.gen_line_gif_llff_path("roty=180_round",np.pi) 
                # self.gen_line_gif_llff_path("roty=90_round",np.pi/2) 
                # self.gen_line_gif_llff_path("roty=-90_round",-np.pi/2)
                # self.gen_line_gif_llff_path("roty=45_round",np.pi/4)
                # self.gen_line_gif_llff_path("roty=-45_round",-np.pi/4)
                # self.gen_line_gif_llff_path("roty=135_round",np.pi/2 + np.pi/4)
                # self.gen_line_gif_llff_path("roty=-135_round",-np.pi/2 - np.pi/4)
                sys.exit()
                save_dir = self.img_check_time
                start_time =time.time()
                vec3_xz =  self.get_uvst(0,0,0)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"make vec3xz time : {elapsed_time} sec")
                
                rm_folder_keep(save_dir)
                self.render_sample_img(self.model, vec3_xz, self.w, self.h, f"{save_dir}/timecheck.png")
                print("finish")
                sys.exit()


            # if(args.test_render):
            #     self.test(epoch)
            #     print("test_render finish")
            #     sys.exit()

            
            if args.renderpose_only:
                
                save_dir_renderpose = self.img_folder_train + f"renderpose-{epoch}"
                rm_folder_keep(save_dir_renderpose)
                for i in range(12):
                    if i<2:
                        n = 600
                    else:
                        n = 240
                    self.gen_rotating_gif_llff_path(f"{save_dir_renderpose}/rotated_video_ani_llff{i}.gif",n,i,1,2)
                self.gen_rotating_gif_llff_path(f"{save_dir_renderpose}/rotated_video_ani_llff_rotnum.gif",n,0,0.7,1.7)
                #self.gen_gif_llff_path(f"{save_dir_renderpose}/video_ani_llff.gif")
                sys.exit()
            
            self.losses = AverageMeter()
            self.losses_rgb = AverageMeter()
            self.losses_rgb_super = AverageMeter()
            self.losses_depth = AverageMeter()

            self.model.train()
            self.step_count +=1

            perm = torch.randperm(self.uvst_whole_len)
            self.uvst_whole_gpu = self.uvst_whole_gpu[perm]
            self.color_whole_gpu = self.color_whole_gpu[perm]
            
            
            self.train_loader = [{'input': self.uvst_whole_gpu[s:e], 
                                    'color': self.color_whole_gpu[s:e]} for s, e in zip(self.start, self.end)]

            pbar = self.train_loader
            for i, data_batch in enumerate(pbar):

                self.optimizer.zero_grad()
                inputs  = data_batch["input"].cuda()
                color = data_batch["color"].cuda()

                preds_color = self.model(inputs.cuda())
                
                loss_rgb = 1000*torch.mean((preds_color - color) * (preds_color - color))

                save_dir = self.img_folder_train + f"epoch-{epoch}"

                loss = loss_rgb 

                self.losses.update(loss.item(), inputs.size(0))
                self.losses_rgb.update(loss_rgb.item(),inputs.size(0))
            
                loss.backward()
               
                self.optimizer.step()
                log_str = 'epoch {}/{}, {}/{}, lr:{}, loss:{:4f}'.format(
                    epoch,self.epoch_num,i+1,len(self.start),
                    self.optimizer.param_groups[0]['lr'],self.losses.avg,)
           
                if (i+1) % 2000 == 0:
                    logging.info(log_str)
            logging.info(log_str)
            self.scheduler.step()
            
            with torch.no_grad():
                self.model.eval()
                if epoch % args.test_freq ==0:
                    
                    
                    save_dir = self.img_folder_train + f"epoch-{epoch}"
                    rm_folder_keep(save_dir)
                   

                    self.gen_rotating_gif_llff_path(f"{save_dir}/video_ani_llff0.gif")
                    self.gen_rotating_gif_llff_path(f"{save_dir}/video_ani_llff3.gif",mode = 3)
                    self.gen_rotating_gif_llff_path(f"{save_dir}/video_ani_llff4.gif",mode = 4)
                    self.gen_rotating_gif_llff_path(f"{save_dir}/video_ani_llff5.gif",mode = 5)
                    self.gen_rotating_gif_llff_path(f"{save_dir}/video_ani_llff6.gif",mode = 6)
                    # if(epoch != 0):
                    #     self.val(epoch)

                if epoch % args.save_checkpoints == 0:
                    cpt_path = self.checkpoints + f"nelf-{epoch}.pth"
                    torch.save(self.model.state_dict(), cpt_path)

    

    def get_uvst2(self,cam_x,cam_y,cam_z,rotmaty, rotation_matric = None ,center_flag=False ):

        theta = np.pi
        theta_y = rotmaty
        rot_z = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

        rot_y = np.array([
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)]
        ])
        rot = rot_z @ rot_y
        W = self.w
        H = self.h

        uvst_tmp = []
        aspect = W/H
       
        u = np.linspace(-1, 1, W, dtype='float32')
        v = np.linspace(1, -1, H, dtype='float32') / aspect
        vu = list(np.meshgrid(u, v))

        u = vu[0]
        v = vu[1] 
        dirs = np.stack((u, v, -np.ones_like(u)), axis=-1)
        dirs = np.sum(dirs[..., np.newaxis, :]* rot,-1)
        dirs = np.array(dirs)
        dirs = np.reshape(dirs,(-1,3))
        
        x = np.ones_like(vu[0]) * cam_x
        y = np.ones_like(vu[0]) * cam_y
        z = np.ones_like(vu[0]) * cam_z
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        x = np.reshape(x,(-1,1))
        y = np.reshape(y,(-1,1))
        z = np.reshape(z,(-1,1))

    
        concatenated_array = np.concatenate((dirs, x, y, z), axis=1)
        uvst_tmp.append(concatenated_array)

        #uvst_tmp = np.concatenate([uvst_tmp, concatenated_array])
        uvst_tmp = concatenated_array


        uvst_tmp = np.asarray(uvst_tmp)
        uvst_tmp = np.reshape(uvst_tmp,(-1,6))
        data_uvst = uvst_tmp

        return data_uvst                

    def val(self,epoch):
        with torch.no_grad():
            i=0
            p = []
            s = []
            l = []

            save_dir = self.img_folder_test + f"epoch-{epoch}"
            rm_folder(save_dir)
            
            count = 0
            while i < self.uvst_whole_val.shape[0]:
                end = i+self.w*self.h
                uvst = self.uvst_whole_val[i:end]
                uvst = torch.from_numpy(uvst.astype(np.float32)).cuda()

                pred_color = self.model(uvst)
                gt_color = self.color_whole_val[i:end]

                # write to file
                pred_img = pred_color.reshape((self.h,self.w,3)).permute((2,0,1))
                gt_img   = torch.tensor(gt_color).reshape((self.h,self.w,3)).permute((2,0,1))
                

                torchvision.utils.save_image(pred_img,f"{save_dir}/test_{count}.png")
                torchvision.utils.save_image(gt_img,f"{save_dir}/gt_{count}.png")

                pred_color = pred_color.cpu().numpy()
                
                psnr = peak_signal_noise_ratio(gt_color, pred_color, data_range=1)
                ssim = structural_similarity(gt_color.reshape((self.h,self.w,3)), pred_color.reshape((self.h,self.w,3)), data_range=pred_color.max() - pred_color.min(),multichannel=True)
                lsp  = self.lpips(pred_img.cpu(),gt_img) 

                p.append(psnr)
                s.append(ssim)
                l.append(np.asscalar(lsp.numpy()))

                i = end
                count+=1

            logging.info(f'>>> val: psnr  mean {np.asarray(p).mean()} full {p}')
            logging.info(f'>>> val: ssim  mean {np.asarray(s).mean()} full {s}')
            logging.info(f'>>> val: lpips mean {np.asarray(l).mean()} full {l}')
            return p

    def render_sample_img(self,model,uvst, w, h, save_path=None,save_depth_path=None,save_flag=True):
        with torch.no_grad():
           
            uvst = torch.from_numpy(uvst.astype(np.float32)).cuda()
            
            if self.timecheck:
                start_time =time.time()

            pred_color = model(uvst)

            if self.timecheck:
                end_time =time.time()
           

            if self.timecheck:
                elapsed_time = end_time - start_time
                print(f"inference time : {elapsed_time} sec")
                self.timecheck = False

            pred_img = pred_color.reshape((h,w,3)).permute((2,0,1))

            if(save_flag):
                torchvision.utils.save_image(pred_img, save_path)
           
            return pred_color.reshape((h,w,3))

    def train_summaries(self):
          
        self.summary_writer.add_scalar('total loss', self.losses.avg, self.step_count)

     

    def load_check_points(self):
        ckpt_paths = glob.glob(self.checkpoints+"*.pth")
        self.iter=0
        if len(ckpt_paths) > 0:
            for ckpt_path in ckpt_paths:
                print(ckpt_path)
                ckpt_id = int(os.path.basename(ckpt_path).split(".")[0].split("-")[1])
                self.iter = max(self.iter, ckpt_id)
            ckpt_name = f"./{self.checkpoints}/nelf-{self.iter}.pth"
        # ckpt_name = f"{self.checkpoints}nelf-{self.fourier_epoch}.pth"
        print(f"Load weights from {ckpt_name}")
        
        ckpt = torch.load(ckpt_name)
        try:
            self.model.load_state_dict(ckpt)
        except:
            tmp = DataParallel(self.model)
            tmp.load_state_dict(ckpt)
            self.model.load_state_dict(tmp.module.state_dict())
            del tmp

    


    def gen_rotating_gif_llff_path(self,savename , n=360,mode = 0,rotnum = 1 ,x_num = 2):
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out   = cv2.VideoWriter(savename+".mp4", fourcc, 24.0, (self.w,self.h))
        view_group = []

        repeats = np.linspace(0, 2*np.pi, n)
        for r in repeats:
            if(mode == 0):
                data_uvst = self.get_uvst2((0.5+0.5*np.sin(x_num*r)) , 0,(0.5-0.5*np.sin(r)), rotnum*r)
            if(mode == 1):
                data_uvst = self.get_uvst2((0.5+0.5*np.sin(x_num*r)) , 0,(0.5-0.5*np.sin(r)),0)
            if(mode == 2):
                data_uvst = self.get_uvst2((0.5+0.5*np.sin(r)) , 0,0.5, rotnum*r)
            if(mode == 3):
                data_uvst = self.get_uvst2((0.5+0.5*np.sin(r)) , 0,0.5, -np.pi)
            if(mode == 4):
                data_uvst = self.get_uvst2((0.5+0.5*np.sin(r)) , 0,0.5, np.pi)
            if(mode == 5):
                data_uvst = self.get_uvst2((0.5+0.5*np.sin(r)) , 0,0.5, -np.pi/2)
            if(mode == 6):
                data_uvst = self.get_uvst2((0.5+0.5*np.sin(r)) , 0,0.5, np.pi/2)
            if(mode == 7):
                data_uvst = self.get_uvst2(0,0.5, 0,(0.5-0.5*np.sin(r)), rotnum*r)
            if(mode == 8):
                data_uvst = self.get_uvst2(0,0.5, 0,(0.5-0.5*np.sin(r)), -np.pi)
            if(mode == 9):
                data_uvst = self.get_uvst2(0,0.5, 0,(0.5-0.5*np.sin(r)), np.pi)
            if(mode == 10):
                data_uvst = self.get_uvst2(0,0.5, 0,(0.5-0.5*np.sin(r)), -np.pi/2)
            if(mode == 11):
                data_uvst = self.get_uvst2(0,0.5, 0,(0.5-0.5*np.sin(r)), np.pi/2)

            
            view_unit = self.render_sample_img(self.model,data_uvst,self.w,self.h,None,None,False)

            view_unit *= 255

            view_unit       = view_unit.cpu().numpy().astype(np.uint8)

            out.write(cv2.cvtColor(view_unit,cv2.COLOR_RGB2BGR))

            view_unit       = imageio.core.util.Array(view_unit)
            view_group.append(view_unit)

        #imageio.mimsave(savename, view_group,fps=30)
        out.release()    

    def gen_line_gif_llff_path(self,savename,rotmaty = np.pi):
        
        path = self.img_folder_train + savename
        rm_folder_keep(path)

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out   = cv2.VideoWriter(path+"/gen_line"+".mp4", fourcc, 24.0, (self.w,self.h))
        
        view_group = []
        i = 0
       
        cam_zp = np.linspace(0.001, 0.1, 100)
        logging.info(cam_zp) 
    

        path_img = path + "/imgs"
        rm_folder_keep(path_img)
        for z in cam_zp:
            
            i +=1
            save_dir = path_img + f"/{i}.png"

            data_uvst = self.get_uvst2(0.05,0,z,rotmaty)
            
            view_unit = self.render_sample_img(self.model,data_uvst,self.w,self.h,save_dir,None,True)

            view_unit *= 255

            view_unit       = view_unit.cpu().numpy().astype(np.uint8)

            out.write(cv2.cvtColor(view_unit,cv2.COLOR_RGB2BGR))

            view_unit       = imageio.core.util.Array(view_unit)
            view_group.append(view_unit)

        #imageio.mimsave(savename, view_group,fps=30)
        out.release()    
    
    

    
    

if __name__ == '__main__':

    args = parser.parse_args()

    m_train = train(args)

    m_train.train_step(args)
