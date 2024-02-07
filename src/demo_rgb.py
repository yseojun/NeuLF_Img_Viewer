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

import torch
import numpy as np
import argparse
import os
import glob
import sys
import time
sys.path.insert(0, os.path.abspath('./'))
from src.model import Nerf4D_relu_ps
from utils import rm_folder, rm_folder_keep
import torchvision

parser = argparse.ArgumentParser() # museum,column2
parser.add_argument('--exp_name',type=str, default = 'Ollie_d8_w256',help = 'exp name')
parser.add_argument('--gpuid',type=str, default = '0',help='data folder name')
parser.add_argument('--mlp_depth', type=int, default = 8)
parser.add_argument('--mlp_width', type=int, default = 256)
parser.add_argument('--scale', type=int, default = 4)
parser.add_argument('--img_form',type=str, default = '.png',help = 'exp name')

class demo_rgb():
    def __init__(self,args):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
        print('>>> Using GPU: {}'.format(args.gpuid))

        # data_root
        self.model = Nerf4D_relu_ps(D=args.mlp_depth,W=args.mlp_width,depth_branch=False)
       

        self.exp = 'Exp_'+args.exp_name
        self.checkpoints = 'result/'+self.exp+'/checkpoints/'

        self.img_folder_test = 'demo_result_rgb/'+self.exp+'/'
        rm_folder_keep(self.img_folder_test)


        path = self.img_folder_test+'test.png'
        print(path)
        self.load_check_points()
        self.model = self.model.cuda()

        vec3_xyz = self.get_vec3_xyz(0,0,0)

        self.get_image(vec3_xyz,path)

        
    
    

        
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
        print(f"Load `` from {ckpt_name}")
        
        ckpt = torch.load(ckpt_name)
    
        self.model.load_state_dict(ckpt)\
        

    def get_vec3_xyz(self,x,y,z,H=720,W=720):
        print("H new")

        aspect = W/H


        theta = np.pi  # 180 degrees in radians
        #theta = 0
        rot_mat = np.array([
            [-1,  0,           0],
            [ 0,  np.cos(theta), -np.sin(theta)],
            [ 0,  np.sin(theta),  np.cos(theta)]
        ])


        rot_x = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])

        # y축 회전 행렬
        rot_y = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        # z축 회전 행렬
        rot_z = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

        


        u = np.linspace(-1, 1, W, dtype='float32')
        v = np.linspace(1, -1, H, dtype='float32') / aspect

        vu = list(np.meshgrid(u, v))


        u = vu[0]
        v = vu[1]
        dirs = np.stack((u, v, -np.ones_like(u)), axis=-1)
        dirs = np.sum(dirs[..., np.newaxis, :]* rot_z[:3,:3],-1)
        dirs = np.reshape(dirs,(-1,3))

        tx = np.ones_like(dirs[:, 0:1]) *x
        ty = np.ones_like(dirs[:, 0:1]) *y
        tz = np.ones_like(dirs[:, 0:1]) *z

        vec3_xyz = np.concatenate((dirs, tx, ty, tz), axis=1)
        vec3_xyz = np.reshape(vec3_xyz,(-1,6))

        return vec3_xyz


    def get_image(self,vec3_xyz,save_path):
        print('test1103')
        vec3_xyz = torch.from_numpy(vec3_xyz.astype(np.float32)).cuda()
        
        start_time =time.time()           

        pred_color = self.model(vec3_xyz)
        end_time =time.time()
           
        elapsed_time = end_time - start_time
        print(f"inference time : {elapsed_time} sec")

        pred_img = pred_color.reshape((720,720,3)).permute((2,0,1))

        torchvision.utils.save_image(pred_img, save_path)
    


        

        


        

if __name__ == '__main__':

    args = parser.parse_args()

    unit = demo_rgb(args)


    print("finish")
 
