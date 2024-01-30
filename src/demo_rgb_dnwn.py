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
import cv2
import imageio
import logging
from datetime import datetime

parser = argparse.ArgumentParser() # museum,column2
parser.add_argument('--exp_name',type=str, default = 'Ollie_d8_w256',help = 'exp name')
parser.add_argument('--gpuid',type=str, default = '0',help='data folder name')
parser.add_argument('--mlp_depth', type=int, default = 8)
parser.add_argument('--scale', type=int, default = 4)
parser.add_argument('--img_form',type=str, default = '.png',help = 'exp name')

class demo_rgb():
    def __init__(self,args,n=2):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
        print('>>> Using GPU: {}'.format(args.gpuid))
        #pth_name = ['1to6.pth','5to10.pth']
        pth_name = ['d4w64-100.pth','5to10.pth']
        #logging
        handlers = [logging.StreamHandler()]
        dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M")
        handlers.append(logging.FileHandler(f'demo_result_rgb/testd4w64_3.log', mode='w'))

        # 로깅 기본 설정: 로그 레벨, 포맷, 날짜 포맷 및 핸들러들 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)-5s %(message)s',
            datefmt='%m-%d %H:%M:%S',
            handlers=handlers,
        )


        # data_root
        self.h = 720
        self.w = 720

        num = 4
        self.model = []
        for i in range(num):
            model =Nerf4D_relu_ps(D=4,W=64,input_ch=64,depth_branch=False)
            ckpt = torch.load("pth_dir/" + pth_name[0])
            model.load_state_dict(ckpt)
            self.model.append(model)

        for i in range(num):
            self.model[i] = self.model[i].cuda()

       
        self.img_folder_test = 'demo_result_rgb/'
        rm_folder_keep(self.img_folder_test)


        path = self.img_folder_test+'test.png'
        print(path)

        test = self.get_divided_uvst(0.5,0,0.5,0)

        print(test.shape)
        n = 60

        z = np.random.rand(n)
        x = np.random.rand(n)

        # 0부터 2*np.pi 사이의 랜덤한 값 60개 생성
        roty = np.random.uniform(0, 2*np.pi, n)

        for i in range(n):
            
            start_func = torch.cuda.Event(enable_timing=True)
            end_func = torch.cuda.Event(enable_timing=True)

            start_func.record()
            #self.render_img_by_multiStream_timecheck_with_porchEvent(x[i],0,z[i],roty[i],self.w,self.h)
            self.render_img_timecheck_with_porchEvent(x[i],0,z[i],roty[i],self.w,self.h)
            end_func.record()
            torch.cuda.synchronize()

            
            logging.info( f'func_time {i} : {start_func.elapsed_time(end_func)}')
            torch.cuda.synchronize()
            






###Render
        # for i in range(len(pth_name)):
        #     path = self.img_folder_test+pth_name[i] + '.png'
        #     vec3_xyz = self.get_vec3_xyz(0,0,0)
        #     self.get_image(self.model[i],vec3_xyz,path)

        # test_path = self.img_folder_test+"test"
        # print("ttttttt")    
        # #self.gen_rotating_gif_llff_path(test_path,360,13)
        # #sys.exit()
        # for i in range(16):
        #     test_path = self.img_folder_test+"gen_rotating_" +str(i)
        #     self.gen_rotating_gif_llff_path(test_path,360,i)


        
        



        
    
    def load_pths(self):
        ckpt_paths = glob.glob("pth_dir/"+"*.pth")
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


    def get_image(self, model,vec3_xyz,save_path):
        print('test1103')
        vec3_xyz = torch.from_numpy(vec3_xyz.astype(np.float32)).cuda()
        
        start_time =time.time()           

        pred_color = model(vec3_xyz)
        end_time =time.time()
           
        elapsed_time = end_time - start_time
        print(f"inference time : {elapsed_time} sec")

        pred_img = pred_color.reshape((720,720,3)).permute((2,0,1))

        torchvision.utils.save_image(pred_img, save_path)


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
    


    def get_divided_uvst(self, cam_x, cam_y, cam_z, rotmaty):
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
        W, H = self.w // 2, self.h // 2  # 분면의 크기

        uvst_quadrants = []

        for i in range(2):
            for j in range(2):
                u_start = -1 + j 
                u_end = 0 + j
                v_start = -1 + i
                v_end = 0 + i
                u = np.linspace(u_start, u_end, W, dtype='float32')
                v = np.linspace(v_start, v_end, H, dtype='float32') / (W/H)
                vu = list(np.meshgrid(u, v))

                dirs = np.stack((vu[0], vu[1], -np.ones_like(vu[0])), axis=-1)
                dirs = np.sum(dirs[..., np.newaxis, :] * rot, -1)
                dirs = dirs.reshape(-1, 3)

                x = np.ones_like(vu[0]) * cam_x
                y = np.ones_like(vu[0]) * cam_y
                z = np.ones_like(vu[0]) * cam_z

                concatenated_array = np.concatenate((dirs, x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), axis=1)
                uvst_quadrants.append(concatenated_array)

        return np.array(uvst_quadrants)
    
    def render_blended_img(self,tx,ty,tz,roty, w, h, save_path=None,save_depth_path=None,save_flag=True):
        with torch.no_grad():
            #print(tz)
            
            if(tz < 0.8):
                uvst = self.get_uvst2(tx,ty,tz,roty)
                uvst = torch.from_numpy(uvst.astype(np.float32)).cuda()
                pred_color = self.model[0](uvst)
                pred_img = pred_color.reshape((h,w,3)).permute((2,0,1))
            elif(0.8<=tz and tz <=1.0):
                uvst = self.get_uvst2(tx,ty,tz,roty)
                uvst = torch.from_numpy(uvst.astype(np.float32)).cuda()
                pred_color1 = self.model[0](uvst)
                alpha = 1 - (tz-0.8)/0.2
                pred_img1 = pred_color1.reshape((h,w,3)).permute((2,0,1))
                
                uvst = self.get_uvst2(tx,ty,tz-0.8,roty)
                uvst = torch.from_numpy(uvst.astype(np.float32)).cuda()
                pred_color2 = self.model[1](uvst)
                pred_img2 = pred_color2.reshape((h,w,3)).permute((2,0,1))
                pred_color = alpha*pred_color1 + (1-alpha)*pred_color2
            else:
                uvst = self.get_uvst2(tx,ty,tz-0.8,roty)
                uvst = torch.from_numpy(uvst.astype(np.float32)).cuda()
                pred_color = self.model[1](uvst)
                pred_img = pred_color.reshape((h,w,3)).permute((2,0,1))


            if(save_flag):
                torchvision.utils.save_image(pred_img, save_path)
           
            return pred_color.reshape((h,w,3))
        
    def render_img_timecheck_with_porchEvent(self,tx,ty,tz,roty, w, h, save_path=None,save_depth_path=None,save_flag=False):
        with torch.no_grad():
            #print(tz)
            

            start_uvst = torch.cuda.Event(enable_timing=True)
            end_uvst = torch.cuda.Event(enable_timing=True)

            start_uvst.record()
            uvst = self.get_uvst2(tx,ty,tz,roty)
            end_uvst.record()
            torch.cuda.synchronize()

            logging.info(f'x:{tx} , y: {ty} , z:{tz} , uvst_time : {start_uvst.elapsed_time(end_uvst)}')

            torch.cuda.synchronize()
            
            start_cuda_copy = torch.cuda.Event(enable_timing=True)
            end_cuda_copy = torch.cuda.Event(enable_timing=True)

            start_cuda_copy.record()
            uvst = torch.from_numpy(uvst.astype(np.float32)).cuda()
            end_cuda_copy.record()
            
            torch.cuda.synchronize()
            logging.info(f'x:{tx} , y: {ty} , z:{tz} , cuda_copy_time : {start_cuda_copy.elapsed_time(end_cuda_copy)}')
            torch.cuda.synchronize()

            start_infer = torch.cuda.Event(enable_timing=True)
            end_infer = torch.cuda.Event(enable_timing=True)

            start_infer.record()
            pred_color = self.model[0](uvst)
            end_infer.record()


            torch.cuda.synchronize()
            logging.info(f'x:{tx} , y: {ty} , z:{tz} , infer_time : {start_infer.elapsed_time(end_infer)}')
            torch.cuda.synchronize()

            pred_img = pred_color.reshape((h,w,3)).permute((2,0,1))
            
            
            

            if(save_flag):
                torchvision.utils.save_image(pred_img, save_path)
           
            return pred_color.reshape((h,w,3))


    def render_img_by_multiStream_timecheck_with_porchEvent(self,tx,ty,tz,roty, w, h, save_path=None,save_depth_path=None,save_flag=False):
        with torch.no_grad():
            #print(tz)
            
                    


            start_uvst = torch.cuda.Event(enable_timing=True)
            end_uvst = torch.cuda.Event(enable_timing=True)

            start_uvst.record()
            uvst = self.get_divided_uvst(tx,ty,tz,roty)
            end_uvst.record()
            torch.cuda.synchronize()

            logging.info(f'x:{tx} , y: {ty} , z:{tz} , uvst_time : {start_uvst.elapsed_time(end_uvst)}')

            torch.cuda.synchronize()
            
            start_cuda_copy = torch.cuda.Event(enable_timing=True)
            end_cuda_copy = torch.cuda.Event(enable_timing=True)

            start_cuda_copy.record()
            uvst = torch.from_numpy(uvst.astype(np.float32)).cuda()
            end_cuda_copy.record()
            
            torch.cuda.synchronize()
            logging.info(f'x:{tx} , y: {ty} , z:{tz} , cuda_copy_time : {start_cuda_copy.elapsed_time(end_cuda_copy)}')
            torch.cuda.synchronize()

            start_infer = torch.cuda.Event(enable_timing=True)
            end_infer = torch.cuda.Event(enable_timing=True)

            pred_color = []
            n = 4

            start_infer.record()
            streams = [torch.cuda.Stream() for _ in range(n)]
            for i in range(n):
                with torch.cuda.stream(streams[i]):
                    pred_color.append(self.model[i](uvst[i,:,:]))

            torch.cuda.synchronize()
            end_infer.record()



            torch.cuda.synchronize()
            logging.info(f'x:{tx} , y: {ty} , z:{tz} , divided_infer_time : {start_infer.elapsed_time(end_infer)}')
            torch.cuda.synchronize()


            temp = []
            # pred_full = pred_color.reshape((h,w,3)).permute((2,0,1))
            
            
            

            # if(save_flag):
            #     torchvision.utils.save_image(pred_full, save_path)
           
            #return pred_color.reshape((h,w,3))    
            return temp    
        
    def gen_rotating_gif_llff_path(self,savename , n=360,mode = 0,rotnum = 1 ,x_num = 2):
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out   = cv2.VideoWriter(savename+".mp4", fourcc, 24.0, (self.w,self.h))
        view_group = []
        tmax = 1.8
        print("start mode:  " + str(mode))
        

        repeats = np.linspace(0, 2*np.pi, n)     
        for r in repeats:
            if(mode == 0):
                view_unit = self.render_blended_img((0.5+0.5*np.sin(x_num*r)) , 0,((tmax/2)-(tmax/2*np.cos(r))), rotnum*r,self.w,self.h,None,None,False)
            if(mode == 1):
                view_unit = self.render_blended_img((0.5+0.5*np.sin(x_num*r)) , 0,((tmax/2)-(tmax/2*np.cos(r))),np.pi/2,self.w,self.h,None,None,False)
            if(mode == 2):
                view_unit = self.render_blended_img((0.5+0.5*np.sin(r)) , 0,tmax/2, rotnum*r,   self.w,self.h,None,None,False)
            if(mode == 3):
                view_unit = self.render_blended_img((0.5+0.5*np.sin(r)) , 0,tmax/2, 0  ,   self.w,self.h,None,None,False)
            if(mode == 4):
                view_unit = self.render_blended_img((0.5+0.5*np.sin(r)) , 0,tmax/2, np.pi   ,   self.w,self.h,None,None,False)
            if(mode == 5):
                view_unit = self.render_blended_img((0.5+0.5*np.sin(r)) , 0,tmax/2, -np.pi/2,   self.w,self.h,None,None,False)
            if(mode == 6):
                view_unit = self.render_blended_img((0.5+0.5*np.sin(r)) , 0,tmax/2, np.pi/2,    self.w,self.h,None,None,False)
            if(mode == 7):
                view_unit = self.render_blended_img((0.5+0.5*np.sin(r)) , 0,tmax/2, -np.pi/4,   self.w,self.h,None,None,False)
            if(mode == 8):
                view_unit = self.render_blended_img((0.5+0.5*np.sin(r)) , 0,tmax/2, np.pi/4,    self.w,self.h,None,None,False)
            if(mode == 9):
                view_unit = self.render_blended_img(0.5, 0,((tmax/2)-(tmax/2*np.cos(r))), rotnum*r,  self.w,self.h,None,None,False)
            if(mode == 10):
                view_unit = self.render_blended_img(0.5, 0,((tmax/2)-(tmax/2*np.cos(r))), 0,    self.w,self.h,None,None,False)
            if(mode == 11):
                view_unit = self.render_blended_img(0.5, 0,((tmax/2)-(tmax/2*np.cos(r))), np.pi,     self.w,self.h,None,None,False)
            if(mode == 12):
                view_unit = self.render_blended_img(0.5, 0,((tmax/2)-(tmax/2*np.cos(r))), -np.pi/2,  self.w,self.h,None,None,False)
            if(mode == 13):
                view_unit = self.render_blended_img(0.5, 0,((tmax/2)-(tmax/2*np.cos(r))), np.pi/2,   self.w,self.h,None,None,False)
            if(mode == 14):
                view_unit = self.render_blended_img(0.5, 0,((tmax/2)-(tmax/2*np.cos(r))), -np.pi/4,  self.w,self.h,None,None,False)
            if(mode == 15):
                view_unit = self.render_blended_img(0.5, 0,((tmax/2)-(tmax/2*np.cos(r))), np.pi/4,   self.w,self.h,None,None,False)

            view_unit *= 255

            view_unit       = view_unit.cpu().numpy().astype(np.uint8)

            out.write(cv2.cvtColor(view_unit,cv2.COLOR_RGB2BGR))

            view_unit       = imageio.core.util.Array(view_unit)
            view_group.append(view_unit)

        #imageio.mimsave(savename, view_group,fps=30)
        
        out.release()
        print("end mode:  " + str(mode))


    


    


        

        


        

if __name__ == '__main__':

    args = parser.parse_args()

    unit = demo_rgb(args)


    print("finish")
 
