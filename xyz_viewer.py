from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import os
import torch
import numpy as np
import glob
import argparse #parser
import time

from src.model import Nerf4D_relu_ps
from src.utils import rm_folder, rm_folder_keep
import torchvision

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

image_folder = "static/image"
current_image_index = 0

parser = argparse.ArgumentParser() # museum,column2
parser.add_argument('--exp_name',type=str, default = '03_9to14_d4w64',help = 'exp name')
# parser.add_argument('--exp_name',type=str, default = '1108_3_9to14',help = 'exp name')
parser.add_argument('--gpuid',type=str, default = '0',help='data folder name')
parser.add_argument('--mlp_depth', type=int, default = 4)
parser.add_argument('--mlp_width', type=int, default = 64)
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
                ckpt_id = int(os.path.basename(ckpt_path).split(".")[0].split("-")[1])
                self.iter = max(self.iter, ckpt_id)
            ckpt_name = f"./{self.checkpoints}/nelf-{self.iter}.pth"
        print(f"Load `` from {ckpt_name}")
        ckpt = torch.load(ckpt_name)
        print(ckpt)
        self.model.load_state_dict(ckpt)
        

    def get_vec3_xyz(self,x,y,z,H=720,W=720):
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
        print('get image called')

        vec3_xyz = torch.from_numpy(vec3_xyz.astype(np.float32)).cuda()
        pred_color = self.model(vec3_xyz)
        pred_img = pred_color.reshape((720,720,3)).permute((2,0,1))
        torchvision.utils.save_image(pred_img, save_path)

demo_instance = None  # Initialize demo instance

@app.route("/static/<path:filename>")
def static_file(filename):
    response = send_from_directory('static', filename)
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route("/")
def home():
    return render_template('home.html')

@socketio.on('request_new_image')
def handle_request_new_image(data):
    global demo_instance

    x = float(data.get('x', 0))
    y = float(data.get('y', 0))
    z = float(data.get('z', 0))

    # print(x, y, z)
    if demo_instance is None:
        args = parser.parse_args()
        socketio.emit('size', {'depth': args.mlp_depth, 'width': args.mlp_width})
        demo_instance = demo_rgb(args)
        
    save_path = 'static/generated_image.png'

    start = time.time()
    vec3_xyz = demo_instance.get_vec3_xyz(x, y, z)
    demo_instance.get_image(vec3_xyz, save_path)
    end = time.time()
    time_val = str(end - start)
    socketio.emit('new_image', {'image_file': 'generated_image.png', 'time' : time_val})
    print('emit finished')
    print(f"Time taken: " + time_val + " seconds")

if __name__ == "__main__":
    socketio.run(app, debug=True, host='0.0.0.0', port=6006)