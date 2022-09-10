import open3d as o3d
import sys,math,random
import os,subprocess
import re
import scipy.io
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
sys.path.append(dir_path+'/scripts/')
from eval_ycb import VOCap
from multiprocessing import cpu_count
import argparse
import torch
from torch import optim
from Utils import *
import numpy as np
import yaml
from data_augmentation import *
from se3_tracknet import *
from datasets import *
from offscreen_renderer import *
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import cv2
from PIL import Image
import copy
import glob
import mpl_toolkits.mplot3d.axes3d as p3
import transformations as T
import Utils as U
from scipy import spatial
from multiprocessing import Pool
import multiprocessing
from functools import partial
from itertools import repeat
import multiprocessing as mp
from vispy_renderer import VispyRenderer
import json
from pyquaternion import Quaternion
from numpy import genfromtxt
from tqdm import tqdm

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = True


def project_points(points,K):
        us = np.divide(points[:,0]*K[0,0],points[:,2]) + K[0,2]
        vs = np.divide(points[:,1]*K[1,1],points[:,2]) + K[1,2]
        us = np.round(us).astype(np.int32).reshape(-1,1)
        vs = np.round(vs).astype(np.int32).reshape(-1,1)
        return np.hstack((us,vs))


class Tracker:
        def __init__(self, dataset_info, images_mean, images_std, ckpt_dir, trans_normalizer=0.03, rot_normalizer=5*np.pi/180):
                self.dataset_info = dataset_info
                self.image_size = (dataset_info['resolution'], dataset_info['resolution'])
                self.object_cloud = o3d.io.read_point_cloud(dataset_info['models'][0]['model_path'])
                self.object_cloud = self.object_cloud.voxel_down_sample(voxel_size=0.005)

                print('self.object_cloud loaded and downsampled')
                if 'object_width' not in dataset_info:
                        object_max_width = compute_obj_max_width(self.object_cloud)
                        bounding_box = dataset_info['boundingbox']
                        with_add = bounding_box / 100 * object_max_width
                        self.object_width = object_max_width + with_add
                else:
                        self.object_width = dataset_info['object_width']
                print('self.object_width=',self.object_width)

                self.mean = images_mean
                self.std = images_std
                cam_cfg = dataset_info['camera']
                self.K = np.array([cam_cfg['focalX'], 0, cam_cfg['centerX'], 0, cam_cfg['focalY'], cam_cfg['centerY'], 0,0,1]).reshape(3,3)

                print('Loading ckpt from ',ckpt_dir)
                checkpoint = torch.load(ckpt_dir)
                print('pose track ckpt epoch={}'.format(checkpoint['epoch']))

                self.model = Se3TrackNet(image_size=self.image_size[0])
                self.model.load_state_dict(checkpoint['state_dict'])
                self.model = self.model.cuda()
                self.model.eval()

                if 'renderer' in dataset_info and dataset_info['renderer']=='pyrenderer':
                        print('Using pyrenderer')
                        self.renderer = Renderer([dataset_info['models'][0]['obj_path']],self.K,cam_cfg['height'],cam_cfg['width'])
                else:
                        print('Using vispy renderer')
                        self.renderer = VispyRenderer(dataset_info['models'][0]['model_path'], self.K, H=dataset_info['resolution'], W=dataset_info['resolution'])

                self.prev_rgb = None
                self.prev_depth = None
                self.frame_cnt = 0
                self.errs = []

                posttransforms = Compose([OffsetDepth(),NormalizeChannels(images_mean, images_std),ToTensor()])

                self.dataset = TrackDataset('','eval',images_mean, images_std,None,None,posttransforms,dataset_info, trans_normalizer=trans_normalizer, rot_normalizer=rot_normalizer)

        def render_window(self, ob2cam):
                '''
                @ob2cam: 4x4 mat ob in opencv cam
                '''
                glcam_in_cvcam = np.array([[1,0,0,0],
                                                                                                                        [0,-1,0,0],
                                                                                                                        [0,0,-1,0],
                                                                                                                        [0,0,0,1]])
                bbox = compute_bbox(ob2cam, self.K, self.object_width, scale=(1000, -1000, 1000))
                ob2cam_gl = np.linalg.inv(glcam_in_cvcam).dot(ob2cam)
                left = np.min(bbox[:, 1])
                right = np.max(bbox[:, 1])
                top = np.min(bbox[:, 0])
                bottom = np.max(bbox[:, 0])
                if isinstance(self.renderer,VispyRenderer):
                        self.renderer.update_cam_mat(self.K, left, right, bottom, top)
                        render_rgb, render_depth = self.renderer.render_image(ob2cam_gl)
                else:
                        bbox = compute_bbox(ob2cam, self.K, self.object_width, scale=(1000, 1000, 1000))
                        rgb, depth = self.renderer.render([ob2cam])
                        depth = (depth*1000).astype(np.uint16)
                        try:
                                render_rgb, render_depth = crop_bbox(rgb, depth, bbox, self.image_size)
                        except:
                                print('fixed bbox')
                                render_rgb, render_depth = crop_bbox_fixed(rgb, depth, bbox, self.image_size)
                return render_rgb, render_depth

        def on_track(self, prev_pose, current_rgb, current_depth, gt_B_in_cam=None, debug=False, samples=1):
                K = self.K
                A_in_cam = prev_pose.copy()
                glcam_in_cvcam = np.array([[1,0,0,0],
                                                                                                                        [0,-1,0,0],
                                                                                                                        [0,0,-1,0],
                                                                                                                        [0,0,0,1]])

                bbs = []
                sample_poses = []
                rgbBs = []
                depthBs = []
                for i in range(samples):
                        if i==0:
                                sample_pose = prev_pose.copy()
                        bb = compute_bbox(sample_pose, self.K, self.object_width, scale=(1000, 1000, 1000))
                        bbs.append(bb)
                        sample_poses.append(sample_pose)
                        try:
                                rgbB, depthB = crop_bbox(current_rgb, current_depth, bb, self.image_size)
                        except:
                                print('fixed bbox')
                                rgbB, depthB = crop_bbox_fixed(current_rgb, current_depth, bb, self.image_size)
                        rgbBs.append(rgbB)
                        depthBs.append(depthB)

                sample_poses = np.array(sample_poses)
                bbs = np.array(bbs)

                rgbAs = []
                depthAs = []
                maskAs = []
                for i in range(samples):
                        rgbA, depthA = self.render_window(sample_poses[i])
                        maskA = depthA>100
                        rgbAs.append(rgbA)
                        depthAs.append(depthA)
                        maskAs.append(maskA)

                rgbAs,depthAs,maskAs = list(map(np.array, [rgbAs,depthAs,maskAs]))

                rgbAs_backup = rgbAs.copy()
                rgbBs_backup = rgbBs.copy()

                if gt_B_in_cam is None:
                        # print('**** gt_B_in_cam set to Identity')
                        gt_B_in_cam = np.eye(4)

                dataAs = []
                dataBs = []
                for i in range(samples):
                        sample = self.dataset.processData(rgbAs[i],depthAs[i],sample_poses[i],rgbBs[i],depthBs[i],gt_B_in_cam)[0]
                        dataAs.append(sample[0].unsqueeze(0))
                        dataBs.append(sample[1].unsqueeze(0))
                dataA = torch.cat(dataAs,dim=0).cuda().float()
                dataB = torch.cat(dataBs,dim=0).cuda().float()

                with torch.no_grad():
                        prediction = self.model(dataA,dataB)

                pred_B_in_cams = []
                for i in range(samples):
                        trans_pred = prediction['trans'][i].data.cpu().numpy()
                        rot_pred = prediction['rot'][i].data.cpu().numpy()
                        pred_B_in_cam = self.dataset.processPredict(sample_poses[i],(trans_pred,rot_pred))
                        pred_B_in_cams.append(pred_B_in_cam)

                pred_B_in_cams = np.array(pred_B_in_cams)
                final_estimate = pred_B_in_cams[0].copy()
                self.prev_rgb = current_rgb
                self.prev_depth = current_depth
                pred_color, pred_depth = self.render_window(final_estimate)
                canvas = makeCanvas([rgbBs_backup[0], pred_color], flipBR=True)
                cv2.imshow('AB',canvas)
                if self.frame_cnt==0:
                                cv2.waitKey(1)
                else:
                                cv2.waitKey(1)
                self.frame_cnt += 1

                if samples==1:
                        return pred_B_in_cams[0]

                return pred_B_in_cams[0]


def extract_gt(path):
        with open(path + str(0).zfill(6) + '.json', 'r', encoding='utf-8') as f:
                info_json = json.load(f)

        q = np.array(info_json['objects'][0]['quaternion_xyzw'])[[3, 0, 1, 2]]

        pose = Quaternion(q).transformation_matrix
        p = np.array(info_json['objects'][0]['location'])
        pose[0:3, 3] = p

        pose[0:3, 3] /= 100

        return pose


def load_dope_pose(path):
        data = np.loadtxt(path + '/dope_poses_ycb.txt')

        pose = Quaternion(axis = data[0, 3:6], angle = data[0, 6]).transformation_matrix
        pose[0:3, 3] = data[0, 0:3]

        return pose


def predictSequenceYcb(path, init_pose, sequence_type):
        debug = True
        test_data_path = os.path.join(path, path.split('/')[-1], 'photorealistic1/')
        depth_postfix = '.depth.mm.16.png'

        if sequence_type == 'real':
                test_data_path = os.path.join(path, 'photorealistic1/')
                depth_postfix = '.depth.png'

        out_dir = outdir

        if sequence_type == 'synthetic':
                poses_indexes = genfromtxt(os.path.join(path, 'index.csv'))
                poses_indexes = poses_indexes.astype(int)
        elif sequence_type == 'real':
                rgb_file_list = glob.glob(test_data_path + '/*.depth.png')
                poses_indexes = np.array(range(len(rgb_file_list)))

        prev_pose = init_pose.copy()
        pred_poses = [prev_pose]

        tracker = Tracker(dataset_info, images_mean, images_std,ckpt_dir)

        K = tracker.K.copy()

        H = 480
        W = 640

        for i in tqdm(poses_indexes):

                if i == 0:
                        continue

                try:
                        rgb = np.array(Image.open(test_data_path + '/' + str(i).zfill(6) + '.png'))
                except FileNotFoundError as e:
                        print(e)
                        break
                rgb = rgb[:, :, 0:3]

                if rgb is None:
                        break
                rgb_viz = rgb.copy()
                depth = cv2.imread(test_data_path + '/' + str(i).zfill(6) + depth_postfix, cv2.IMREAD_UNCHANGED).astype(np.uint16)

                A_in_cam = prev_pose.copy()
                cur_pose = tracker.on_track(A_in_cam, rgb, depth, gt_B_in_cam=None, debug=False,samples=1)
                A_in_cam = cur_pose.copy()

                prev_pose = cur_pose.copy()
                pred_poses.append(cur_pose)

                model = copy.deepcopy(tracker.object_cloud)
                model.transform(cur_pose)
                K = tracker.K.copy()
                uvs = project_points(np.asarray(model.points),K)
                cur_bgr = cv2.cvtColor(rgb_viz,cv2.COLOR_RGB2BGR)
                for ii in range(len(uvs)):
                        cv2.circle(cur_bgr,(uvs[ii,0],uvs[ii,1]),radius=1,color=(0,255,255),thickness=-1)
                cv2.putText(cur_bgr,"frame:{}".format(i), (W//2,H-50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,thickness=4,color=(255,0,0))

                if debug:
                        cv2.imwrite(out_dir+'%07d.png'%(i),cur_bgr)
                cur_bgr = cv2.resize(cur_bgr,(W//2,H//2))

        pred_poses = np.array(pred_poses)

        return pred_poses

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--ycb_dir', default='/home/user/iros20-6d-pose-tracking/datasets/YCB_Video_Dataset')
        parser.add_argument('--sequence_path')

        args = parser.parse_args()

        outdir = args.sequence_path + '/output/'
        os.makedirs(outdir, exist_ok = True)

        dataset_info_path = './info_custom.yml'
        with open(dataset_info_path,'r') as ff:
                dataset_info = yaml.safe_load(ff)

        object_name = ''
        with open(args.sequence_path + '/object_name.txt', 'r') as ff:
                object_name = ff.readline().rstrip('\n')

        ckpt_root = '/home/user/iros20-6d-pose-tracking/YCB_weights/' + '_'.join(object_name.split('_')[1:])
        ckpt_map = {
                '002_master_chef_can' : ckpt_root + '/model_epoch180.pth.tar',
                '003_cracker_box' : ckpt_root + '/model_epoch165.pth.tar',
                '004_sugar_box' : ckpt_root + '/model_epoch150.pth.tar',
                '005_tomato_soup_can' : ckpt_root + '/model_epoch270.pth.tar',
                '006_mustard_bottle' : ckpt_root + '/model_epoch150.pth.tar',
                '007_tuna_fish_can' : ckpt_root + '/model_epoch230.pth.tar',
                '008_pudding_box' : ckpt_root + '/model_epoch180.pth.tar',
                '009_gelatin_box' : ckpt_root + '/model_last.pth.tar',
                '010_potted_meat_can' : ckpt_root + '/model_epoch210.pth.tar',
                '011_banana' : ckpt_root + '/model_epoch160.pth.tar',
                '019_pitcher_base' : ckpt_root + '/model_epoch160.pth.tar',
                '021_bleach_cleanser' : ckpt_root + '/model_best_val.pth.tar',
                '024_bowl' : ckpt_root + '/model_epoch225.pth.tar',
                '025_mug' : ckpt_root + '/model_epoch235.pth.tar',
                '035_power_drill' : ckpt_root + '/model_epoch215.pth.tar',
                '036_wood_block' : ckpt_root + '/model_epoch175.pth.tar',
                '037_scissors' : ckpt_root + '/model_best_val.pth.tar',
                '040_large_marker' : ckpt_root + '/model_epoch240.pth.tar',
                '051_large_clamp' : ckpt_root + '/model_epoch165.pth.tar',
                '052_extra_large_clamp' : ckpt_root + '/model_best_val.pth.tar',
                '061_foam_brick' : ckpt_root + '/model_epoch155.pth.tar'
                }


        ckpt_dir = ckpt_map[object_name]
        images_mean = np.load(os.path.join(ckpt_root, "mean.npy"))
        images_std = np.load(os.path.join(ckpt_root, "std.npy"))

        dataset_info['models'][0]['model_path'] = '/home/user/iros20-6d-pose-tracking/datasets/YCB_Video_Dataset/CADmodels/' + object_name + '/textured.ply'
        dataset_info['models'][0]['obj_path'] = '/home/user/iros20-6d-pose-tracking/datasets/YCB_Video_Dataset/models/' + object_name + '/textured.obj'

        # Evaluate YCB to YCBV transformation
        ycb_mesh_path = '/home/user/iros20-6d-pose-tracking/datasets/Dataset_Utilities/nvdu/data/ycb/original/' + object_name + '/google_16k/textured.obj'
        ycb_mesh = trimesh.load(ycb_mesh_path)

        ycbv_mesh_path = dataset_info['models'][0]['obj_path']
        ycbv_mesh = trimesh.load(ycbv_mesh_path)

        transform = np.eye(4)
        transform[0:3, 3] = ycb_mesh.centroid - ycbv_mesh.centroid

        # Extract initial pose in YCB frame
        init_pose = load_dope_pose(args.sequence_path)
        init_pose = init_pose @ transform

        print('*********************************************************')
        print(object_name)
        print('dataset_info_path', dataset_info_path)
        print('mean: ' + os.path.join(ckpt_root, "mean.npy"))
        print('std: ' + os.path.join(ckpt_root, "std.npy"))
        print('ckpt: ' + ckpt_dir)
        print('ply: ' + dataset_info['models'][0]['model_path'])
        print('obj: ' + dataset_info['models'][0]['obj_path'])
        print('out: ' + outdir)
        print('*********************************************************')

        predicted_poses = predictSequenceYcb(args.sequence_path, init_pose, args.sequence_type)
        print('Number of poses:')
        print(predicted_poses.shape)

        # Transform poses in the YCB reference frame
        header = 'p_x,p_y,p_z,q_1,q_2,q_3,q_4'
        output_data = np.zeros(shape = (predicted_poses.shape[0], 7))
        for i in range(predicted_poses.shape[0]):
                pose = predicted_poses[i, :] @ np.linalg.inv(transform)

                u, s, vh = np.linalg.svd(pose[0:3, 0:3], full_matrices=False)
                quaternion = Quaternion(matrix = u @ vh)


                output_data[i, 0:3] = pose[0:3, 3]
                output_data[i, 3:7] = [quaternion.w, quaternion.x, quaternion.y, quaternion.z]

        np.savetxt(os.path.join(outdir, args.sequence_path.split('/')[-1] + '_se3_pose.csv'), output_data, delimiter = ',', header = header)
