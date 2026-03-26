import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore')  # TODO

import argparse
import cv2
import glob
import imageio
import logging
import os
import numpy as np
from typing import List
import copy

import omegaconf
import onnxruntime as ort
import open3d as o3d
import torch
import yaml
import time
#from onnx_tensorrt import tensorrt_engine
import tensorrt as trt

import sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

from Utils import *
from core.foundation_stereo import FoundationStereo
from core.utils.utils import InputPadder


def preprocess(image_path, args):
    input_image = imageio.imread(image_path)
    if args.height and args.width:
      input_image = cv2.resize(input_image, (args.width, args.height))
    resized_image = torch.as_tensor(input_image.copy()).float()[None].permute(0, 3, 1, 2).contiguous()
    return resized_image, input_image


def get_onnx_model(args):
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    #options: CUDAExecutionProvider, TensorrtExecutionProvider
    model = ort.InferenceSession(args.pretrained, sess_options=session_options,
                                 providers=['TensorrtExecutionProvider'])
    return model


def get_engine_model(args):
    with open(args.pretrained, 'rb') as file:
        engine_data = file.read()
    engine = None
    # engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(engine_data)
    # engine = tensorrt_engine.Engine(engine)
    return engine


def inference(left_img_path: str, right_img_path: str, model, args: argparse.Namespace):
    left_img, input_left = preprocess(left_img_path, args)
    right_img, _ = preprocess(right_img_path, args)

    torch.cuda.synchronize()
    start_time = time.time()
    if args.pretrained.endswith('.onnx'):
        left_disp = model.run(None, {'left_image': left_img.numpy(),
                                     'right_image': right_img.numpy()})[0]
    else:
        left_disp = model.run([left_img.numpy(), right_img.numpy()])[0]
    torch.cuda.synchronize()
    end_time = time.time()
    logging.info(f'Inference time: {end_time - start_time:.3f} seconds')

    left_disp = left_disp.squeeze()  # HxW

    vis = vis_disparity(left_disp)
    vis = np.concatenate([input_left, vis], axis=1)
    img_directory, left_img_filename = os.path.split(left_img_path)
    imageio.imwrite(os.path.join(args.save_path, 'visual', left_img_filename), vis)

    if args.pc:
        save_path = left_img_path.split('/')[-1].split('.')[0] + '.ply'
        baseline = 193.001 / 1e3
        doffs = 0
        K = np.array([1998.842, 0, 588.364,
                      0, 1998.842, 505.864,
                      0, 0, 1]).reshape(3, 3)
        depth = K[0, 0] * baseline / (left_disp + doffs)
        xyz_map = depth2xyzmap(depth, K)
        pcd = toOpen3dCloud(xyz_map.reshape(-1,3), input_left.reshape(-1,3))
        keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=args.z_far)
        keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
        pcd = pcd.select_by_index(keep_ids)
        o3d.io.write_point_cloud(os.path.join(args.save_path, 'cloud', save_path), pcd)

    return end_time - start_time


def parse_args() -> omegaconf.OmegaConf:
    parser = argparse.ArgumentParser(description='Stereo 2025')
    code_dir = os.path.dirname(os.path.realpath(__file__))

    # File options
    parser.add_argument('--input_dir', '-i', required=True,
                        help='Directory containing subfolders with left/right image pairs.')
    parser.add_argument('--save_path', '-s', default=f'{code_dir}/../output',
                        help='Path to save results.')
    parser.add_argument('--pretrained', default='2024-12-13-23-51-11/model_best_bp2.pth',
                        help='Path to pretrained model')

    # Inference options
    parser.add_argument('--height', type=int, default=448, help='Image height')
    parser.add_argument('--width', type=int, default=672, help='Image width')
    parser.add_argument('--pc', action='store_true', help='Save point cloud')
    parser.add_argument('--z_far', default=100, type=float, help='max depth to clip in point cloud')
    args = parser.parse_args()
    return args


def main():
    start_time = time.time()
    args = parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    paths = ['continuous/disparity', 'visual', 'denoised_cloud', 'cloud']
    for p in paths:
        os.makedirs(os.path.join(args.save_path, p), exist_ok=True)

    assert os.path.isfile(args.pretrained), f'Pretrained model {args.pretrained} not found'
    logging.info('Pretrained model loaded from %s', args.pretrained)
    set_seed(0)
    if args.pretrained.endswith('.onnx'):
        model = get_onnx_model(args)
    elif args.pretrained.endswith('.engine') or args.pretrained.endswith('.plan'):
        model = get_engine_model(args)
    else:
        assert False, f'Unknown model format {args.pretrained}.'

    # Process each subdirectory containing left/right image pairs
    subdirs = [d for d in sorted(os.listdir(args.input_dir))
               if os.path.isdir(os.path.join(args.input_dir, d))]
    
    end_time = time.time()
    logging.info(f'Startup time: {end_time - start_time:.3f} seconds')

    times = []
    for sub in subdirs:
        sub_path = os.path.join(args.input_dir, sub)
        # Find left and right images (assuming naming pattern *_left.* and *_right.*)
        left_candidates = [f for f in os.listdir(sub_path)
                           if f.lower().endswith('_left.png') or
                           f.lower().endswith('_left.jpg') or
                           f.lower().endswith('_left.jpeg')]
        right_candidates = [f for f in os.listdir(sub_path)
                            if f.lower().endswith('_right.png') or
                            f.lower().endswith('_right.jpg') or
                            f.lower().endswith('_right.jpeg')]
        if not left_candidates or not right_candidates:
            logging.warning(f'No left/right pair found in {sub_path}, skipping.')
            continue
        left_img_path = os.path.join(sub_path, left_candidates[0])
        right_img_path = os.path.join(sub_path, right_candidates[0])
        inftime = inference(left_img_path, right_img_path, model, args)
        times.append(inftime)
    
    print(f"Average time: {sum(times)/len(times)}")
    print(f"Max time: {max(times)}")
    print(f"Min time: {min(times)}")

    print(f"Average time (excluding 1st): {sum(times[1:])/len(times)}")
    print(f"Max time(excluding 1st): {max(times[1:])}")
    print(f"Min time(excluding 1st): {min(times[1:])}")


if __name__ == '__main__':
    main()