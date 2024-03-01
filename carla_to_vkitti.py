#!/usr/bin/env python

import os
import json
import argparse
import cv2
import numpy as np
import shutil
import argparse

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--cam_num", type=int)
parser.add_argument("--fov", type=int)

def copy_depth(depth_dir, cam_num):
    #reading all images and depth
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])
    
    #creating separate camera folders
    for i in range(cam_num):
        cam_folder = os.path.join('./data/CarlaDataset/frames/depth/', f'Camera_{i}')
        os.makedirs(cam_folder, exist_ok=True)
        
    #copying images to camera-wise folders
    newDepthID = 0
    for i, depth in enumerate(depth_files):
        folder_index = i % cam_num
        src_path = os.path.join(depth_dir, depth)
        img = cv2.imread(src_path)
        dest_path = os.path.join('./data/CarlaDataset/frames/depth/', f'Camera_{folder_index}', f'depth_{newDepthID:05d}.png')
        cv2.imwrite(dest_path, img)
        if folder_index==cam_num-1:
            newDepthID+=1
    
        
def camera_intrinsics(data_dir, cam_num, fov):
    #reading all images and depth
    images = sorted([f for f in os.listdir(data_dir) if f.endswith('.jpg') or f.endswith('.png')])
    
    #creating separate camera folders
    for i in range(cam_num):
        cam_folder = os.path.join('./data/CarlaDataset/frames/rgb/', f'Camera_{i}')
        os.makedirs(cam_folder, exist_ok=True)
    
    #copying images to camera-wise folders
    newImageID = 0
    for i, image in enumerate(images):
        folder_index = i % cam_num
        src_path = os.path.join(data_dir, image)
        img = cv2.imread(src_path)
        dest_path = os.path.join('./data/CarlaDataset/frames/rgb/', f'Camera_{folder_index}', f'rgb_{newImageID:05d}.jpg')
        cv2.imwrite(dest_path, img)
        if folder_index==cam_num-1:
            newImageID+=1
    
    #calculating camera intrinsics
    with open('./data/CarlaDataset/intrinsic.txt', 'w') as f:
        f.write("frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]\n")
        frameID = 0
        cameraID = 0
        for i, image_file in enumerate(images):
            image_path = os.path.join(data_dir, image_file)
            img = cv2.imread(image_path)
            cx = img.shape[1] // 2
            cy = img.shape[0] // 2
            width = img.shape[1]
            height = img.shape[0]
            fov_rad = np.deg2rad(fov)
            fx = width / (2 * np.tan(fov_rad / 2))
            fy = width / (2 * np.tan(fov_rad / 2))
            f.write(f"{frameID} {cameraID} {fx} {fy} {cx} {cy}\n")
            cameraID+=1
            if cameraID == cam_num:
                cameraID = 0
                frameID+=1
    print("Saved camera intrinsics!")

#this function extracts numeric part of image file name for sorting JSON file
def extract_numeric(file_path):
    return int(file_path.split('_')[-1].split('.')[0])
    
def camera_extrinsics(data_dir, cam_num):
    with open(data_dir, 'r') as file:
        data = json.load(file)
        
    # COLMAP might generate transform file in non-sorted order
    sorted_frames = sorted(data['frames'], key=lambda x: extract_numeric(x['file_path']))
    transform_matrices = [frame['transform_matrix'] for frame in sorted_frames]
    
    #calculating camera extrinsics
    flattened_matrices = []
    for matrix in transform_matrices:
        flattened_matrix = [str(value) for row in matrix for value in row]
        flattened_matrices.append(flattened_matrix)
    
    with open('./data/CarlaDataset/extrinsic.txt', 'w') as file:
        file.write("frame cameraID r1,1 r1,2 r1,3 t1 r2,1 r2,2 r2,3 t2 r3,1 r3,2 r3,3 t3 0 0 0 1\n")
        frameID = 0
        cameraID = 0
        for matrix in flattened_matrices:
            ext = ' '.join(matrix)
            file.write(f"{frameID} 0 {ext}\n")
            cameraID+=1
            file.write(f"{frameID} 1 {ext}\n")
            cameraID+=1
            if cameraID == cam_num:
                cameraID = 0
                frameID+=1
    
    print("Saved camera extrinsics!")
    
def dummy_kitti_files(data_dir, cam_num):
    images = sorted([f for f in os.listdir(data_dir) if f.endswith('.jpg') or f.endswith('.png')])
    
    with open('./data/CarlaDataset/bbox.txt', 'w') as f:
        f.write("frame cameraID trackID left right top bottom number_pixels truncation_ratio occupancy_ratio isMoving\n")
        frameID = 0
        cameraID = 0
        for i, image_file in enumerate(images):
            f.write(f"{frameID} {cameraID} 0 0 0 0 0 0 0 0 False\n")
            cameraID+=1
            if cameraID == cam_num:
                cameraID = 0
                frameID+=1
                
    with open('./data/CarlaDataset/pose.txt', 'w') as f:
        f.write("frame cameraID trackID alpha width height length world_space_X world_space_Y world_space_Z rotation_world_space_y rotation_world_space_x rotation_world_space_z camera_space_X camera_space_Y camera_space_Z rotation_camera_space_y rotation_camera_space_x rotation_camera_space_z\n")
        frameID = 0
        cameraID = 0
        for i, image_file in enumerate(images):
            f.write(f"{frameID} {cameraID} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n")
            cameraID+=1
            if cameraID == cam_num:
                cameraID = 0
                frameID+=1
    
    print("Saved dummy bbox and pose files!")

if __name__ == "__main__":
    args = parser.parse_args()
    image_dir = os.path.join(args.data_dir, "images")
    depth_dir = os.path.join(args.data_dir, "depth")
    transform_dir = os.path.join(args.data_dir, "transforms.json")
    cam_num = args.cam_num
    fov = args.fov

    #copy_depth(depth_dir, cam_num)
    camera_intrinsics(image_dir, cam_num, fov)
    camera_extrinsics(transform_dir, cam_num)
    dummy_kitti_files(image_dir, cam_num)

