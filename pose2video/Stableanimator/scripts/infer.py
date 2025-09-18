import os
import shutil
import subprocess
from glob import glob
import argparse

def run_inference(raw_path, devices='1,2,3,4',
                  mode='sample_images',
                  posenet_model_name_or_path="path/checkpoints/Animation/pose_net.pth", \
                  face_encoder_model_name_or_path="path/checkpoints/Animation/face_encoder.pth", \
                  unet_model_name_or_path="path/checkpoints/Animation/unet.pth", \
                  ):
    output_base_path = raw_path+'-copy'
    # mode control: sample_images refer to sample_pose to render，poses refer to pose to render，smooth_gt_imagesrefer to gt_smooth_pose to render
    # modes = ['sample_images', 'poses', 'smooth_gt_images']
    modes = [mode]
    gpus = devices.split(',')

    num_gpus = len(gpus)
    # Create output directories
    os.makedirs(output_base_path, exist_ok=True)
    for i in range(num_gpus):
        os.makedirs(os.path.join(output_base_path, f'part_{i}'), exist_ok=True)

    # Distribute files into 8 parts
    all_files = sorted(glob(os.path.join(raw_path, '*')))
    for idx, file in enumerate(all_files):
        part_idx = idx % num_gpus
        shutil.move(file, os.path.join(output_base_path, f'part_{part_idx}', os.path.basename(file)))

    # Run inference for each mode sequentially
    for mode in modes:
        processes = []
        for i in range(num_gpus):
            cmd = f'CUDA_VISIBLE_DEVICES={gpus[i]} OMP_NUM_THREADS=8 python inference_use.py --raw_path {os.path.join(output_base_path, f"part_{i}")} --mode {mode} --gradient_checkpointing --height 512 --width 512 \
            --posenet_model_name_or_path {posenet_model_name_or_path} --face_encoder_model_name_or_path {face_encoder_model_name_or_path} --unet_model_name_or_path {unet_model_name_or_path}'
            p = subprocess.Popen(cmd, shell=True)
            processes.append(p)
        for p in processes:
            p.wait()
            
    # Move files back to the original directory
    for i in range(num_gpus):
        part_path = os.path.join(output_base_path, f'part_{i}')
        for file in glob(os.path.join(part_path, '*')):
            shutil.move(file, raw_path)
        os.rmdir(part_path)
    os.system(f'rm -r {output_base_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', type=str, default='', help='Path to the ready to infer path.')
    parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6,7')
    parser.add_argument('--mode', type=str, default='sample_images', help="Choose from ['sample_images', 'poses', 'smooth_gt_images']")
    parser.add_argument(
        "--posenet_model_name_or_path",
        type=str,
        default='./checkpoints/Animation/pose_net.pth',
        help="Path to pretrained posenet model",
    )
    parser.add_argument(
        "--face_encoder_model_name_or_path",
        type=str,
        default='./checkpoints/Animation/face_encoder.pth',
        help="Path to pretrained face encoder model",
    )
    parser.add_argument(
        "--unet_model_name_or_path",
        type=str,
        default='./checkpoints/Animation/unet.pth',
        help="Path to pretrained unet model",
    )
    args = parser.parse_args()
    run_inference(args.raw_path, args.devices, args.mode, args.posenet_model_name_or_path, args.face_encoder_model_name_or_path, args.unet_model_name_or_path)