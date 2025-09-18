import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
import argparse

def copy_and_ready_ref_image(txt_file, raw_ref_img_path, raw_sample_pose_path, raw_mp4_path, save_ready_path, cfg='3.5', if_gt=False):
    os.makedirs(save_ready_path, exist_ok=True)

    with open(txt_file, 'r') as file:
        for line in file:
            vid1, vid2 = line.strip().split()
            save_folder_long = os.path.join(save_ready_path, vid1+'_'+vid1)
            save_folder_long_unpaired = os.path.join(save_ready_path, f'unpaired_cfg_{cfg}_'+vid1+'_'+vid2)
            os.makedirs(save_folder_long, exist_ok=True)
            os.makedirs(save_folder_long_unpaired, exist_ok=True)
            long_vid1 = vid1+'_'+vid1
            long_unpaired_vid2 = f'unpaired_cfg_{cfg}_'+vid1+'_'+vid2
            if if_gt:
                os.makedirs(os.path.join(save_folder_long, 'gt_poses'), exist_ok=True)
                os.makedirs(os.path.join(save_folder_long_unpaired, 'gt_poses'), exist_ok=True)
                shutil.copy(os.path.join(raw_sample_pose_path, long_vid1+'.npy'), os.path.join(save_folder_long, 'gt_poses', long_vid1+'.npy'))
                shutil.copy(os.path.join(raw_sample_pose_path, long_unpaired_vid2+'.npy'), os.path.join(save_folder_long_unpaired, 'gt_poses', long_unpaired_vid2+'.npy'))
            else:
                os.makedirs(os.path.join(save_folder_long, 'sample_poses'), exist_ok=True)
                os.makedirs(os.path.join(save_folder_long_unpaired, 'sample_poses'), exist_ok=True)
                shutil.copy(os.path.join(raw_sample_pose_path, long_vid1+'.npy'), os.path.join(save_folder_long, 'sample_poses', long_vid1+'.npy'))
                shutil.copy(os.path.join(raw_sample_pose_path, long_unpaired_vid2+'.npy'), os.path.join(save_folder_long_unpaired, 'sample_poses', long_unpaired_vid2+'.npy'))
            shutil.copy(os.path.join(raw_ref_img_path, vid1+'.png'), os.path.join(save_folder_long, long_vid1+'.png'))
            shutil.copy(os.path.join(raw_ref_img_path, vid2+'.png'), os.path.join(save_folder_long_unpaired, long_unpaired_vid2+'.png'))
            shutil.copy(os.path.join(raw_mp4_path, vid1+'.mp4'), os.path.join(save_folder_long, long_vid1+'.mp4'))
            shutil.copy(os.path.join(raw_mp4_path, vid2+'.mp4'), os.path.join(save_folder_long_unpaired, long_unpaired_vid2+'.mp4'))
            

def run_inference(txt_file, raw_sample_pose_path, raw_ref_img_path, raw_mp4_path, save_ready_path, devices='1,2,3,4', cfg='3.5', if_gt=False):
    gpus = devices.split(',')
    gpu_num = len(gpus)
    os.makedirs(save_ready_path, exist_ok=True)
    os.makedirs(save_ready_path+'-copy', exist_ok=True)
    
    copy_and_ready_ref_image(txt_file, raw_ref_img_path, raw_sample_pose_path, raw_mp4_path, save_ready_path, cfg=cfg, if_gt=if_gt)

    def create_subfolders(raw_path, output_path, num_folders=gpu_num):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        subfolders = [os.path.join(output_path, f'subfolder_{i}') for i in range(num_folders)]
        for subfolder in subfolders:
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)
        
        all_files = [os.path.join(raw_path, f) for f in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, f))]
        for i, file in enumerate(all_files):
            shutil.move(file, subfolders[i % num_folders])
        
        return subfolders

    def run_command(command, env):
        subprocess.run(command, shell=True, env=env)

    def run_programs_on_subfolders(subfolders, if_gt=False):
        if if_gt:
            commands = [
                "OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES={gpu} python ge_kp2kpvideo.py --raw_path {path} --if_gt"
            ]
        else:
            commands = [
                "OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES={gpu} python ge_kp2kpvideo.py --raw_path {path}",
                # "CUDA_VISIBLE_DEVICES={gpu} python ge_kp2kpvideo.py --raw_path {path} --if_gt"
                # "cd DWPose && OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES={gpu} python skeleton_extraction.py --raw_path {path} && cd ..",
            ]

        env = os.environ.copy()
        with ThreadPoolExecutor(max_workers=gpu_num) as executor:
            for i, subfolder in enumerate(subfolders):
                for j, command in enumerate(commands):
                    gpu_id = gpus[i % gpu_num]

                    formatted_command = command.format(gpu=gpu_id, path=subfolder)
                    executor.submit(run_command, formatted_command, env)

    def move_files_back(subfolders, raw_path):
        for subfolder in subfolders:
            for item in os.listdir(subfolder):
                shutil.move(os.path.join(subfolder, item), raw_path)
            os.rmdir(subfolder)

    # ready 处理
    subfolders = create_subfolders(save_ready_path, save_ready_path+'-copy')
    run_programs_on_subfolders(subfolders, if_gt=if_gt)
    move_files_back(subfolders, save_ready_path)
    os.system(f'rm -rf {save_ready_path}-copy')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='', help='Path to the dataset.')
    parser.add_argument('--save_stage1_path', type=str, default='', help='Path to save stage1 outputs.')
    parser.add_argument('--output_path', type=str, default='', help='Path to save outputs.')
    parser.add_argument('--cfg', type=str, default='3.5')
    parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6,7')
    parser.add_argument('--if_gt', action='store_true', help='Whether to process gt kps.')
    args = parser.parse_args()
    test_split_txt_file = os.path.join(args.dataset_path, 'test.txt')
    raw_sample_pose_path = os.path.join(args.save_stage1_path, 'sample_pose')
    raw_ref_img_path = os.path.join(args.output_path, 'ref_images_512')
    raw_mp4_path = os.path.join(args.output_path, 'mp4_cropped')
    save_ready_path = os.path.join(args.output_path, 'results')
    run_inference(test_split_txt_file, raw_sample_pose_path, raw_ref_img_path, raw_mp4_path, save_ready_path, args.devices, args.cfg, args.if_gt)
