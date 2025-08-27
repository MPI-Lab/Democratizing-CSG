import numpy as np
import os

motion_folder = '/home/yangxu/tencent-cospeech/2d_kp_generation_fixed/dataset/miniset_40min_raw/motions'
filter_motion_folder = '/home/yangxu/tencent-cospeech/2d_kp_generation_fixed/dataset/miniset_40min/motions'
filter_motion_folder_normal = '/home/yangxu/tencent-cospeech/2d_kp_generation_fixed/dataset/miniset_40min/motions_normalized'
filter_motion_folder_meta = "/home/yangxu/tencent-cospeech/2d_kp_generation_fixed/dataset/miniset_40min/metadata"
filter_motion_folder_audios = '/home/yangxu/tencent-cospeech/2d_kp_generation_fixed/dataset/miniset_40min/audios'
filter_motion_folder_vis = '/home/yangxu/tencent-cospeech/2d_kp_generation_fixed/dataset/miniset_40min/motions_vis'
filter_motion_folder_normal_vis = '/home/yangxu/tencent-cospeech/2d_kp_generation_fixed/dataset/miniset_40min/motions_normalized_vis'


filter_list = []
for motion_file in os.listdir(motion_folder):
    if motion_file.endswith('.npy'):
        motion_file = os.path.join(motion_folder, motion_file)
        data = np.load(motion_file)
        # print velocity of nose in each frame
        nose_coords = data[:, 0, :2]
        nose_velocity = np.diff(nose_coords, axis=0)
        # get the distance between two points
        nose_velocity_norm = np.linalg.norm(nose_velocity, axis=1)

        # change the data type to .3f
        nose_velocity_norm = np.around(nose_velocity_norm, 0)
        
        # if there is a frame with velocity > 20, print the file name and its velocity

        if np.max(nose_velocity_norm) > 40 and np.argmax(nose_velocity_norm) >= 10 and np.argmax(nose_velocity_norm) < len(nose_velocity_norm) - 10:
            print(np.max(nose_velocity_norm), motion_file)
            filename = motion_file.split('/')[-1].split('.')[0]
            frame_start = filename.split('_')[-2]
            frame_end = filename.split('_')[-1]
            frame_start_new = int(frame_start) + 10
            frame_end_new = int(frame_end) - 10
            new_filename = "_".join(filename.split('_')[:-2]) + f"_{frame_start_new}_{frame_end_new}"
            filter_list.append(new_filename)

# rm the file in the filter_list
for file_name in filter_list:
    os.remove(os.path.join(filter_motion_folder, file_name + '.npy'))
    os.remove(os.path.join(filter_motion_folder_normal, file_name + '.npy'))
    os.remove(os.path.join(filter_motion_folder_meta, file_name + '.json'))
    os.remove(os.path.join(filter_motion_folder_audios, file_name + '.wav'))
    os.remove(os.path.join(filter_motion_folder_vis, file_name + '.mp4'))
    os.remove(os.path.join(filter_motion_folder_normal_vis, file_name + '.mp4'))
    print(f"remove {file_name}")
        