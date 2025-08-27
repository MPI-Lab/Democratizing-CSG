import os
from types import SimpleNamespace
import numpy as np
from data_loaders.audio_to_2dkp.dataset import Audio_to_2dkp_Dataset, collate_with_mask
from data_loaders.audio_to_2dkp.skeleton import SKELETON_LINKS,SKELETON_LINKS_only_face,SKELETON_LINKS_no_leg
from data_loaders.audio_to_2dkp.skeleton import SKELETON_LINK_COLORS,SKELETON_LINK_COLORS_only_face,SKELETON_LINK_COLORS_no_leg

train_from_3000000_steps = SimpleNamespace(
    **{
        "skeleton_links": SKELETON_LINKS_no_leg,
        "skeleton_links_colors": SKELETON_LINK_COLORS_no_leg,

        # body index
        "face_start": 13,
        "mouth_start": 61,
        "left_hand_start": 81,
        "right_hand_start": 102,
        "left_wrist": 9,
        "right_wrist": 10,

        # "split": "train", # set in train_mdm script
        "fps": 25,
        "dims": (123, 2),
        "sequence_length": 250,
        "initial_frames_num": 25,
        "pose_mode": "no legs",
        "random_start": True,
        "datapath": "./dataset/filtered_400h_2_13_testset",
        "distance": 7,
        "sample_elevation_angle": lambda: np.pi / 16,
        "visualization_scale": 0.75,
        "confidence_threshold": 0.7,
        "data_augmentations": [],
        "dataset_class": Audio_to_2dkp_Dataset,
        "collate_fn": collate_with_mask,
        "use_z_score": True,
        "use_local_kp": True,
        "face_scale": 1,
        "mouth_scale": 1,
        "pose_folder": "pose_rescaled_local_smooth",
        "mean_path": "./dataset/filtered_400h_2_13_testset/Mean_pose_rescaled_local_smooth_no_leg.npy",
        "std_path": "./dataset/filtered_400h_2_13_testset/Std_pose_rescaled_local_smooth_no_leg.npy",
    }
)