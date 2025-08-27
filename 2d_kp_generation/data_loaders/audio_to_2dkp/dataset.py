from abc import abstractmethod
from copy import deepcopy
import math
import librosa
import soundfile as sf
import random
import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
from tqdm import tqdm
import json
from data_loaders.tensors import collate_tensors, lengths_to_mask
from transformers import AutoProcessor, Wav2Vec2Model
from data_loaders.dataset_utils import global_to_local, local_to_global
# proxies = {
#     'http': '127.0.0.1:7890',
#     'https': '127.0.0.1:7890',
# }

def collate(batch):
    filenames, filenames_reference, filepaths, filepaths_reference, motions, masks, lengths, conditions = zip(*batch)
    collate_motions = collate_tensors([torch.as_tensor(motion) for motion in motions])
    collate_lengths = torch.as_tensor([length for length in lengths])
    collate_masks = lengths_to_mask(collate_lengths, collate_motions.shape[1]).unsqueeze(-1).unsqueeze(-1).expand(collate_motions.shape)

    collate_motions = collate_motions.permute(0, 2, 3, 1)  # [bs, seq_len, n_joints, n_feats] -> [bs, n_joints, n_feats, seq_len]
    collate_masks = collate_masks.permute(0, 2, 3, 1)  # [bs, seq_len, n_joints] -> [bs, n_joints, 1, seq_len]

    if conditions[0] != {}:
        new_conditions = {}
        keys = conditions[0].keys()
        for key in keys:
            if key == 'long_video_condition':
                new_conditions[key] = {}
                for sub_key in conditions[0][key].keys():
                    if sub_key == 'motion':
                        new_conditions[key][sub_key] = collate_tensors([torch.as_tensor(condition[key][sub_key]) for condition in conditions]).permute(0, 1, 3, 4, 2)
                    if sub_key == 'lengths':
                        new_conditions[key][sub_key] = torch.as_tensor([condition[key][sub_key] for condition in conditions])
                    if sub_key == 'mask':
                        motion_long = new_conditions[key]['motion']
                        full_mask = lengths_to_mask(new_conditions[key]['lengths'], motion_long.shape[-1]*motion_long.shape[1]).reshape(-1, motion_long.shape[1], motion_long.shape[-1])
                        full_mask = full_mask.unsqueeze(-2).unsqueeze(-2).expand(motion_long.shape)
                        masks = collate_tensors([torch.as_tensor(condition[key][sub_key]) for condition in conditions]).unsqueeze(-1).permute(0, 1, 3, 4, 2)
                        new_conditions[key][sub_key] = masks * full_mask
                    if sub_key == 'audio':
                        new_conditions[key][sub_key] = collate_tensors([condition[key][sub_key].float() for condition in conditions])
                    if sub_key == 'audio_attention':
                        new_conditions[key][sub_key] = collate_tensors([condition[key][sub_key] for condition in conditions])
                    if sub_key == 'initial_kp':
                        initial_kp_tensor = collate_tensors([torch.as_tensor(condition[key][sub_key]) for condition in conditions])
                        new_conditions[key][sub_key] = initial_kp_tensor.unsqueeze(1).permute(0, 1, 3, 4, 2)
                    if sub_key == 'initial_kp_ref':
                        initial_kp_ref_tensor = collate_tensors([torch.as_tensor(condition[key][sub_key]) for condition in conditions])
                        new_conditions[key][sub_key] = initial_kp_ref_tensor.unsqueeze(1).permute(0, 1, 3, 4, 2)
                    if sub_key == 'initial_frames':
                        new_conditions[key][sub_key] = collate_tensors([torch.as_tensor(condition[key][sub_key]) for condition in conditions]).permute(0, 2, 3, 1)
                    if sub_key == 'initial_frames_ref':
                        new_conditions[key][sub_key] = collate_tensors([torch.as_tensor(condition[key][sub_key]) for condition in conditions]).permute(0, 2, 3, 1)
                    if sub_key == 'start_frame':
                        new_conditions[key][sub_key] = [condition[key][sub_key] for condition in conditions]
                    if sub_key == 'start_frame_initial':
                        new_conditions[key][sub_key] = [condition[key][sub_key] for condition in conditions]
                    if sub_key == 'start_frame_initial_ref':
                        new_conditions[key][sub_key] = [condition[key][sub_key] for condition in conditions]
                    
            if key == 'audio':
                new_conditions[key] = collate_tensors([condition[key].float() for condition in conditions])
            if key == 'audio_attention':
                new_conditions[key] = collate_tensors([condition[key] for condition in conditions])
            if key == 'initial_kp':
                initial_kp_tensor = collate_tensors([torch.as_tensor(condition[key]) for condition in conditions])
                new_conditions[key] = initial_kp_tensor.unsqueeze(1).permute(0, 2, 3, 1)
            if key == 'initial_kp_ref':
                initial_kp_ref_tensor = collate_tensors([torch.as_tensor(condition[key]) for condition in conditions])
                new_conditions[key] = initial_kp_ref_tensor.unsqueeze(1).permute(0, 2, 3, 1)
            # if key == "reference_kp":
            #     new_conditions[key] = collate_tensors([torch.as_tensor(condition[key]) for condition in conditions]).unsqueeze(1).permute(0, 2, 3, 1)
            if key == "initial_frames":
                new_conditions[key] = collate_tensors([torch.as_tensor(condition[key]) for condition in conditions]).permute(0, 2, 3, 1)
            if key == "initial_frames_ref":
                new_conditions[key] = collate_tensors([torch.as_tensor(condition[key]) for condition in conditions]).permute(0, 2, 3, 1)
            if key == 'start_frame':
                new_conditions[key] = [condition[key] for condition in conditions]
            if key == 'start_frame_initial':
                new_conditions[key] = [condition[key] for condition in conditions]
            if key == 'start_frame_initial_ref':
                new_conditions[key] = [condition[key] for condition in conditions]
            if key == 'description':
                pass
            if key == 'transcription':
                pass
        conditions = new_conditions
    
    return collate_motions, {"y": {"lengths": collate_lengths, "mask": collate_masks, 
                                   "filenames": filenames, "filenames_reference":filenames_reference, 
                                   "filepaths": filepaths, "filepaths_reference":filepaths_reference, "conditions": conditions}}


def collate_with_mask(batch):
    filenames, filenames_reference, filepaths, filepaths_reference, motions, masks, lengths, conditions = zip(*batch)
    motion, model_kwargs = collate(batch)
    # The original mask is based on the confidence level of the pose estimator.
    # The new mask is a combination of the original mask and the mask based on the length of the sequence
    masks = collate_tensors([torch.as_tensor(mask) for mask in masks]).unsqueeze(-1).permute(0, 2, 3, 1)  # [bs, seq_len, n_joints] -> [bs, n_joints, 1, seq_len]
    model_kwargs["y"]["mask"] = model_kwargs["y"]["mask"] * masks
    return motion, model_kwargs


def normalize_motion(motion, mean, std):
    return (motion - mean) / std


class Audio_to_2dkp_Dataset(data.Dataset):
    def __init__(self, datapath, mean_path, std_path, skeleton_links, skeleton_links_colors, data_size=None, pose_folder="pose_smooth_normalized_mouth_width_0.075_local",
                 sequence_length=375, split="train", fps=25, confidence_threshold=0.7, cond='no_cond', use_z_score=True, use_local_kp=False, initial_frames_num=0,
                 face_scale=1, mouth_scale=1, hands_scale=1, pose_mode="full body", random_start=False, data_augmentations=[], 
                 face_start=23, mouth_start=71, left_hand_start=91, right_hand_start=112, left_wrist=9, right_wrist=10, **kwargs):
        # TODO add train, test, validation split
        self.data_path = datapath
        self.mean_path = mean_path
        self.std_path = std_path
        self.data_size = data_size
        self.pose_folder = pose_folder
        self.use_z_score = use_z_score
        self.use_local_kp = use_local_kp
        self.face_scale = face_scale
        self.mouth_scale = mouth_scale
        self.hands_scale = hands_scale
        self.split = split
        self.fps = fps
        self.condition = cond
        self.confidence_threshold = confidence_threshold
        self.sequence_length = sequence_length
        self.skeleton_links = skeleton_links
        self.skeleton_links_colors = skeleton_links_colors
        self.random_start = random_start
        self.pose_mode = pose_mode
        self.data_augmentations = data_augmentations
        self.initial_frames_num = initial_frames_num

        self.face_start = face_start
        self.mouth_start = mouth_start
        self.left_hand_start = left_hand_start
        self.right_hand_start = right_hand_start
        self.left_wrist = left_wrist
        self.right_wrist = right_wrist

        if 'audio' in self.condition:
            # self.aud_p = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h", proxies=proxies)
            self.aud_p = AutoProcessor.from_pretrained("./pretrained_weights/wav2vec2-base-960h")
    
        if self.use_z_score:
            self.mean = np.load(mean_path)
            self.std = np.load(std_path)
        
            if self.use_local_kp:
                self.mean = self.mean[:,1:,:]
                self.std = self.std[:,1:,:]

        self.file_names = []
        self.file_name_path = os.path.join(datapath, self.split + ".txt")
        with open(self.file_name_path, "r") as f:
            for line in f:
                self.file_names.append(line.strip())


        self.motions_files = []
        self.motion_ref_files = []
        self.audios_files = []
        self.descriptions_files = []
        self.transcriptions_files = []

        for file_pair in self.file_names:
            if len(file_pair.split()) == 2:
                file, ref_file = file_pair.split()
            else:
                file, ref_file = file_pair, file_pair
            self.motions_files.append(os.path.join(datapath, self.pose_folder, file+".npy"))
            self.motion_ref_files.append(os.path.join(datapath, self.pose_folder, ref_file+".npy"))
            if self.condition != "no_cond":
                if 'audio' in self.condition:
                    self.audios_files.append(os.path.join(datapath, "wav", file+".wav"))

                if 'description' in self.condition:
                    self.descriptions_files.append(os.path.join(datapath, "description", file+".txt"))
                if 'transcription' in self.condition:
                    self.transcriptions_files.append(os.path.join(datapath, "transcription", file+".txt"))

        self.length = len(self.motions_files)



    def __getitem__(self, index):
        filepath = {}
        motion_path = self.motions_files[index]
        filepath['motion'] = motion_path
        filepath['video'] = os.path.join(self.data_path, "mp4", os.path.basename(motion_path).replace(".npy", ".mp4"))
        filepath['audio'] = os.path.join(self.data_path, "wav", os.path.basename(motion_path).replace(".npy", ".wav"))
        if len(self.audios_files) > 0:
            audio_path = self.audios_files[index]
            filepath['audio'] = audio_path
        if len(self.descriptions_files) > 0:
            description_path = self.descriptions_files[index]
            filepath['description'] = description_path
        if len(self.transcriptions_files) > 0:
            transcription_path = self.transcriptions_files[index]
            filepath['transcription'] = transcription_path
        filename = os.path.basename(motion_path).replace(".npy", "")


        filepath_reference = {}
        motion_path_ref = self.motion_ref_files[index]
        filepath_reference['motion'] = motion_path_ref
        filepath_reference['video'] = os.path.join(self.data_path, "mp4", os.path.basename(motion_path_ref).replace(".npy", ".mp4"))    
        filename_reference = os.path.basename(motion_path_ref).replace(".npy", "")


        motion_w_conf = self.load_motion(motion_path)

        motion = motion_w_conf[..., :2]
        confidence = motion_w_conf[..., 2]
        mask = confidence > self.confidence_threshold

        
        motion_ref_w_conf = self.load_motion(motion_path_ref)

        motion_ref = motion_ref_w_conf[..., :2]
        confidence_ref = motion_ref_w_conf[..., 2]
        mask_ref = confidence_ref > self.confidence_threshold



        if self.split == "test" and "long_video" in self.condition:
            long_video_condition = {}
            # save motion
            iteration = math.ceil(len(motion) / self.sequence_length)
            start_frame_long = 0
            length_long = len(motion)
            long_video_condition['start_frame'] = start_frame_long
            long_video_condition['start_frame_initial'] = start_frame_long
            long_video_condition['start_frame_initial_ref'] = start_frame_long
            long_video_condition['lengths'] = length_long

            total_sequence_length_long = self.sequence_length * iteration
            motion_long = motion.copy()
            mask_long = mask.copy()
            if len(motion_long) < total_sequence_length_long:
                pad_long = np.zeros((total_sequence_length_long - len(motion_long), motion_long.shape[1], motion_long.shape[2]))
                motion_long = np.concatenate([motion_long, pad_long], axis=0)
                mask_long = np.concatenate([mask_long, np.zeros((total_sequence_length_long - len(mask_long), mask_long.shape[1]))], axis=0)
            motion_long = motion_long.reshape(-1, self.sequence_length, motion_long.shape[1], motion_long.shape[2])
            mask_long = mask_long.reshape(-1, self.sequence_length, mask_long.shape[1])
            long_video_condition['motion'] = motion_long
            long_video_condition['mask'] = mask_long
            # save audio
            if 'audio' in self.condition:
                speech_array_long, sampling_rate = librosa.load(audio_path, sr=16000)
                max_audio_length_long = int(sampling_rate / self.fps * total_sequence_length_long)
                
                speech_array_long = speech_array_long[:max_audio_length_long]
                input_values_long = np.squeeze(self.aud_p(speech_array_long,sampling_rate=16000).input_values)
                audio_attention_long = np.ones_like(input_values_long)

                if len(input_values_long) < max_audio_length_long:
                    pad_long = np.zeros((max_audio_length_long - len(input_values_long)))
                    input_values_long = np.concatenate([input_values_long, pad_long], axis=0)
                    audio_attention_long = np.concatenate([audio_attention_long, pad_long], axis=0)

                input_values_long = input_values_long.reshape(iteration, -1)
                audio_attention_long = audio_attention_long.reshape(iteration, -1)
                long_video_condition['audio'] = torch.as_tensor(input_values_long)
                long_video_condition['audio_attention'] = torch.as_tensor(audio_attention_long)
            if 'initial_kp' in self.condition:
                initial_kp_long = motion_long[:, 0] * mask_long[:, 0][..., np.newaxis]
                long_video_condition['initial_kp'] = initial_kp_long
                
                intial_kp_long_ref = motion_ref[0] * mask_ref[0][..., np.newaxis]
                long_video_condition['initial_kp_ref'] = intial_kp_long_ref
            
            if 'long_video' in self.condition:
                motion_frames_long = motion_long[0, :1] * mask_long[0, :1][..., np.newaxis]
                motion_frames_long = np.repeat(motion_frames_long, self.initial_frames_num, 0)
                long_video_condition['initial_frames'] = motion_frames_long

                motion_frames_long_ref = motion_ref[:1] * mask_ref[:1][..., np.newaxis]
                motion_frames_long_ref = np.repeat(motion_frames_long_ref, self.initial_frames_num, 0)
                long_video_condition['initial_frames_ref'] = motion_frames_long_ref


        total_sequence_length = self.sequence_length + self.initial_frames_num
        if len(motion) <= total_sequence_length:
            start_frame = 0 + self.initial_frames_num
            length = len(motion) - self.initial_frames_num
            
            pad = np.zeros((total_sequence_length - len(motion), motion.shape[1], motion.shape[2]))
            motion = np.concatenate([motion, pad], axis=0)
            mask = np.concatenate([mask, np.zeros((total_sequence_length - len(mask), mask.shape[1]))], axis=0)

        else:
            if self.random_start:
                select_frame = random.randint(0, len(motion) - total_sequence_length)
            else:
                select_frame = 0
            start_frame = select_frame + self.initial_frames_num
            length = self.sequence_length

            motion = motion[select_frame:select_frame+total_sequence_length]
            mask = mask[select_frame:select_frame+total_sequence_length]



        # # pad the motion to the sequence length
        # if len(motion) <= self.sequence_length:
        #     start_frame = 0
        #     length = len(motion)
        #     pad = np.zeros((self.sequence_length - len(motion), motion.shape[1], motion.shape[2]))
        #     motion = np.concatenate([motion, pad], axis=0)
        #     mask = np.concatenate([mask, np.zeros((self.sequence_length - len(mask), mask.shape[1]))], axis=0)
        # else:
        #     if self.random_start:
        #         start_frame = random.randint(0, len(motion) - self.sequence_length)
        #     else:
        #         start_frame = 0
        #     motion = motion[start_frame:start_frame+self.sequence_length]
        #     mask = mask[start_frame:start_frame+self.sequence_length]
        #     length = self.sequence_length


        condition = {}
        # TODO add condition handling
        if self.condition != "no_cond":
            if 'audio' in self.condition:
                speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
                # speech_array, sampling_rate = sf.read(audio_path)
                # if sampling_rate != 16000:
                #     speech_array = librosa.resample(speech_array.T, orig_sr=sampling_rate, target_sr=16000).T
                #     sampling_rate = 16000
                # # 如果是多声道，将其转换为单声道
                # if speech_array.ndim > 1:
                #     speech_array = np.mean(speech_array, axis=1)
                    
                max_audio_length = int(sampling_rate / self.fps * self.sequence_length)
                # if speech_array.shape[0] > max_audio_length:
                #     if self.random_start:
                #         start_bit = start_frame * sampling_rate // self.fps
                #     else:
                #         start_bit = 0
                #     speech_array = speech_array[start_bit:start_bit+max_audio_length]
                start_bit = start_frame * sampling_rate // self.fps
                speech_array = speech_array[start_bit:start_bit + length * sampling_rate // self.fps]

                input_values = np.squeeze(self.aud_p(speech_array,sampling_rate=16000).input_values)
                audio_attention = np.ones_like(input_values)

                if input_values.shape[0] < max_audio_length:
                    pad = np.zeros((max_audio_length - len(input_values)))
                    input_values = np.concatenate([input_values, pad], axis=0)
                    audio_attention = np.concatenate([audio_attention, pad], axis=0)

                condition['audio'] = torch.as_tensor(input_values)
                condition['audio_attention'] = torch.as_tensor(audio_attention)
                
            if 'initial_kp' in self.condition:
                initial_kp = motion[0] * mask[0].reshape(-1, 1)
                condition['initial_kp'] = initial_kp

                intial_kp_ref = motion_ref[0] * mask_ref[0].reshape(-1, 1)
                condition['initial_kp_ref'] = intial_kp_ref

            if "long_video" in self.condition:
                motion_frames = motion[:self.initial_frames_num] * mask[:self.initial_frames_num][..., np.newaxis]
                if random.random() < 0.3:
                    motion_frames = np.repeat(motion_frames[-1:], self.initial_frames_num, 0)

                motion = motion[self.initial_frames_num:]
                mask = mask[self.initial_frames_num:]
                
                # ref_id = random.randint(0, len(motion))
                # condition['reference_kp'] = motion[ref_id] * mask[ref_id].reshape(-1, 1)
                condition['initial_frames'] = motion_frames

                motion_frames_ref = motion_ref[:self.initial_frames_num] * mask_ref[:self.initial_frames_num][..., np.newaxis]
                if random.random() < 0.3:
                    motion_frames_ref = np.repeat(motion_frames_ref[-1:], self.initial_frames_num, 0)
                
                condition['initial_frames_ref'] = motion_frames_ref

                if self.split == "test":
                    condition['long_video_condition'] = long_video_condition

            if 'description' in self.condition:
                pass
            if 'transcription' in self.condition:
                pass

            if self.random_start:
                condition['start_frame'] = start_frame
                condition['start_frame_initial'] = start_frame - self.initial_frames_num
                condition['start_frame_initial_ref'] = 0
            else:
                condition['start_frame'] = 0 + self.initial_frames_num
                condition['start_frame_initial'] = 0
                condition['start_frame_initial_ref'] = 0


        return filename, filename_reference, filepath, filepath_reference, motion, mask, length, condition

    def __len__(self):
        return self.length

    def load_motion(self, motion_path):
        motion_w_conf = np.load(motion_path)

        # TODO only support 1 person for now
        person_num = motion_w_conf.shape[1]
        motion_w_conf = motion_w_conf[:, 0, ...]

        if self.use_local_kp:
            motion_w_conf = motion_w_conf[...,1:,:]

        if self.pose_mode == "only face":
            motion_w_conf = motion_w_conf[:, 24:92, :]

        elif self.pose_mode == "no legs":
            motion_w_conf = np.concatenate([motion_w_conf[:, :13, :], motion_w_conf[:, 23:, :]], axis=1)

        if self.split == "train" and self.data_augmentations:
            if "rescale" in self.data_augmentations:
                motion_w_conf[...,:2] = self.augment_rescale(motion_w_conf[...,:2])
            if "rotate" in self.data_augmentations:
                motion_w_conf[...,:2] = self.augment_rotate(motion_w_conf[...,:2])
            if "flip" in self.data_augmentations:
                motion_w_conf[...,:2] = self.augment_flip(motion_w_conf[...,:2])

        # TODO only save 1 person's mean and std for now
        if self.use_z_score:
            motion_w_conf[...,:2] = self.normalize_motion(motion_w_conf[...,:2], self.mean, self.std)

        if self.use_local_kp:
            motion_w_conf = self.rescale_local_kp(motion_w_conf)
        
        return motion_w_conf




    def normalize_motion(self, motion, mean, std):
        return (motion - mean) / std

    def denormalize_motion(self, motion, mean, std):
        return motion * std + mean

    def augment_rescale(self,motion, scale=None):
        if scale is None:
            scale = random.uniform(0.7, 1.3)
        motion[..., :2] *= scale
        return motion

    def augment_rotate(self, motion, angle=None):
        if angle is None:
            angle = random.uniform(-1/6 * np.pi, 1/6 * np.pi)  # 随机生成一个旋转角度（弧度制）
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        # 构建旋转矩阵
        rotation_matrix = np.array([[cos_angle, -sin_angle],
                                    [sin_angle, cos_angle]])
        
        # 对所有关键点应用旋转矩阵
        original_shape = motion.shape
        motion_reshaped = motion.reshape(-1, motion.shape[-1])
        motion_reshaped[:, :2] = np.dot(motion_reshaped[:, :2], rotation_matrix.T)
        motion = motion_reshaped.reshape(original_shape)
        return motion
    
    def augment_flip(self, motion):
        if random.random() < 0.5:
            motion[..., 0] *= -1
        return motion

    def rescale_local_kp(self, motion):
            motion[..., self.face_start:self.face_start+68, :2] *= self.face_scale # 脸部放大
            motion[..., self.mouth_start:self.mouth_start+20, :2] *= (self.mouth_scale / self.face_scale) # 嘴巴放大
            motion[..., self.left_hand_start:self.left_hand_start+21, :2] *= self.hands_scale # 左手
            motion[..., self.right_hand_start:self.right_hand_start+21, :2] *= self.hands_scale # 右手
            return motion

    def inverse_rescale_local_kp(self, motion):
            motion[..., self.mouth_start:self.mouth_start+20, :2] /= (self.mouth_scale / self.face_scale)  # 嘴巴部分
            motion[..., self.face_start:self.face_start+68, :2] /= self.face_scale     # 脸部分
            motion[..., self.left_hand_start:self.left_hand_start+21, :2] /= self.hands_scale # 左手
            motion[..., self.right_hand_start:self.right_hand_start+21, :2] /= self.hands_scale # 右手
            return motion
    
