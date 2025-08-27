from copy import deepcopy
import os
import itertools
import shutil
import torch
from diffusion.gaussian_diffusion import GaussianDiffusion
from diffusion.resample import create_named_schedule_sampler
from train.trainer import Trainer
from utils.fixseed import fixseed
from utils.parser_utils import train_args
from utils import dist_utils
from data_loaders.dataset_utils import get_dataset_loader_from_args, global_to_local, local_to_global
from utils.model_utils import create_model_and_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform, get_train_platform  # The platform classes are required when evaluating the train_platform argument
import torch.distributed as dist
from sample.sampler import model_kwargs_to_device
import numpy as np
from tqdm import tqdm
from utils.motion_vis import visualize_motion_with_audio, visualize_kp, visualize_local_motion_with_audio

# MEAN_LOSS_WEIGHT = np.ones(133, dtype=np.float32)
# MEAN_LOSS_WEIGHT[0:17] = 1 # body
# MEAN_LOSS_WEIGHT[17:23] = 1 # foots
# MEAN_LOSS_WEIGHT[23:91] = 1 # face
# MEAN_LOSS_WEIGHT[91:133] = 1 # hands

# EXTRA_MOUTH_LOSS_WEIGHT = MEAN_LOSS_WEIGHT.copy()
# EXTRA_MOUTH_LOSS_WEIGHT[71:91] = 2 # mouth

# EXTRA_MOUTH_HAND_LOSS_WEIGHT = MEAN_LOSS_WEIGHT.copy()
# EXTRA_MOUTH_HAND_LOSS_WEIGHT[71:91] = 2 # mouth
# EXTRA_MOUTH_HAND_LOSS_WEIGHT[91:133] = 2 # hands

# LOSS_WEIGHT = {
#     "mean": MEAN_LOSS_WEIGHT,
#     "extra_mouth": EXTRA_MOUTH_LOSS_WEIGHT,
#     "extra_mouth_hand": EXTRA_MOUTH_HAND_LOSS_WEIGHT
# }

def sum_batched(tensor: torch.Tensor, keepdim=False):
    return tensor.sum(dim=list(range(1, len(tensor.shape))), keepdim=keepdim)


def masked_weighted_l2(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, batch_weight: torch.Tensor, kp_loss_weight: str):
    # assuming a.shape == b.shape == mask.shape == bs, J, Jdim, seqlen
    loss = (a - b) ** 2
    if kp_loss_weight == "mean":
        MEAN_LOSS_WEIGHT = np.ones(loss.shape[1], dtype=np.float32) # TODO use a.shape as representation 
        MEAN_LOSS_WEIGHT_tensor = torch.tensor(MEAN_LOSS_WEIGHT, device=a.device).float().view(1, -1, 1, 1)
        loss = loss * MEAN_LOSS_WEIGHT_tensor
    # TODO depracted
    elif kp_loss_weight == "extra_mouth": # bug -> wrong for 
        EXTRA_MOUTH_LOSS_WEIGHT = np.ones(loss.shape[1], dtype=np.float32)
        EXTRA_MOUTH_LOSS_WEIGHT[71+loss.shape[1]-133:91+loss.shape[1]-133] = 2
        EXTRA_MOUTH_LOSS_WEIGHT_tensor = torch.tensor(EXTRA_MOUTH_LOSS_WEIGHT, device=a.device).float().view(1, -1, 1, 1)
        loss = loss * EXTRA_MOUTH_LOSS_WEIGHT_tensor
    elif kp_loss_weight == "extra_mouth_hand":
        EXTRA_MOUTH_HAND_LOSS_WEIGHT = np.ones(133, dtype=np.float32)
        EXTRA_MOUTH_HAND_LOSS_WEIGHT[71+loss.shape[1]-133:91+loss.shape[1]-133] = 2
        EXTRA_MOUTH_HAND_LOSS_WEIGHT[91+loss.shape[1]-133:133+loss.shape[1]-133] = 2
        EXTRA_MOUTH_HAND_LOSS_WEIGHT_tensor = torch.tensor(EXTRA_MOUTH_HAND_LOSS_WEIGHT, device=a.device).float().view(1, -1, 1, 1)
        loss = loss * EXTRA_MOUTH_HAND_LOSS_WEIGHT_tensor

    loss = sum_batched(loss * mask) * batch_weight
    unmasked_elements = torch.clamp(sum_batched(mask), min=1)
    return loss / unmasked_elements


class DiffusionTrainer(Trainer):
    def __init__(self, diffusion: GaussianDiffusion, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diffusion = diffusion
        self.schedule_sampler_type = "uniform"
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.load_diffusion_coeff()


    def load_diffusion_coeff(self):
        self.coef1 = torch.tensor(self.diffusion.posterior_mean_coef1, device=self.device).float().view(-1, 1, 1, 1)
        self.coef2 = torch.tensor(self.diffusion.posterior_mean_coef2, device=self.device).float().view(-1, 1, 1, 1)
        self.std = torch.sqrt(torch.tensor(self.diffusion.posterior_variance, device=self.device)).float().view(-1, 1, 1, 1)
        self.num_timesteps = self.diffusion.num_timesteps

    def sample_xt(self, samples, x_t, t):
        return self.coef1[t] * samples + self.coef2[t] * x_t + self.std[t] * torch.randn_like(x_t)


    def sample(self, x_T, schedule, cond):
        x_t = x_T
        sample_progressive = {}
        for t in tqdm(schedule):
            t_batch = torch.full((x_t.shape[0],), t, dtype=torch.long, device=self.device)
            sample = self.model(x_t, t_batch, **cond)
            x_t = self.sample_xt(sample, x_t, t)
            if t % 10 == 0:
                sample_progressive[t] = x_t
        x_0 = x_t
        return x_0, sample_progressive
    
    def sample_with_cfg(self, x_T, schedule, cond, scale):
        uncond = deepcopy(cond)
        uncond['y']["use_learnable_uncond"] = True
        uncond['y']["scale"] = torch.full((x_T.shape[0],), scale, device=self.device)
        x_t = x_T
        sample_progressive = {}
        for t in tqdm(schedule):
            t_batch = torch.full((x_t.shape[0],), t, dtype=torch.long, device=self.device)
            
            sample = self.model(x_t, t_batch, **cond)
            sample_uncond = self.model(x_t, t_batch, **uncond)
            sample = sample_uncond + (uncond['y']["scale"].view(-1, 1, 1, 1) * (sample - sample_uncond)) 
            
            x_t = self.sample_xt(sample, x_t, t)
            if t % 10 == 0:
                sample_progressive[t] = x_t
        x_0 = x_t
        return x_0, sample_progressive


    def save_conditions(self, cond, save_num, save_path, prefix, combine_videos_num):
        combine_videos_num += 2 # gt videos, reference frame
        if "initial_kp" in cond["y"]["conditions"]:
            combine_videos_num += 2
        if "initial_frames" in cond["y"]["conditions"]:
            combine_videos_num += 1

        for i in range(save_num if save_num < len(cond['y']['filenames']) else len(cond['y']['filenames'])):
            filename = prefix + cond["y"]["filenames"][i] + '_' + cond["y"]["filenames_reference"][i]
            
            filepath = cond["y"]["filepaths"][i]
            filepath_ref = cond["y"]["filepaths_reference"][i]


            length = cond["y"]["lengths"][i].cpu().numpy()
            length_time = length / self.test_data.dataset.fps

            # set start_time for visualization's wav and mp4
            global_start_frame = cond["y"]["conditions"]["start_frame"][i] if "start_frame" in cond["y"]["conditions"] else 0 + self.test_data.dataset.initial_frames_num
            global_start_time = global_start_frame / self.test_data.dataset.fps

            # save video
            os.makedirs(os.path.join(save_path, "mp4"), exist_ok=True)
            # shutil.copy(filepath["video"], os.path.join(save_path, "mp4", filename + ".mp4"))
            video_cut_command = f"ffmpeg -y -i {filepath['video']} -ss {global_start_time} -t {length_time} -c:v libx264 -c:a aac {os.path.join(save_path, 'mp4', filename + '.mp4')}"
            os.system(video_cut_command)

            # save reference video
            start_frame_ref = cond["y"]["conditions"]["start_frame_initial"][i]
            start_time_ref = start_frame_ref / self.test_data.dataset.fps

            os.makedirs(os.path.join(save_path, "mp4_initial_frames"), exist_ok=True)
            # shutil.copy(filepath["video"], os.path.join(save_path, "mp4", filename + ".mp4"))
            initial_frames_num = self.test_data.dataset.initial_frames_num if self.test_data.dataset.initial_frames_num > 0 else 1
            video_cut_command = f"ffmpeg -y -i {filepath_ref['video']} -ss {start_time_ref} -t {initial_frames_num / self.test_data.dataset.fps} -c:v libx264 -c:a aac {os.path.join(save_path, 'mp4_initial_frames', filename + '.mp4')}"
            os.system(video_cut_command)

            # save audio
            os.makedirs(os.path.join(save_path, "wav"), exist_ok=True)
            # shutil.copy(filepath["audio"], os.path.join(save_path, "wav", filename + ".wav"))
            audio_cut_command = (
                f"ffmpeg -y  -ss {global_start_time} -i {filepath['audio']} -t {length_time} -c copy {os.path.join(save_path, 'wav', filename + '.wav')}"
            )
            os.system(audio_cut_command)
            
            # save description txt
            if "description" in cond["y"]["conditions"]:
                os.makedirs(os.path.join(save_path, "description"), exist_ok=True)
                shutil.copy(filepath["description"], os.path.join(save_path, "description", filename + ".txt"))

            if "initial_frames" in cond["y"]["conditions"]:
                initial_frames = cond["y"]["conditions"]["initial_frames"].cpu().numpy().transpose(0, 3, 1, 2)[i]
                if self.test_data.dataset.use_local_kp:
                    initial_frames = self.test_data.dataset.inverse_rescale_local_kp(initial_frames)
                if self.test_data.dataset.use_z_score:
                    initial_frames = np.where(initial_frames != np.array([0, 0]), self.test_data.dataset.denormalize_motion(initial_frames, self.test_data.dataset.mean, self.test_data.dataset.std), 0)
                    # initial_frames = initial_frames * self.test_data.dataset.std + self.test_data.dataset.mean
                if self.test_data.dataset.use_local_kp:
                    initial_frames = np.where(initial_frames != np.array([0, 0]), local_to_global(initial_frames, 
                                    face_start=self.test_data.dataset.face_start, mouth_start=self.test_data.dataset.mouth_start,
                                    left_hand_start=self.test_data.dataset.left_hand_start, right_hand_start=self.test_data.dataset.right_hand_start,
                                    left_wrist=self.test_data.dataset.left_wrist, right_wrist=self.test_data.dataset.right_wrist)[0], 0)
                
                os.makedirs(os.path.join(save_path, "initial_frames"), exist_ok=True)
                np.save(os.path.join(save_path, "initial_frames", filename + ".npy"), initial_frames)

                initial_frames[:, :, 1] = -initial_frames[:, :, 1]
                os.makedirs(os.path.join(save_path, "initial_frames_vis"), exist_ok=True)
                visualize_motion_with_audio(motion=initial_frames, audio_path=None, start_time=-1,
                                            title="Initial frames", filename=filename, save_path=os.path.join(save_path, "initial_frames_vis", filename + ".mp4"),
                                            skeleton_links=self.test_data.dataset.skeleton_links,
                                            skeleton_links_colors=self.test_data.dataset.skeleton_links_colors)

            # save initial kp npy
            if "initial_kp" in cond["y"]["conditions"]:
                start_frame_ref = cond["y"]["conditions"]["start_frame_initial"][i]
                start_time_ref = start_frame_ref / self.test_data.dataset.fps

                os.makedirs(os.path.join(save_path, "mp4_initial_kp"), exist_ok=True)
                video_cut_command = f"ffmpeg -y -i {filepath_ref['video']} -ss {start_time_ref} -t {1 / self.test_data.dataset.fps} -c:v libx264 -c:a aac {os.path.join(save_path, 'mp4_initial_kp', filename + '.mp4')}"


                initial_kp = cond["y"]["conditions"]["initial_kp"].cpu().numpy().transpose(0, 3, 1, 2)[i]
                
                if self.test_data.dataset.use_local_kp:
                    initial_kp = self.test_data.dataset.inverse_rescale_local_kp(initial_kp)

                if self.test_data.dataset.use_z_score:
                    initial_kp = np.where(initial_kp != np.array([0, 0]), self.test_data.dataset.denormalize_motion(initial_kp, self.test_data.dataset.mean, self.test_data.dataset.std), 0)
                    # initial_kp = initial_kp * self.test_data.dataset.std + self.test_data.dataset.mean
                
                if self.test_data.dataset.use_local_kp:
                    initial_kp = np.where(initial_kp != np.array([0, 0]), local_to_global(initial_kp, 
                                    face_start=self.test_data.dataset.face_start, mouth_start=self.test_data.dataset.mouth_start,
                                    left_hand_start=self.test_data.dataset.left_hand_start, right_hand_start=self.test_data.dataset.right_hand_start,
                                    left_wrist=self.test_data.dataset.left_wrist, right_wrist=self.test_data.dataset.right_wrist)[0], 0)
                
                # save initial kp npy
                os.makedirs(os.path.join(save_path, "initial_kp"), exist_ok=True)
                np.save(os.path.join(save_path, "initial_kp", filename + ".npy"), initial_kp)

                # save initial kp visualization
                initial_kp[:, :, 1] = -initial_kp[:, :, 1]
                os.makedirs(os.path.join(save_path, "initial_kp_vis"), exist_ok=True)
                visualize_kp(initial_kp=initial_kp, title="Initial kp", filename=filename, save_path=os.path.join(save_path, "initial_kp_vis", filename + ".jpg"),
                             skeleton_links=self.test_data.dataset.skeleton_links, skeleton_links_colors=self.test_data.dataset.skeleton_links_colors)
                
        return combine_videos_num

    def save_local_kp(self, motion, sample, sample_progressive, cond, save_num, save_path, prefix, combine_videos_num):
        combine_videos_num += 2
        for i in range(save_num if save_num < motion.shape[0] else motion.shape[0]):
            filename = prefix + cond["y"]["filenames"][i] + '_' + cond["y"]["filenames_reference"][i]
            filepath = cond["y"]["filepaths"][i]
            
            length = cond["y"]["lengths"][i].cpu().numpy()
            mask = cond["y"]["mask"][i].cpu().numpy().transpose(2, 0, 1)[:length]
            
            # set start_time for visualization's wav and mp4
            global_start_frame = cond["y"]["conditions"]["start_frame"][i] if "start_frame" in cond["y"]["conditions"] else 0
            global_start_time = global_start_frame / self.test_data.dataset.fps

            # save GT
            save_gt = motion[i][:length].copy()*mask
            vis_gt = save_gt.copy()
            vis_gt[:, :, 1] = -vis_gt[:, :, 1]
            # save GT's local kp npy
            os.makedirs(os.path.join(save_path, "gt_pose_local"), exist_ok=True)
            np.save(os.path.join(save_path, "gt_pose_local", filename + ".npy"), save_gt)
            # save GT's local kp visualization
            os.makedirs(os.path.join(save_path, "gt_pose_local_vis"), exist_ok=True)
            visualize_local_motion_with_audio(motion=vis_gt, audio_path=filepath["audio"], start_time=global_start_time,
                                        title="GT Local", filename=filename, save_path=os.path.join(save_path, "gt_pose_local_vis", filename + ".mp4"),
                                        skeleton_links=self.test_data.dataset.skeleton_links,
                                        skeleton_links_colors=self.test_data.dataset.skeleton_links_colors)
            
            # save sample
            save_sample = sample[i][:length].copy()
            vis_sample = save_sample.copy()
            vis_sample[:, :, 1] = -vis_sample[:, :, 1]
            # save sample's local kp npy
            os.makedirs(os.path.join(save_path, "sample_pose_local"), exist_ok=True)
            np.save(os.path.join(save_path, "sample_pose_local", filename + ".npy"), save_sample)
            # save sample's local kp visualization
            os.makedirs(os.path.join(save_path, "sample_pose_local_vis"), exist_ok=True)
            visualize_local_motion_with_audio(motion=vis_sample, audio_path=filepath["audio"], start_time=global_start_time,
                                        title="Sample Local", filename=filename,  save_path=os.path.join(save_path, "sample_pose_local_vis", filename + ".mp4"),
                                        skeleton_links=self.test_data.dataset.skeleton_links,
                                        skeleton_links_colors=self.test_data.dataset.skeleton_links_colors)

            if self.save_sample_progressive:
                # save sample progressive's local kp npy
                os.makedirs(os.path.join(save_path, "sample_pose_local_progressive"), exist_ok=True)
                for t in sample_progressive:
                    np.save(os.path.join(save_path, "sample_pose_local_progressive", f"{filename}_{t}steps.npy"), sample_progressive[t][i][:length])

                # save sample progressive's local kp visualization
                os.makedirs(os.path.join(save_path, "sample_pose_local_progressive_vis"), exist_ok=True)
                for t in sample_progressive:
                    vis_sample_progressive = sample_progressive[t][i][:length].copy()
                    vis_sample_progressive[:, :, 1] = -vis_sample_progressive[:, :, 1]
                    visualize_local_motion_with_audio(motion=vis_sample_progressive, audio_path=filepath["audio"], start_time=global_start_time,
                                                title=f"Sample Local {t} steps", filename=f"{filename}_{t}steps",  save_path=os.path.join(save_path, "sample_pose_local_progressive_vis", f"{filename}_{t}steps.mp4"),
                                                skeleton_links=self.test_data.dataset.skeleton_links,
                                                skeleton_links_colors=self.test_data.dataset.skeleton_links_colors)
        return combine_videos_num

    def save_global_kp(self, motion, sample, sample_progressive, cond, save_num, save_path, prefix, combine_videos_num):
        combine_videos_num += 2 # gt videos; sample videos; initial kp

        for i in range(save_num if save_num < motion.shape[0] else motion.shape[0]):
            filename = prefix + cond["y"]["filenames"][i] + '_' + cond["y"]["filenames_reference"][i]
            
            length = cond["y"]["lengths"][i].cpu().numpy()
            mask = cond["y"]["mask"][i].cpu().numpy().transpose(2, 0, 1)[:length]
            
            # set start_time for visualization's wav and mp4
            global_start_frame = cond["y"]["conditions"]["start_frame"][i] if "start_frame" in cond["y"]["conditions"] else 0
            global_start_time = global_start_frame / self.test_data.dataset.fps


            # save GT
            save_gt = motion[i][:length].copy()*mask
            vis_gt = save_gt.copy()
            vis_gt[:, :, 1] = -vis_gt[:, :, 1]
            os.makedirs(os.path.join(save_path, "gt_pose"), exist_ok=True)
            np.save(os.path.join(save_path, "gt_pose", filename + ".npy"), save_gt)
            # save GT's motion visualization
            os.makedirs(os.path.join(save_path, "gt_pose_vis"), exist_ok=True)
            visualize_motion_with_audio(motion=vis_gt, audio_path=os.path.join(save_path, "wav", filename + ".wav"), start_time=global_start_time,
                                        title="GT", filename=filename, save_path=os.path.join(save_path, "gt_pose_vis", filename + ".mp4"),
                                        skeleton_links=self.test_data.dataset.skeleton_links,
                                        skeleton_links_colors=self.test_data.dataset.skeleton_links_colors)

            # save sample
            save_sample = sample[i][:length].copy()
            vis_sample = save_sample.copy()
            vis_sample[:, :, 1] = -vis_sample[:, :, 1]
            # save sample's motion npy
            os.makedirs(os.path.join(save_path, "sample_pose"), exist_ok=True)
            np.save(os.path.join(save_path, "sample_pose", filename + ".npy"), save_sample)
            # save sample's motion visualization
            os.makedirs(os.path.join(save_path, "sample_pose_vis"), exist_ok=True)
            visualize_motion_with_audio(motion=vis_sample, audio_path=os.path.join(save_path, "wav", filename + ".wav"), start_time=global_start_time,
                                        title="Sample", filename=filename,  save_path=os.path.join(save_path, "sample_pose_vis", filename + ".mp4"),
                                        skeleton_links=self.test_data.dataset.skeleton_links,
                                        skeleton_links_colors=self.test_data.dataset.skeleton_links_colors)

            # save sample progressive's motion npy
            if self.save_sample_progressive:
                # save sample progressive's motion npy
                os.makedirs(os.path.join(save_path, "sample_pose_progressive"), exist_ok=True)
                for t in sample_progressive:
                    np.save(os.path.join(save_path, "sample_pose_progressive", f"{filename}_{t}steps.npy"), sample_progressive[t][i][:length])
                
                # save sample progressive's motion visualization
                os.makedirs(os.path.join(save_path, "sample_pose_progressive_vis"), exist_ok=True)
                for t in sample_progressive:
                    vis_sample_progressive = sample_progressive[t][i][:length].copy()
                    vis_sample_progressive[:, :, 1] = -vis_sample_progressive[:, :, 1]
                    visualize_motion_with_audio(motion=vis_sample_progressive, audio_path=os.path.join(save_path, "wav", filename + ".wav"), start_time=global_start_time,
                                                title=f"Sample {t} steps", filename=f"{filename}_{t}steps",  save_path=os.path.join(save_path, "sample_pose_progressive_vis", f"{filename}_{t}steps.mp4"),
                                                skeleton_links=self.test_data.dataset.skeleton_links,
                                                skeleton_links_colors=self.test_data.dataset.skeleton_links_colors)
        return combine_videos_num
    
    def save_combined_vis(self, motion, cond, save_num, save_path, prefix, combine_videos_num):
        for i in range(save_num if save_num < motion.shape[0] else motion.shape[0]):
            filename = prefix + cond["y"]["filenames"][i] + '_' + cond["y"]["filenames_reference"][i]

            # combine GT and sample visualization
            os.makedirs(os.path.join(save_path, "combined_vis"), exist_ok=True)
            combine_command = f"""ffmpeg -i {os.path.join(save_path, 'mp4', filename + '.mp4')} \
                    {"-i " + os.path.join(save_path, 'initial_kp_vis', filename + '.jpg') if "initial_kp" in cond["y"]["conditions"] else ""} \
                    {"-i " + os.path.join(save_path, 'mp4_initial_kp', filename + '.mp4') if "initial_kp" in cond["y"]["conditions"] else ""} \
                    {"-i " + os.path.join(save_path, 'initial_frames_vis', filename + '.mp4') if "initial_frames" in cond["y"]["conditions"] else ""} \
                    {"-i " + os.path.join(save_path, 'mp4_initial_frames', filename + '.mp4')} \
                    {"-i " + os.path.join(save_path, 'gt_pose_local_vis', filename + '.mp4') if self.test_data.dataset.use_local_kp else ""} \
                    -i {os.path.join(save_path, 'gt_pose_vis', filename + '.mp4')} \
                    {"-i " + os.path.join(save_path, 'sample_pose_local_vis', filename + '.mp4') if self.test_data.dataset.use_local_kp else ""} \
                    -i {os.path.join(save_path, 'sample_pose_vis', filename + '.mp4')} \
                    -filter_complex '"""

            stack_command = ""
            for i in range(combine_videos_num):
                stack_command += f"[{i}:v]scale=ceil(iw*512/ih/2)*2:512[v{i}]; "
            for i in range(combine_videos_num):
                stack_command += f"[v{i}]"
            stack_command += f"hstack={combine_videos_num}' "
            combine_command += stack_command + f"{os.path.join(save_path, 'combined_vis', filename + '.mp4')}"
            os.system(combine_command)

    def sample_and_visualize(self, motion, cond, save_num, save_path, prefix="", scale=1):
        cond = deepcopy(cond)
        cond['y']['filenames_reference'] = cond['y']['filenames']
        cond['y']['filepaths_reference'] = cond['y']['filepaths']

        x_T = torch.randn(motion.shape, device=self.device)
        schedule = range(self.num_timesteps - 1, -1, -1)

        if self.args.cond_mask_prob > 0 and scale > 1:
            sample, sample_progressive = self.sample_with_cfg(x_T, schedule, cond, scale)
        else:
            sample, sample_progressive = self.sample(x_T, schedule, cond)
        
        # turn to numpy and transpose to (batch, seqlen, J, Jdim)
        motion = motion.cpu().numpy().transpose(0, 3, 1, 2)
        sample = sample.cpu().numpy().transpose(0, 3, 1, 2)
        for t in sample_progressive:
            sample_progressive[t] = sample_progressive[t].cpu().numpy().transpose(0, 3, 1, 2)

        # rescale for local kp
        if self.test_data.dataset.use_local_kp:
            motion = self.test_data.dataset.inverse_rescale_local_kp(motion)
            sample = self.test_data.dataset.inverse_rescale_local_kp(sample)
            for t in sample_progressive:
                sample_progressive[t] = self.test_data.dataset.inverse_rescale_local_kp(sample_progressive[t])
    
        # denormalize for z-score
        if self.test_data.dataset.use_z_score:
            motion = self.test_data.dataset.denormalize_motion(motion, self.test_data.dataset.mean, self.test_data.dataset.std)
            sample = self.test_data.dataset.denormalize_motion(sample, self.test_data.dataset.mean, self.test_data.dataset.std)
            for t in sample_progressive:
                sample_progressive[t] = self.test_data.dataset.denormalize_motion(sample_progressive[t], self.test_data.dataset.mean, self.test_data.dataset.std)


        combine_videos_num = 0

        # save conditions
        combine_videos_num = self.save_conditions(cond, save_num, save_path, prefix, combine_videos_num)

        if self.test_data.dataset.use_local_kp:
            # save local kp
            combine_videos_num = self.save_local_kp(motion, sample, sample_progressive, cond, save_num, save_path, prefix, combine_videos_num)

            # local to global
            motion = local_to_global(motion, face_start=self.test_data.dataset.face_start, mouth_start=self.test_data.dataset.mouth_start,
                                    left_hand_start=self.test_data.dataset.left_hand_start, right_hand_start=self.test_data.dataset.right_hand_start,
                                    left_wrist=self.test_data.dataset.left_wrist, right_wrist=self.test_data.dataset.right_wrist)
            sample = local_to_global(sample, face_start=self.test_data.dataset.face_start, mouth_start=self.test_data.dataset.mouth_start,
                                    left_hand_start=self.test_data.dataset.left_hand_start, right_hand_start=self.test_data.dataset.right_hand_start,
                                    left_wrist=self.test_data.dataset.left_wrist, right_wrist=self.test_data.dataset.right_wrist)
            for t in sample_progressive:
                sample_progressive[t] = local_to_global(sample_progressive[t], face_start=self.test_data.dataset.face_start, mouth_start=self.test_data.dataset.mouth_start,
                                    left_hand_start=self.test_data.dataset.left_hand_start, right_hand_start=self.test_data.dataset.right_hand_start,
                                    left_wrist=self.test_data.dataset.left_wrist, right_wrist=self.test_data.dataset.right_wrist)

        # save global kp
        combine_videos_num = self.save_global_kp(motion, sample, sample_progressive, cond, save_num, save_path, prefix, combine_videos_num)

        # save combined visualization
        self.save_combined_vis(motion, cond, save_num, save_path, prefix, combine_videos_num)

    def sample_and_visualize_unpaired(self, motion, cond, save_num, save_path, prefix="", scale=1):
        # # TODO: to be fixed
        # def derangement(n):
        #     while True:
        #         perm = np.random.permutation(n)
        #         if np.all(perm != np.arange(n)):
        #             return perm
        # random_order = derangement(motion.shape[0])
        # 调整 filenames 的顺序为原来位置的filename+'_'+新位置的filename
        cond_unpaired = deepcopy(cond)

        # cond_unpaired["y"]["filenames_reference"] = [cond_unpaired["y"]["filenames_reference"][i] for i in random_order]
        # cond_unpaired["y"]["filepaths_reference"] = [cond_unpaired["y"]["filepaths_reference"][i] for i in random_order]
        cond_unpaired["y"]["conditions"]["start_frame_initial"] = cond_unpaired["y"]["conditions"]["start_frame_initial_ref"]

        if "initial_frames" in cond_unpaired["y"]["conditions"]:
            cond_unpaired["y"]["conditions"]["initial_frames"] = cond_unpaired["y"]["conditions"]["initial_frames_ref"]

        if "initial_kp" in cond_unpaired["y"]["conditions"]:
            cond_unpaired["y"]["conditions"]["initial_kp"] = cond_unpaired["y"]["conditions"]["initial_kp_ref"]


        x_T = torch.randn(motion.shape, device=self.device)
        schedule = range(self.num_timesteps - 1, -1, -1)

        if self.args.cond_mask_prob > 0 and scale > 1:
            sample, sample_progressive = self.sample_with_cfg(x_T, schedule, cond_unpaired, scale)
        else:
            sample, sample_progressive = self.sample(x_T, schedule, cond_unpaired)
        
        # turn to numpy and transpose to (batch, seqlen, J, Jdim)
        motion = motion.cpu().numpy().transpose(0, 3, 1, 2)
        sample = sample.cpu().numpy().transpose(0, 3, 1, 2)
        for t in sample_progressive:
            sample_progressive[t] = sample_progressive[t].cpu().numpy().transpose(0, 3, 1, 2)

        # rescale for local kp
        if self.test_data.dataset.use_local_kp:
            motion = self.test_data.dataset.inverse_rescale_local_kp(motion)
            sample = self.test_data.dataset.inverse_rescale_local_kp(sample)
            for t in sample_progressive:
                sample_progressive[t] = self.test_data.dataset.inverse_rescale_local_kp(sample_progressive[t])
    
        # denormalize for z-score
        if self.test_data.dataset.use_z_score:
            motion = self.test_data.dataset.denormalize_motion(motion, self.test_data.dataset.mean, self.test_data.dataset.std)
            sample = self.test_data.dataset.denormalize_motion(sample, self.test_data.dataset.mean, self.test_data.dataset.std)
            for t in sample_progressive:
                sample_progressive[t] = self.test_data.dataset.denormalize_motion(sample_progressive[t], self.test_data.dataset.mean, self.test_data.dataset.std)


        combine_videos_num = 0

        # save conditions
        combine_videos_num = self.save_conditions(cond_unpaired, save_num, save_path, prefix, combine_videos_num)

        if self.test_data.dataset.use_local_kp:
            # save local kp
            combine_videos_num = self.save_local_kp(motion, sample, sample_progressive, cond_unpaired, save_num, save_path, prefix, combine_videos_num)

            # local to global
            motion = local_to_global(motion, face_start=self.test_data.dataset.face_start, mouth_start=self.test_data.dataset.mouth_start,
                                    left_hand_start=self.test_data.dataset.left_hand_start, right_hand_start=self.test_data.dataset.right_hand_start,
                                    left_wrist=self.test_data.dataset.left_wrist, right_wrist=self.test_data.dataset.right_wrist)
            sample = local_to_global(sample, face_start=self.test_data.dataset.face_start, mouth_start=self.test_data.dataset.mouth_start,
                                    left_hand_start=self.test_data.dataset.left_hand_start, right_hand_start=self.test_data.dataset.right_hand_start,
                                    left_wrist=self.test_data.dataset.left_wrist, right_wrist=self.test_data.dataset.right_wrist)
            for t in sample_progressive:
                sample_progressive[t] = local_to_global(sample_progressive[t], face_start=self.test_data.dataset.face_start, mouth_start=self.test_data.dataset.mouth_start,
                                    left_hand_start=self.test_data.dataset.left_hand_start, right_hand_start=self.test_data.dataset.right_hand_start,
                                    left_wrist=self.test_data.dataset.left_wrist, right_wrist=self.test_data.dataset.right_wrist)

        # save global kp
        combine_videos_num = self.save_global_kp(motion, sample, sample_progressive, cond_unpaired, save_num, save_path, prefix, combine_videos_num)

        # save combined visualization
        self.save_combined_vis(motion, cond_unpaired, save_num, save_path, prefix, combine_videos_num)

    def sample_and_visualize_long(self, motion, cond, save_num, save_path, prefix="", scale=1):
        cond_long = deepcopy(cond)
        long_video_condition = deepcopy(cond['y']['conditions']['long_video_condition'])

        cond_long['y']['filenames_reference'] = cond_long['y']['filenames']
        cond_long['y']['filepaths_reference'] = cond_long['y']['filepaths']

        cond_long['y']['conditions']['start_frame'] = long_video_condition['start_frame']
        cond_long['y']['conditions']['start_frame_initial'] = long_video_condition['start_frame_initial']
        
        iteration = long_video_condition['motion'].shape[1]
        seq_frames = long_video_condition['motion'].shape[-1]

        motion_long = []
        sample_long = []
        for i in range(iteration):
            motion = long_video_condition['motion'][:,i]
            cond_long['y']['lengths'] = torch.where(long_video_condition['lengths']<seq_frames*(i+1), torch.clamp(long_video_condition['lengths']-seq_frames*i, min=0), seq_frames)
            cond_long['y']['mask'] = long_video_condition['mask'][:,i]
            cond_long['y']['conditions']['audio'] = long_video_condition['audio'][:,i]
            cond_long['y']['conditions']['audio_attention'] = long_video_condition['audio_attention'][:,i]
            if i == 0:
                cond_long['y']['conditions']['initial_frames'] = long_video_condition['initial_frames']
            else:
                cond_long['y']['conditions']['initial_frames'] = sample_long[-1][...,-long_video_condition['initial_frames'].shape[-1]:]

            x_T = torch.randn(motion.shape, device=self.device)
            schedule = range(self.num_timesteps - 1, -1, -1)
            if self.args.cond_mask_prob > 0 and scale > 1:
                sample, _ = self.sample_with_cfg(x_T, schedule, cond_long, scale)
            else:
                sample, _ = self.sample(x_T, schedule, cond_long)
            motion_long.append(motion)
            sample_long.append(sample)
        
        motion_long = torch.concat(motion_long, dim=-1)
        sample_long = torch.concat(sample_long, dim=-1)
        cond_long['y']['lengths'] = long_video_condition['lengths']
        cond_long['y']['mask'] = torch.concat([long_video_condition['mask'][:,i] for i in range(iteration)], dim=-1)
        cond_long['y']['conditions']['initial_frames'] = long_video_condition['initial_frames']

        # turn to numpy and transpose to (batch, seqlen, J, Jdim)
        motion_long = motion_long.cpu().numpy().transpose(0, 3, 1, 2)
        sample_long = sample_long.cpu().numpy().transpose(0, 3, 1, 2)

        # rescale for local kp
        if self.test_data.dataset.use_local_kp:
            motion_long = self.test_data.dataset.inverse_rescale_local_kp(motion_long)
            sample_long = self.test_data.dataset.inverse_rescale_local_kp(sample_long)

        # denormalize for z-score
        if self.test_data.dataset.use_z_score:
            motion_long = self.test_data.dataset.denormalize_motion(motion_long, self.test_data.dataset.mean, self.test_data.dataset.std)
            sample_long = self.test_data.dataset.denormalize_motion(sample_long, self.test_data.dataset.mean, self.test_data.dataset.std)

        combine_videos_num = 0

        # save conditions
        combine_videos_num = self.save_conditions(cond_long, save_num, save_path, prefix, combine_videos_num)

        if self.test_data.dataset.use_local_kp:
            # save local kp
            combine_videos_num = self.save_local_kp(motion_long, sample_long, None, cond_long, save_num, save_path, prefix, combine_videos_num)

            # local to global
            motion_long = local_to_global(motion_long, face_start=self.test_data.dataset.face_start, mouth_start=self.test_data.dataset.mouth_start,
                                    left_hand_start=self.test_data.dataset.left_hand_start, right_hand_start=self.test_data.dataset.right_hand_start,
                                    left_wrist=self.test_data.dataset.left_wrist, right_wrist=self.test_data.dataset.right_wrist)
            sample_long = local_to_global(sample_long, face_start=self.test_data.dataset.face_start, mouth_start=self.test_data.dataset.mouth_start,
                                    left_hand_start=self.test_data.dataset.left_hand_start, right_hand_start=self.test_data.dataset.right_hand_start,
                                    left_wrist=self.test_data.dataset.left_wrist, right_wrist=self.test_data.dataset.right_wrist)

        # save global kp
        combine_videos_num = self.save_global_kp(motion_long, sample_long, None, cond_long, save_num, save_path, prefix, combine_videos_num)

        # save combined visualization
        self.save_combined_vis(motion_long, cond_long, save_num, save_path, prefix, combine_videos_num)

    def sample_and_visualize_long_unpaired(self, motion, cond, save_num, save_path, prefix="", scale=1):
        cond_long = deepcopy(cond)
        long_video_condition = deepcopy(cond['y']['conditions']['long_video_condition'])

        cond_long['y']['conditions']['start_frame'] = long_video_condition['start_frame']
        cond_long['y']['conditions']['start_frame_initial'] = long_video_condition['start_frame_initial_ref']
        
        iteration = long_video_condition['motion'].shape[1]
        seq_frames = long_video_condition['motion'].shape[-1]

        motion_long = []
        sample_long = []
        for i in range(iteration):
            motion = long_video_condition['motion'][:,i]
            cond_long['y']['lengths'] = torch.where(long_video_condition['lengths']<seq_frames*(i+1), torch.clamp(long_video_condition['lengths']-seq_frames*i, min=0), seq_frames)
            cond_long['y']['mask'] = long_video_condition['mask'][:,i]
            cond_long['y']['conditions']['audio'] = long_video_condition['audio'][:,i]
            cond_long['y']['conditions']['audio_attention'] = long_video_condition['audio_attention'][:,i]
            if i == 0:
                cond_long['y']['conditions']['initial_frames'] = long_video_condition['initial_frames_ref']
            else:
                cond_long['y']['conditions']['initial_frames'] = sample_long[-1][...,-long_video_condition['initial_frames_ref'].shape[-1]:]

            x_T = torch.randn(motion.shape, device=self.device)
            schedule = range(self.num_timesteps - 1, -1, -1)
            if self.args.cond_mask_prob > 0 and scale > 1:
                sample, _ = self.sample_with_cfg(x_T, schedule, cond_long, scale)
            else:
                sample, _ = self.sample(x_T, schedule, cond_long)
            motion_long.append(motion)
            sample_long.append(sample)
        
        motion_long = torch.concat(motion_long, dim=-1)
        sample_long = torch.concat(sample_long, dim=-1)
        cond_long['y']['lengths'] = long_video_condition['lengths']
        cond_long['y']['mask'] = torch.concat([long_video_condition['mask'][:,i] for i in range(iteration)], dim=-1)
        cond_long['y']['conditions']['initial_frames'] = long_video_condition['initial_frames_ref']

        # turn to numpy and transpose to (batch, seqlen, J, Jdim)
        motion_long = motion_long.cpu().numpy().transpose(0, 3, 1, 2)
        sample_long = sample_long.cpu().numpy().transpose(0, 3, 1, 2)

        # rescale for local kp
        if self.test_data.dataset.use_local_kp:
            motion_long = self.test_data.dataset.inverse_rescale_local_kp(motion_long)
            sample_long = self.test_data.dataset.inverse_rescale_local_kp(sample_long)

        # denormalize for z-score
        if self.test_data.dataset.use_z_score:
            motion_long = self.test_data.dataset.denormalize_motion(motion_long, self.test_data.dataset.mean, self.test_data.dataset.std)
            sample_long = self.test_data.dataset.denormalize_motion(sample_long, self.test_data.dataset.mean, self.test_data.dataset.std)

        combine_videos_num = 0

        # save conditions
        combine_videos_num = self.save_conditions(cond_long, save_num, save_path, prefix, combine_videos_num)

        if self.test_data.dataset.use_local_kp:
            # save local kp
            combine_videos_num = self.save_local_kp(motion_long, sample_long, None, cond_long, save_num, save_path, prefix, combine_videos_num)

            # local to global
            motion_long = local_to_global(motion_long, face_start=self.test_data.dataset.face_start, mouth_start=self.test_data.dataset.mouth_start,
                                    left_hand_start=self.test_data.dataset.left_hand_start, right_hand_start=self.test_data.dataset.right_hand_start,
                                    left_wrist=self.test_data.dataset.left_wrist, right_wrist=self.test_data.dataset.right_wrist)
            sample_long = local_to_global(sample_long, face_start=self.test_data.dataset.face_start, mouth_start=self.test_data.dataset.mouth_start,
                                    left_hand_start=self.test_data.dataset.left_hand_start, right_hand_start=self.test_data.dataset.right_hand_start,
                                    left_wrist=self.test_data.dataset.left_wrist, right_wrist=self.test_data.dataset.right_wrist)

        # save global kp
        combine_videos_num = self.save_global_kp(motion_long, sample_long, None, cond_long, save_num, save_path, prefix, combine_videos_num)

        # save combined visualization
        self.save_combined_vis(motion_long, cond_long, save_num, save_path, prefix, combine_videos_num)


    def sample_and_visualize_old(self, motion, cond, save_num, save_path, prefix=""):
        x_t = torch.randn(motion.shape, device=self.device)
        schedule = range(self.num_timesteps - 1, -1, -1)
        sample_progressive = {}
        for t in tqdm(schedule):
            t_batch = torch.full((motion.shape[0],), t, dtype=torch.long, device=self.device)
            sample = self.model(x_t, t_batch, **cond)
            x_t = self.sample_xt(sample, x_t, t)
            if t % 10 == 0:
                sample_progressive[t] = x_t.cpu().numpy().transpose(0, 3, 1, 2)
        motion = motion.cpu().numpy().transpose(0, 3, 1, 2)
        sample = x_t.cpu().numpy().transpose(0, 3, 1, 2)

        if self.test_data.dataset.use_local_kp:
            motion = self.test_data.dataset.inverse_rescale_local_kp(motion)
            sample = self.test_data.dataset.inverse_rescale_local_kp(sample)
            if self.save_sample_progressive:
                for t in sample_progressive:
                    sample_progressive[t] = self.test_data.dataset.inverse_rescale_local_kp(sample_progressive[t])
        
        if self.test_data.dataset.use_z_score:
            motion = motion * self.test_data.dataset.std + self.test_data.dataset.mean
            sample = sample * self.test_data.dataset.std + self.test_data.dataset.mean
            if self.save_sample_progressive:
                for t in sample_progressive:
                    sample_progressive[t] = sample_progressive[t] * self.test_data.dataset.std + self.test_data.dataset.mean

        videos_num_global = 0

        if self.test_data.dataset.use_local_kp:
            for i in range(save_num if save_num < motion.shape[0] else motion.shape[0]):
                filename = cond["y"]["filenames"][i]
                filename = prefix + filename
                filepath = cond["y"]["filepaths"][i]
                start_frame = cond["y"]["conditions"]["start_frame"][i] if "start_frame" in cond["y"]["conditions"] else 0
                start_time = start_frame / self.test_data.dataset.fps
                length = cond["y"]["lengths"][i].cpu().numpy()
                mask = cond["y"]["mask"][i].cpu().numpy().transpose(2, 0, 1)[:length]
                # save GT's local kp npy
                os.makedirs(os.path.join(save_path, "gt_pose_local"), exist_ok=True)
                np.save(os.path.join(save_path, "gt_pose_local", filename + ".npy"), motion[i][:length]*mask)

                # save GT's local kp visualization
                os.makedirs(os.path.join(save_path, "gt_pose_local_vis"), exist_ok=True)
                vis_gt = motion[i][:length]*mask
                vis_gt[:, :, 1] = -vis_gt[:, :, 1]
                visualize_local_motion_with_audio(motion=vis_gt, audio_path=filepath["audio"], start_time=start_time,
                                            title="GT Local", filename=filename, save_path=os.path.join(save_path, "gt_pose_local_vis", filename + ".mp4"),
                                            skeleton_links=self.test_data.dataset.skeleton_links,
                                            skeleton_links_colors=self.test_data.dataset.skeleton_links_colors)

                # save sample's local kp npy
                os.makedirs(os.path.join(save_path, "sample_pose_local"), exist_ok=True)
                np.save(os.path.join(save_path, "sample_pose_local", filename + ".npy"), sample[i][:length])
                
                # save sample's local kp visualization
                os.makedirs(os.path.join(save_path, "sample_pose_local_vis"), exist_ok=True)
                vis_sample = sample[i][:length].copy()
                vis_sample[:, :, 1] = -vis_sample[:, :, 1]
                visualize_local_motion_with_audio(motion=vis_sample, audio_path=filepath["audio"], start_time=start_time,
                                            title="Sample Local", filename=filename,  save_path=os.path.join(save_path, "sample_pose_local_vis", filename + ".mp4"),
                                            skeleton_links=self.test_data.dataset.skeleton_links,
                                            skeleton_links_colors=self.test_data.dataset.skeleton_links_colors)

                if self.save_sample_progressive:
                    # save sample progressive's local kp npy
                    os.makedirs(os.path.join(save_path, "sample_pose_local_progressive"), exist_ok=True)
                    for t in sample_progressive:
                        np.save(os.path.join(save_path, "sample_pose_local_progressive", f"{filename}_{t}steps.npy"), sample_progressive[t][i][:length])

                    # save sample progressive's local kp visualization
                    os.makedirs(os.path.join(save_path, "sample_pose_local_progressive_vis"), exist_ok=True)
                    for t in sample_progressive:
                        vis_sample_progressive = sample_progressive[t][i][:length].copy()
                        vis_sample_progressive[:, :, 1] = -vis_sample_progressive[:, :, 1]
                        visualize_local_motion_with_audio(motion=vis_sample_progressive, audio_path=filepath["audio"], start_time=start_time,
                                                    title=f"Sample Local {t} steps", filename=f"{filename}_{t}steps",  save_path=os.path.join(save_path, "sample_pose_local_progressive_vis", f"{filename}_{t}steps.mp4"),
                                                    skeleton_links=self.test_data.dataset.skeleton_links,
                                                    skeleton_links_colors=self.test_data.dataset.skeleton_links_colors)
            videos_num_global += 2

            motion = local_to_global(motion, face_start=self.test_data.dataset.face_start, mouth_start=self.test_data.dataset.mouth_start,
                                    left_hand_start=self.test_data.dataset.left_hand_start, right_hand_start=self.test_data.dataset.right_hand_start,
                                    left_wrist=self.test_data.dataset.left_wrist, right_wrist=self.test_data.dataset.right_wrist,
                                     face_scale=1, mouth_scale=1)
            sample = local_to_global(sample, face_start=self.test_data.dataset.face_start, mouth_start=self.test_data.dataset.mouth_start,
                                    left_hand_start=self.test_data.dataset.left_hand_start, right_hand_start=self.test_data.dataset.right_hand_start,
                                    left_wrist=self.test_data.dataset.left_wrist, right_wrist=self.test_data.dataset.right_wrist,
                                     face_scale=1, mouth_scale=1)
            
            if self.save_sample_progressive:    
                for t in sample_progressive:
                    sample_progressive[t] = local_to_global(sample_progressive[t], face_start=self.test_data.dataset.face_start, mouth_start=self.test_data.dataset.mouth_start,
                                        left_hand_start=self.test_data.dataset.left_hand_start, right_hand_start=self.test_data.dataset.right_hand_start,
                                        left_wrist=self.test_data.dataset.left_wrist, right_wrist=self.test_data.dataset.right_wrist,
                                        face_scale=1, mouth_scale=1)

        videos_num_global += 3 # gt videos; sample videos; initial kp
        # print("save_validated_results")
        for i in range(save_num if save_num < motion.shape[0] else motion.shape[0]):
            filename = cond["y"]["filenames"][i]
            filename = prefix + filename
            filepath = cond["y"]["filepaths"][i]
            start_frame = cond["y"]["conditions"]["start_frame"][i] if "start_frame" in cond["y"]["conditions"] else 0
            start_time = start_frame / self.test_data.dataset.fps
            length = cond["y"]["lengths"][i].cpu().numpy()
            length_time = length / self.test_data.dataset.fps
            mask = cond["y"]["mask"][i].cpu().numpy().transpose(2, 0, 1)[:length]
            # if self.test_data.dataset.use_local_kp:
            #     mask = mask[:,1:,:]

            videos_num = videos_num_global

            # save GT's motion npy
            os.makedirs(os.path.join(save_path, "gt_pose"), exist_ok=True)
            np.save(os.path.join(save_path, "gt_pose", filename + ".npy"), motion[i][:length]*mask)

            # save video
            os.makedirs(os.path.join(save_path, "mp4"), exist_ok=True)
            # shutil.copy(filepath["video"], os.path.join(save_path, "mp4", filename + ".mp4"))
            video_cut_command = f"ffmpeg -y -ss {start_time} -i {filepath['video']} -t {length_time} -c:v libx264 -c:a aac {os.path.join(save_path, 'mp4', filename + '.mp4')}"
            os.system(video_cut_command)

            # save audio
            os.makedirs(os.path.join(save_path, "wav"), exist_ok=True)
            # shutil.copy(filepath["audio"], os.path.join(save_path, "wav", filename + ".wav"))
            audio_cut_command = (
                f"ffmpeg -y  -ss {start_time} -i {filepath['audio']} -t {length_time} -c copy {os.path.join(save_path, 'wav', filename + '.wav')}"
            )
            os.system(audio_cut_command)
            

            # save description txt
            if "description" in cond["y"]["conditions"]:
                os.makedirs(os.path.join(save_path, "description"), exist_ok=True)
                shutil.copy(filepath["description"], os.path.join(save_path, "description", filename + ".txt"))

            # save initial kp npy
            if "initial_kp" in cond["y"]["conditions"]:
                initial_kp = cond["y"]["conditions"]["initial_kp"].cpu().numpy().transpose(0, 3, 1, 2)[i]
                
                if self.test_data.dataset.use_local_kp:
                    initial_kp = self.test_data.dataset.inverse_rescale_local_kp(initial_kp)

                if self.test_data.dataset.use_z_score:
                    initial_kp = np.where(initial_kp != np.array([0, 0]), initial_kp * self.test_data.dataset.std + self.test_data.dataset.mean, 0)
                    # initial_kp = initial_kp * self.test_data.dataset.std + self.test_data.dataset.mean
                
                if self.test_data.dataset.use_local_kp:
                    initial_kp = np.where(initial_kp[:,:,:] != np.array([0, 0]), local_to_global(initial_kp, 
                                    face_start=self.test_data.dataset.face_start, mouth_start=self.test_data.dataset.mouth_start,
                                    left_hand_start=self.test_data.dataset.left_hand_start, right_hand_start=self.test_data.dataset.right_hand_start,
                                    left_wrist=self.test_data.dataset.left_wrist, right_wrist=self.test_data.dataset.right_wrist, face_scale=1, mouth_scale=1)[0], 0)


                
                initial_kp[:, :, 1] = -initial_kp[:, :, 1]

                os.makedirs(os.path.join(save_path, "initial_kp"), exist_ok=True)
                np.save(os.path.join(save_path, "initial_kp", filename + ".npy"), initial_kp)
                # save initial kp visualization
                os.makedirs(os.path.join(save_path, "initial_kp_vis"), exist_ok=True)
                visualize_kp(initial_kp=initial_kp, title="Initial kp", filename=filename, save_path=os.path.join(save_path, "initial_kp_vis", filename + ".jpg"),
                             skeleton_links=self.test_data.dataset.skeleton_links, skeleton_links_colors=self.test_data.dataset.skeleton_links_colors)
                videos_num += 1

            # save sample's motion npy
            os.makedirs(os.path.join(save_path, "sample_pose"), exist_ok=True)
            np.save(os.path.join(save_path, "sample_pose", filename + ".npy"), sample[i][:length])

            if self.save_sample_progressive:
                # save sample progressive's motion npy
                os.makedirs(os.path.join(save_path, "sample_pose_progressive"), exist_ok=True)
                for t in sample_progressive:
                    np.save(os.path.join(save_path, "sample_pose_progressive", f"{filename}_{t}steps.npy"), sample_progressive[t][i][:length])

            # save GT's motion visualization
            os.makedirs(os.path.join(save_path, "gt_pose_vis"), exist_ok=True)
            vis_gt = motion[i][:length]*mask
            vis_gt[:, :, 1] = -vis_gt[:, :, 1]
            visualize_motion_with_audio(motion=vis_gt, audio_path=os.path.join(save_path, "wav", filename + ".wav"), start_time=start_time,
                                        title="GT", filename=filename, save_path=os.path.join(save_path, "gt_pose_vis", filename + ".mp4"),
                                        skeleton_links=self.test_data.dataset.skeleton_links,
                                        skeleton_links_colors=self.test_data.dataset.skeleton_links_colors)

            # save sample's motion visualization
            os.makedirs(os.path.join(save_path, "sample_pose_vis"), exist_ok=True)
            vis_sample = sample[i][:length].copy()
            vis_sample[:, :, 1] = -vis_sample[:, :, 1]
            visualize_motion_with_audio(motion=vis_sample, audio_path=os.path.join(save_path, "wav", filename + ".wav"), start_time=start_time,
                                        title="Sample", filename=filename,  save_path=os.path.join(save_path, "sample_pose_vis", filename + ".mp4"),
                                        skeleton_links=self.test_data.dataset.skeleton_links,
                                        skeleton_links_colors=self.test_data.dataset.skeleton_links_colors)
            
            # save sample progressive's motion visualization
            if self.save_sample_progressive:
                os.makedirs(os.path.join(save_path, "sample_pose_progressive_vis"), exist_ok=True)
                for t in sample_progressive:
                    vis_sample_progressive = sample_progressive[t][i][:length].copy()
                    vis_sample_progressive[:, :, 1] = -vis_sample_progressive[:, :, 1]
                    visualize_motion_with_audio(motion=vis_sample_progressive, audio_path=os.path.join(save_path, "wav", filename + ".wav"), start_time=start_time,
                                                title=f"Sample {t} steps", filename=f"{filename}_{t}steps",  save_path=os.path.join(save_path, "sample_pose_progressive_vis", f"{filename}_{t}steps.mp4"),
                                                skeleton_links=self.test_data.dataset.skeleton_links,
                                                skeleton_links_colors=self.test_data.dataset.skeleton_links_colors)
            
            # combine GT and sample visualization
            os.makedirs(os.path.join(save_path, "combined_vis"), exist_ok=True)
            combine_command = f"""ffmpeg -i {os.path.join(save_path, 'mp4', filename + '.mp4')} \
                    {"-i " + os.path.join(save_path, 'initial_kp_vis', filename + '.jpg') if "initial_kp" in cond["y"]["conditions"] else ""} \
                    {"-i " + os.path.join(save_path, 'gt_pose_local_vis', filename + '.mp4') if self.test_data.dataset.use_local_kp else ""} \
                    -i {os.path.join(save_path, 'gt_pose_vis', filename + '.mp4')} \
                    {"-i " + os.path.join(save_path, 'sample_pose_local_vis', filename + '.mp4') if self.test_data.dataset.use_local_kp else ""} \
                    -i {os.path.join(save_path, 'sample_pose_vis', filename + '.mp4')} \
                    -filter_complex '"""

            stack_command = ""
            for i in range(videos_num):
                stack_command += f"[{i}:v]scale=ceil(iw*512/ih/2)*2:512[v{i}]; "
            for i in range(videos_num):
                stack_command += f"[v{i}]"
            stack_command += f"hstack={videos_num}' "
            combine_command += stack_command + f"{os.path.join(save_path, 'combined_vis', filename + '.mp4')}"
            os.system(combine_command)

    def calculate_loss(self, motion: torch.Tensor, cond):
        t, batch_weight = self.schedule_sampler.sample(motion.shape[0], self.device)
        noise = torch.randn_like(motion)
        x_t = self.diffusion.q_sample(motion, t, noise)
        model_output = self.model(x=x_t, timesteps=t, **cond)
        loss = masked_weighted_l2(motion, model_output, cond["y"]["mask"], batch_weight, self.kp_loss_weight)
        loss_mean = loss.mean()
        # if loss_mean > 0.3:
        #     print(f"Loss too large: {loss_mean}", "step:", t, "mean_step:", t.float().mean())
        # if loss_mean < 0.05:
        #     print(f"Loss too small: {loss_mean}", "step:", t, "mean_step:", t.float().mean())
        return loss_mean


class DiffusionTrainerEval(DiffusionTrainer):
    def __init__(self, *args, evaluator_args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluator_args = evaluator_args

    def inverse_transform(self, motion):
            return motion * self.data_std.unsqueeze(-1) + self.data_mean.unsqueeze(-1)

    def load_std_mean(self):
        self.data_mean = torch.tensor(self.data.dataset.mean, device=self.device, dtype=torch.float32)
        self.data_std = torch.tensor(self.data.dataset.std, device=self.device, dtype=torch.float32)


def main():
    args, evaluator_args = train_args()
    fixseed(args.seed)
    dist_utils.setup_dist(args.device, args.distributed)

    if os.path.exists(args.save_dir) and not args.overwrite_model:
        raise FileExistsError(f"Save dir [{args.save_dir}] already exists.")
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print("Loading training data...")
    train_data = get_dataset_loader_from_args(args, batch_size=args.train_batch_size, split="train")

    print("Loading test data...")
    test_data = get_dataset_loader_from_args(args, batch_size=args.train_batch_size, split="test")

    print("Creating model...")

    model, diffusion = create_model_and_diffusion(args)
    model.to(dist_utils.dev())
    # torch.cuda.empty_cache()

    print("Training...")
    trainer = DiffusionTrainerEval(diffusion, args, model, train_data, test_data, get_train_platform(args), evaluator_args = evaluator_args)
    trainer.train()


if __name__ == "__main__":
    main()
