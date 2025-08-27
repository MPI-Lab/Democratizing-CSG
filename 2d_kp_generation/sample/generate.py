import torch
import numpy as np
from tqdm import tqdm
from sample.sampler import DiffusionSampler, model_kwargs_to_device
from data_loaders.dataset_utils import get_dataset_loader_from_args, global_to_local, local_to_global
from utils.fixseed import fixseed
from utils.parser_utils import generate_args
from utils.motion_vis import visualize_motion_with_audio, visualize_kp
import shutil
import random
import os

class Generator(DiffusionSampler):
    def task_name(self):
        return "sample"

    @torch.no_grad()
    def __call__(self, save=True, visualize=True, model_kwargs=None, progress=True):
        if self.args.use_data:
            self.setup_dir()
            with torch.no_grad():
                for motion, cond in tqdm(self.data):
                    motion = motion.to(self.device)
                    cond = model_kwargs_to_device(cond, self.device)
                    self.sample_and_visualize(motion, cond, save_num=self.args.batch_size, save_path=self.args.output_dir)
        else:
            if save or visualize:
                self.setup_dir()
                # original_length = self.args.motion_length
                if self.args.motion_length == "random":
                    self.args.motion_length = random.randint(5, 15)

                self.load_model_kwargs(model_kwargs)

                x_t = self.sample_initial_xt()
                schedule = range(self.num_timesteps - 1, 0, -1)
                for t in tqdm(schedule) if progress else schedule:
                    t_batch = torch.full((self.shape[0],), t, dtype=torch.long, device=self.device)
                    sample = self.model(x_t, t_batch, **self.model_kwargs)
                    x_t = self.sample_xt(sample, x_t, t)

                sample = self.inverse_transform(sample)
                if save:
                    self.save_motions(motions=sample, title=f"result")
                if visualize:
                    self.visualize(sample, f"result")

    def sample_and_visualize(self, motion, cond, save_num, save_path, prefix=""):
        x_t = torch.randn(motion.shape, device=self.device)
        schedule = range(self.num_timesteps - 1, 0, -1)
        for t in tqdm(schedule):
            t_batch = torch.full((motion.shape[0],), t, dtype=torch.long, device=self.device)
            sample = self.model(x_t, t_batch, **cond)
            x_t = self.sample_xt(sample, x_t, t)
        motion = motion.cpu().numpy().transpose(0, 3, 1, 2)
        sample = x_t.cpu().numpy().transpose(0, 3, 1, 2)

        if self.data.dataset.use_local_kp:
            motion[:, :, 72:92, :2] /= (self.data.dataset.mouth_scale / self.data.dataset.face_scale)  # 嘴巴部分
            motion[:, :, 24:92, :2] = motion[:, :, 24:92, :2] / self.data.dataset.face_scale     # 脸部分
            motion[:, :, 93:113, :2] /= self.data.dataset.hands_scale # 左手
            motion[:, :, 114:134, :2] /= self.data.dataset.hands_scale # 右手

            sample[:, :, 72:92, :2] /= (self.data.dataset.mouth_scale / self.data.dataset.face_scale)  # 嘴巴部分
            sample[:, :, 24:92, :2] = sample[:, :, 24:92, :2] / self.data.dataset.face_scale     # 脸部分
            sample[:, :, 93:113, :2] /= self.data.dataset.hands_scale # 左手
            sample[:, :, 114:134, :2] /= self.data.dataset.hands_scale # 右手

        if self.data.dataset.use_z_score:
            motion = motion * self.data.dataset.std + self.data.dataset.mean
            sample = sample * self.data.dataset.std + self.data.dataset.mean

        if self.data.dataset.use_local_kp:
            motion = local_to_global(motion, face_scale=1, mouth_scale=1)
            sample = local_to_global(sample, face_scale=1, mouth_scale=1)


        # print("save_validated_results")
        for i in range(save_num if save_num < motion.shape[0] else motion.shape[0]):
            filename = cond["y"]["filenames"][i]
            filename = prefix + filename
            filepath = cond["y"]["filepaths"][i]

            length = cond["y"]["lengths"][i].cpu().numpy()
            mask = cond["y"]["mask"][i].cpu().numpy().transpose(2, 0, 1)[:length]
            if self.data.dataset.use_local_kp:
                mask = mask[:,1:,:]
            videos_num = 3

            # save GT's motion npy
            os.makedirs(os.path.join(save_path, "gt_pose"), exist_ok=True)
            np.save(os.path.join(save_path, "gt_pose", filename + ".npy"), motion[i][:length]*mask)

            # save video
            os.makedirs(os.path.join(save_path, "mp4"), exist_ok=True)
            shutil.copy(filepath["audio"].replace("/wav/", "/mp4/").replace(".wav", ".mp4"), os.path.join(save_path, "mp4", filename + ".mp4"))

            # save audio
            os.makedirs(os.path.join(save_path, "wav"), exist_ok=True)
            shutil.copy(filepath["audio"], os.path.join(save_path, "wav", filename + ".wav"))

            # save initial kp npy
            if "initial_kp" in cond["y"]["conditions"]:
                initial_kp = cond["y"]["conditions"]["initial_kp"].cpu().numpy().transpose(0, 3, 1, 2)[i]
                
                if self.data.dataset.use_local_kp:
                    initial_kp[:, 72:92, :2] /= (self.data.dataset.mouth_scale / self.data.dataset.face_scale)  # 嘴巴部分
                    initial_kp[:, 24:92, :2] = initial_kp[:, 24:92, :2] / self.data.dataset.face_scale     # 脸部分
                    initial_kp[:, 93:113, :2] /= self.data.dataset.hands_scale # 左手
                    initial_kp[:, 114:134, :2] /= self.data.dataset.hands_scale # 右手

                if self.data.dataset.use_z_score:
                    initial_kp = np.where(initial_kp != np.array([0, 0]), initial_kp * self.data.dataset.std + self.data.dataset.mean, 0)
                    # initial_kp = initial_kp * self.data.dataset.std + self.data.dataset.mean
                
                if self.data.dataset.use_local_kp:
                    initial_kp = np.where(initial_kp[:,1:,:] != np.array([0, 0]), local_to_global(initial_kp, face_scale=1, mouth_scale=1)[0], 0)

                
                
                initial_kp[:, :, 1] = -initial_kp[:, :, 1]

                os.makedirs(os.path.join(save_path, "initial_kp"), exist_ok=True)
                np.save(os.path.join(save_path, "initial_kp", filename + ".npy"), initial_kp)
                # save initial kp visualization
                os.makedirs(os.path.join(save_path, "initial_kp_vis"), exist_ok=True)
                visualize_kp(initial_kp=initial_kp, title="Initial kp", filename=filename, save_path=os.path.join(save_path, "initial_kp_vis", filename + ".jpg"),
                                skeleton_links=self.data.dataset.skeleton_links, skeleton_links_colors=self.data.dataset.skeleton_links_colors)
                videos_num += 1

            # save sample's motion npy
            os.makedirs(os.path.join(save_path, "sample_pose"), exist_ok=True)
            np.save(os.path.join(save_path, "sample_pose", filename + ".npy"), sample[i][:length])


            # save GT's motion visualization
            os.makedirs(os.path.join(save_path, "gt_pose_vis"), exist_ok=True)
            vis_gt = motion[i][:length]*mask
            vis_gt[:, :, 1] = -vis_gt[:, :, 1]
            visualize_motion_with_audio(motion=vis_gt, audio_path=os.path.join(save_path, "wav", filename + ".wav"),
                                        title="GT", filename=filename, save_path=os.path.join(save_path, "gt_pose_vis", filename + ".mp4"),
                                        skeleton_links=self.data.dataset.skeleton_links,
                                        skeleton_links_colors=self.data.dataset.skeleton_links_colors)

            # save sample's motion visualization
            os.makedirs(os.path.join(save_path, "sample_pose_vis"), exist_ok=True)
            vis_sample = sample[i][:length]
            vis_sample[:, :, 1] = -vis_sample[:, :, 1]
            visualize_motion_with_audio(motion=vis_sample, audio_path=os.path.join(save_path, "wav", filename + ".wav"),
                                        title="Sample", filename=filename,  save_path=os.path.join(save_path, "sample_pose_vis", filename + ".mp4"),
                                        skeleton_links=self.data.dataset.skeleton_links,
                                        skeleton_links_colors=self.data.dataset.skeleton_links_colors)
            
            # combine GT and sample visualization
            os.makedirs(os.path.join(save_path, "combined_vis"), exist_ok=True)
            combine_command = f"""ffmpeg -i {os.path.join(save_path, 'mp4', filename + '.mp4')} \
                    {"-i " + os.path.join(save_path, 'initial_kp_vis', filename + '.jpg') if "initial_kp" in cond["y"]["conditions"] else ""} \
                    -i {os.path.join(save_path, 'gt_pose_vis', filename + '.mp4')} \
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




def main():
    args = generate_args()
    fixseed(args.seed)
    generator = Generator(args)
    generator(visualize=False)


if __name__ == "__main__":
    main()
