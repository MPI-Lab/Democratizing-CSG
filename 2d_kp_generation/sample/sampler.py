from copy import deepcopy
import shutil
from data_loaders.dataset_utils import get_dataset_config
import os
import numpy as np
import torch
from utils.model_utils import create_model_and_diffusion, load_model_wo_clip
from utils import dist_utils
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.dataset_utils import get_dataset_loader_from_args
from utils.plot_script import plot_motion
from utils.parser_utils import generate_args
from datetime import datetime
import librosa
from transformers import AutoProcessor, Wav2Vec2Model
# proxies = {
#     'http': '127.0.0.1:7890',
#     'https': '127.0.0.1:7890',
# }

def model_kwargs_to_device(model_kwargs, device):
    for key in model_kwargs["y"]:
        if key == "conditions":
            for condition_key in model_kwargs["y"]["conditions"]:
                if condition_key == "long_video_condition":
                    for long_video_key in model_kwargs["y"]["conditions"][condition_key]:
                        if type(model_kwargs["y"]["conditions"][condition_key][long_video_key]) == torch.Tensor:
                            model_kwargs["y"]["conditions"][condition_key][long_video_key] = model_kwargs["y"]["conditions"][condition_key][long_video_key].to(device)

                if type(model_kwargs["y"]["conditions"][condition_key]) == torch.Tensor:
                    model_kwargs["y"]["conditions"][condition_key] = model_kwargs["y"]["conditions"][condition_key].to(device)
            continue
        if type(model_kwargs["y"][key]) == torch.Tensor:
            model_kwargs["y"][key] = model_kwargs["y"][key].to(device)
    return model_kwargs


def model_kwargs_to_numpy(model_kwargs):
    for key in model_kwargs["y"]:
        if type(model_kwargs["y"][key]) == torch.Tensor:
            model_kwargs["y"][key] = model_kwargs["y"][key].cpu().numpy()
    return model_kwargs

def normalize_motion(motion, mean, std):
    return (motion - mean) / std

def get_title(model_kwargs, sample_i):
    try:
        return model_kwargs["y"]["text"][sample_i]
    except:
        return ""


class Sampler:
    def __init__(self, args):
        self.args = deepcopy(args)
        self.setup_device()
        self.load_dataset_properties()
        self.load_model()

    def get_args(self):
        return generate_args()

    def setup_device(self):
        dist_utils.setup_dist(self.args.device)
        self.device = dist_utils.dev()

    def load_model(self):
        pass  # To be overridden

    def task_name(self):
        return ""

    def setup_dir(self):
        current_time = datetime.now().strftime("%Y_%m%d_%H_%M_%S")
        dir_name = f"{self.task_name()}_seed_{self.args.seed}_{current_time}_{os.path.basename(self.args.model_path)[:-4]}"
    
        if hasattr(self.args, "text_prompt") and self.args.text_prompt is not None:
            dir_name += f"""_{self.args.text_prompt.replace(' ', '_')}"""
        if self.args.output_dir == "":
            self.args.output_dir = os.path.dirname(self.args.model_path)
        self.args.output_dir = os.path.join(self.args.output_dir, dir_name)

        if os.path.exists(self.args.output_dir):
            assert self.args.overwrite, f"Output directory [{self.args.output_dir}] already exists."
            shutil.rmtree(self.args.output_dir)
        os.makedirs(self.args.output_dir, exist_ok=True)

    def visualize(self, sample, title="", normalized=False, model_kwargs=None, use_mask=False):
        if normalized:
            sample = self.inverse_transform(sample)
        model_kwargs = self.model_kwargs if model_kwargs is None else model_kwargs
        sample = sample.cpu().numpy()
        for sample_i in range(sample.shape[0]):
            length = model_kwargs["y"]["lengths"][sample_i].item()
            mask = model_kwargs["y"]["mask"][sample_i].cpu().numpy().transpose(2, 0, 1)[:length] if use_mask else None
            motion = sample[sample_i].transpose(2, 0, 1)[:length]
            save_path = os.path.join(self.args.output_dir, f"{title}_{sample_i}")
            plot_motion(save_path, motion, dataset=self.args.dataset, title=get_title(model_kwargs, sample_i), mask=mask, fps=self.fps)

    def save_motions(self, motions: torch.Tensor, model_kwargs=None, title="results"):
        model_kwargs = self.model_kwargs if model_kwargs is None else model_kwargs
        save_path = os.path.join(self.args.output_dir, f"{title}.npy")
        results = {"motions": motions.detach().cpu().numpy(), "model_kwargs": model_kwargs_to_numpy(deepcopy(model_kwargs))}
        np.save(save_path, results)
        print(f"Saved motions to [{os.path.abspath(save_path)}]")

    def load_model_kwargs(self, model_kwargs=None):
        if model_kwargs is not None:
            self.model_kwargs = model_kwargs
        else:
            if self.args.use_data:
                self.load_model_kwargs_from_data()
            else:
                self.model_kwargs = {"y": {}}

            if self.args.text_prompt is not None:
                self.model_kwargs["y"]["text"] = [self.args.text_prompt] * self.args.num_samples

            if self.args.audio_folder is not None:
                aud_p = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h", proxies=proxies)
                audio_paths = [os.path.join(self.args.audio_folder, i) for i in os.listdir(self.args.audio_folder)]
                self.args.num_samples = len(audio_paths)
                self.args.batch_size = len(audio_paths)
                self.args.data_size = len(audio_paths)
                max_audio_length = int(16000 / self.fps * self.sequence_length)
                length = []
                audio_tensor = []
                audio_attention_tensor = []
                for audio_path in audio_paths:
                    speech_array, _ = librosa.load(audio_path, sr=16000)
                    if speech_array.shape[0] > max_audio_length:
                        speech_array = speech_array[:max_audio_length]
                    length.append(speech_array.shape[0] / 16000 * self.fps)
                    input_values = np.squeeze(aud_p(speech_array,sampling_rate=16000).input_values)
                    audio_attention = np.ones_like(input_values)
                    if input_values.shape[0] < max_audio_length:
                        pad = np.zeros((max_audio_length - len(input_values)))
                        input_values = np.concatenate([input_values, pad], axis=0)
                        audio_attention = np.concatenate([audio_attention, pad], axis=0)
                    audio_tensor.append(input_values)
                    audio_attention_tensor.append(audio_attention)
                self.model_kwargs["y"]["conditions"] = {"audio": torch.as_tensor(audio_tensor, device=self.device).float(), "audio_attention": torch.as_tensor(audio_attention_tensor, device=self.device)}
                self.model_kwargs["y"]["lengths"] = torch.as_tensor(length, device=self.device, dtype=torch.long)
                self.model_kwargs["y"]["filenames"] = [os.path.basename(i).replace(".wav", "") for i in audio_paths]
            
            
            if self.args.kp_folder is not None:
                kp_paths = [os.path.join(self.args.kp_folder, f"{i}.npy") for i in self.model_kwargs["y"]["filenames"]]
                initial_kp_tensor = []
                for motion_path in kp_paths:
                    motion_w_conf = np.load(motion_path)
                    person_num = 1
                    motion_w_conf = motion_w_conf[:, 0, ...]
                    motion = motion_w_conf[..., :2]
                    confidence = motion_w_conf[..., 2]
                    mask = confidence > self.confidence_threshold
                    if self.use_z_score:
                        motion = normalize_motion(motion, self.data_mean.cpu().numpy(), self.data_std.cpu().numpy())

                    initial_kp = motion[0] * mask[0].reshape(-1, 1)
                    initial_kp_tensor.append(initial_kp)
                self.model_kwargs["y"]["conditions"]["initial_kp"] = torch.as_tensor(initial_kp_tensor, device=self.device).unsqueeze(1).permute(0, 2, 3, 1)
                

            
            
            if self.args.motion_length is not None:
                self.model_kwargs["y"]["lengths"] = torch.full([self.args.num_samples], float(self.args.motion_length) * self.fps, device=self.device, dtype=torch.long)

            if self.args.cond != "no_cond":
                self.model_kwargs["y"]["scale"] = torch.full([self.args.num_samples], self.args.guidance_param, device=self.device)

        self.model_kwargs = model_kwargs_to_device(self.model_kwargs, self.device)
        self.n_frames = self.sequence_length
        self.shape = (self.args.batch_size, self.n_joints, self.n_feats, self.n_frames)

    def load_model_kwargs_from_data(self):
        self.load_dataset()
        self.input_motions, self.model_kwargs = next(self.data_iter)
        self.input_motions = self.input_motions.to(self.device)
        if self.args.show_input_motions:
            self.visualize(self.input_motions, title="input_motions", normalized=True, use_mask=True)

    def load_dataset(self):
        self.data = get_dataset_loader_from_args(self.args, batch_size=self.args.batch_size, split=self.args.data_split)
        self.data_iter = iter(self.data)

    def load_dataset_properties(self):
        self.data, self.data_iter = None, None
        if self.args.use_data:
            self.load_dataset()
        self.fps = get_dataset_config(self.args.dataset_config).fps
        self.sequence_length = get_dataset_config(self.args.dataset_config).sequence_length
        self.confidence_threshold = get_dataset_config(self.args.dataset_config).confidence_threshold
        # self.skeleton = get_dataset_config(self.args.dataset_config).skeleton
        # self.skeleton_names = get_dataset_config(self.args.dataset_config).landmarks
        self.n_joints, self.n_feats = get_dataset_config(self.args.dataset_config).dims
        self.use_z_score = get_dataset_config(self.args.dataset_config).use_z_score

        if self.use_z_score:
            self.data_mean = torch.tensor(np.load(get_dataset_config(self.args.dataset_config).mean_path), device=self.device, dtype=torch.float32)
            self.data_std = torch.tensor(np.load(get_dataset_config(self.args.dataset_config).std_path), device=self.device, dtype=torch.float32)

    def transform(self, motion):
        # motion = [batch_size, n_joints, n_feats, motion_length]
        return (motion - self.data_mean.unsqueeze(-1)) / self.data_std.unsqueeze(-1)

    def inverse_transform(self, motion):
        # motion = [batch_size, n_joints, n_feats, motion_length]
        if get_dataset_config(self.args.dataset_config).use_z_score:
            return motion * self.data_std.unsqueeze(-1) + self.data_mean.unsqueeze(-1)
        return motion


class DiffusionSampler(Sampler):
    def load_model(self):
        self.load_model_and_diffusion()
        self.load_diffusion_coeff()

    def load_model_and_diffusion(self):
        print("Creating model and diffusion...")
        self.model, self.diffusion = create_model_and_diffusion(self.args)

        print(f"Loading checkpoint from [{self.args.model_path}]...")
        state_dict = torch.load(self.args.model_path, map_location="cpu")
        load_model_wo_clip(self.model, state_dict["model"])

        if self.args.cond != "no_cond" and self.args.guidance_param not in [0, 1]:
            self.model = ClassifierFreeSampleModel(self.model)
        self.model.to(self.device)
        self.model.eval()

    def load_diffusion_coeff(self):
        self.coef1 = torch.tensor(self.diffusion.posterior_mean_coef1, device=self.device).float().view(-1, 1, 1, 1)
        self.coef2 = torch.tensor(self.diffusion.posterior_mean_coef2, device=self.device).float().view(-1, 1, 1, 1)
        self.std = torch.sqrt(torch.tensor(self.diffusion.posterior_variance, device=self.device)).float().view(-1, 1, 1, 1)
        self.num_timesteps = self.diffusion.num_timesteps

    def sample_xt(self, samples, x_t, t):
        return self.coef1[t] * samples + self.coef2[t] * x_t + self.std[t] * torch.randn_like(x_t)

    def sample_initial_xt(self):
        return torch.randn(self.shape, device=self.device)
