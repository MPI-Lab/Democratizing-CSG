import logging
from pathlib import Path
import torch
from requests.packages import target
from torchvision.io import write_video
from diffusers.utils.torch_utils import is_compiled_module
import inspect
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize
import accelerate

logger = logging.getLogger(__name__)

def decode_latents(
    vae,
    latents, 
    num_frames, 
    decode_chunk_size=8):
    # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
    latents = latents.flatten(0, 1)

    latents = 1 / vae.config.scaling_factor * latents

    forward_vae_fn = vae._orig_mod.forward if is_compiled_module(vae) else vae.forward
    accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())

    # decode decode_chunk_size frames at a time to avoid OOM
    frames = []
    for i in range(0, latents.shape[0], decode_chunk_size):
        num_frames_in = latents[i: i + decode_chunk_size].shape[0]
        decode_kwargs = {}
        if accepts_num_frames:
            # we only pass num_frames_in if it's expected
            decode_kwargs["num_frames"] = num_frames_in

        frame = vae.decode(latents[i: i + decode_chunk_size], **decode_kwargs).sample
        frames.append(frame.cpu())
    frames = torch.cat(frames, dim=0)

    # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
    frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    frames = frames.float()
    return frames

def tensor2vid(video, processor, output_type="np"):
    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    if output_type == "np":
        outputs = np.stack(outputs)

    elif output_type == "pt":
        outputs = torch.stack(outputs)

    elif not output_type == "pil":
        # raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil]")
        return outputs

    return outputs


def get_aligned_face(face_loss_helper, frames, device):
    print("This is the process of getting aligned faces")
    print("Please check the number of detected faces")
    print(frames.shape)
    print(1/0)
    face_loss_helper.clean_all()
    faces = mtcnn.align(frames)
    transfroms = Compose(
        [ToTensor(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    return transfroms(faces).to(device)


def save_to_mp4(frames, save_path, fps=7):
    frames = frames.permute((0, 2, 3, 1))  # (f, c, h, w) to (f, h, w, c)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    write_video(save_path, frames, fps=fps)

def faceid_loss_compute(vae, latents, target_images, num_frames, decode_chunk_size, image_processor, device, face_loss_model=None, face_loss_helper=None):

    print("--------------------------")
    while True:
        x = 1 + 1
    frames = decode_latents(vae, latents, num_frames, decode_chunk_size)
    frames = tensor2vid(frames, image_processor, output_type="np")
    print("This is faceid loss computation")
    print(frames.shape)
    print(type(target_images))
    print(target_images.size())
    print(1/0)

    pred_faces = get_aligned_face(face_loss_helper, frames, device)
    target_faces = get_aligned_face(face_loss_helper, target_images, device)

    pred_embed = face_loss_model(pred_faces)[0]
    target_embed = face_loss_model(target_faces)[0]

    face_loss = pred_embed.dot(target_embed).item()
    face_loss = face_loss.mean()
    return face_loss