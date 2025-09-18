import argparse
import os
import cv2
import numpy as np
from PIL import Image
from diffusers.models.attention_processor import XFormersAttnProcessor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torch
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler

from animation.modules.attention_processor import AnimationAttnProcessor
from animation.modules.attention_processor_normalized import AnimationIDAttnNormalizedProcessor
from animation.modules.face_model import FaceModel
from animation.modules.id_encoder import FusionFaceId
from animation.modules.pose_net import PoseNet
from animation.modules.unet import UNetSpatioTemporalConditionModel
from animation.pipelines.inference_pipeline_animation import InferenceAnimationPipeline
import random

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)

def export_frames_to_video(output_video_path, video_frame_crop, fps):
    from moviepy.editor import ImageSequenceClip
    """
    Export a list of frames to a video file.

    Args:
    - video_frame_crop (list): List of frames (numpy arrays).
    - output_video_path (str): Path to save the output video file.
    - fps (int): Frames per second for the output video.
    """
    # Create a video clip from the list of frames
    clip = ImageSequenceClip(video_frame_crop, fps=fps)

    # Write the video clip to a file
    clip.write_videofile(output_video_path, codec='libx264')

def save_vid_side_by_side(batch_output, validation_control_images, output_file, fps):
    # Helper function to convert tensors to PIL images and save as GIF
    flattened_batch_output = [Image.fromarray(img) for sublist in batch_output for img in sublist]
    video_path = output_file
    final_images = []
    outputs = []
    # Helper function to concatenate images horizontally
    def get_concat_h(im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst
    for image_list in zip(validation_control_images, flattened_batch_output):
        predict_img = image_list[1].resize(image_list[0].size)
        result = get_concat_h(image_list[0], predict_img)
        final_images.append(np.array(result))
        outputs.append(np.array(predict_img))

    output_path = output_file.replace('.mp4', '_w_pose.mp4')
    export_frames_to_video(video_path, outputs, fps=fps)
    export_frames_to_video(output_path, final_images, fps=fps)

def load_images_from_folder(folder, width, height):
    images = []
    files = os.listdir(folder)
    png_files = [f for f in files if f.endswith('.png')]
    png_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    for filename in png_files:
        img = Image.open(os.path.join(folder, filename)).convert('RGB')
        img = img.resize((width, height))
        images.append(img)

    return images

def save_frames_as_png(frames, output_path):
    pil_frames = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in frames]
    num_frames = len(pil_frames)
    for i in range(num_frames):
        pil_frame = pil_frames[i]
        save_path = os.path.join(output_path, f'frame_{i}.png')
        pil_frame.save(save_path)

def save_frames_as_mp4(frames, output_mp4_path, fps):
    print("Starting saving the frames as mp4")
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'H264' for better quality
    out = cv2.VideoWriter(output_mp4_path, fourcc, fps, (width, height))
    for frame in frames:
        frame_bgr = frame if frame.shape[2] == 3 else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()


def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    duration = 1/fps*1000
    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=duration,
                       loop=0)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Diffusion XL for InstructPix2Pix."
    )

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='./checkpoints/stable-video-diffusion-img2vid-xt',
        # required=True
        help="Path to pretrained model",
    )

    parser.add_argument(
        "--validation_image",
        type=str,
        default='./reference.png',
        help=(
            "A set of paths to the controlnext conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )

    parser.add_argument(
        "--validation_control_folder",
        type=str,
        default='./pose',
        help=(
            "the validation control image"
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default='./outputs',
        # required=True
    )

    parser.add_argument(
        "--height",
        type=int,
        default=512,
        required=False
    )

    parser.add_argument(
        "--width",
        type=int,
        default=512,
        required=False
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.0,
        required=False
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        required=False
    )

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

    parser.add_argument(
        "--tile_size",
        type=int,
        default=16,
        required=False
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=4,
        required=False
    )

    parser.add_argument(
        "--noise_aug_strength",
        type=float,
        default=0.02,  # or set to 0.02
        required=False
    )
    parser.add_argument(
        "--frames_overlap",
        type=int,
        default=4,
        required=False
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--decode_chunk_size",
        type=int,
        default=16,
        required=False
    )
    parser.add_argument(
        "--raw_path",
        type=str,
        default='',
        help="Path to raw data",
    )
    parser.add_argument(
        "--mode",
        # 从poses和sample_poses中选择
        type=str,
        choices=['poses', 'sample_images', 'smooth_gt_images'],
        default='poses',
        help="Mode to generate animation",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # torch.set_default_dtype(torch.float16)
    seed = 42
    seed_everything(seed)
    generator = torch.Generator(device='cuda').manual_seed(seed)

    feature_extractor = CLIPImageProcessor.from_pretrained(args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision)
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        low_cpu_mem_usage=True,
    )
    pose_net = PoseNet(noise_latent_channels=unet.config.block_out_channels[0])
    face_encoder = FusionFaceId(
        cross_attention_dim=1024,
        id_embeddings_dim=512,
        # clip_embeddings_dim=image_encoder.config.hidden_size,
        clip_embeddings_dim=1024,
        num_tokens=4, )
    face_model = FaceModel()

    lora_rank = 128
    attn_procs = {}
    unet_svd = unet.state_dict()

    for name in unet.attn_processors.keys():
        if "transformer_blocks" in name and "temporal_transformer_blocks" not in name:
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                # print(f"This is AnimationAttnProcessor: {name}")
                attn_procs[name] = AnimationAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
            else:
                # print(f"This is AnimationIDAttnProcessor: {name}")
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_svd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_svd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = AnimationIDAttnNormalizedProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
                attn_procs[name].load_state_dict(weights, strict=False)
        elif "temporal_transformer_blocks" in name:
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = XFormersAttnProcessor()
            else:
                attn_procs[name] = XFormersAttnProcessor()
    unet.set_attn_processor(attn_procs)

    # resume the previous checkpoint
    if args.posenet_model_name_or_path is not None and args.face_encoder_model_name_or_path is not None and args.unet_model_name_or_path is not None:
        print("Loading existing posenet weights, face_encoder weights and unet weights.")
        if args.posenet_model_name_or_path.endswith(".pth"):
            pose_net_state_dict = torch.load(args.posenet_model_name_or_path, map_location="cpu")
            pose_net.load_state_dict(pose_net_state_dict, strict=True)
        else:
            print("posenet weights loading fail")
            print(1/0)
        if args.face_encoder_model_name_or_path.endswith(".pth"):
            face_encoder_state_dict = torch.load(args.face_encoder_model_name_or_path, map_location="cpu")
            face_encoder.load_state_dict(face_encoder_state_dict, strict=True)
        else:
            print("face_encoder weights loading fail")
            print(1/0)
        if args.unet_model_name_or_path.endswith(".pth"):
            unet_state_dict = torch.load(args.unet_model_name_or_path, map_location="cpu")
            unet.load_state_dict(unet_state_dict, strict=True)
        else:
            print("unet weights loading fail")
            print(1/0)

    torch.cuda.empty_cache()
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    pose_net.requires_grad_(False)
    face_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    weight_dtype = torch.float16
    # weight_dtype = torch.float32
    # weight_dtype = torch.bfloat16

    pipeline = InferenceAnimationPipeline(
        vae=vae,
        image_encoder=image_encoder,
        unet=unet,
        scheduler=noise_scheduler,
        feature_extractor=feature_extractor,
        pose_net=pose_net,
        face_encoder=face_encoder,
    ).to(device='cuda', dtype=weight_dtype)


    #########################################
    # 以下是进行修改的部分，以便不用每次都重新加载模型
    raw_path = args.raw_path
    for folder in sorted(os.listdir(raw_path)):

        # 将所有的args重写
        args.output_dir = os.path.join(raw_path, folder)
        args.validation_control_folder = os.path.join(raw_path, folder, args.mode)
        args.validation_image = os.path.join(raw_path, folder, f'{folder}.png')
        os.makedirs(args.output_dir, exist_ok=True)
        if args.mode == 'sample_images':
            out_file = os.path.join(
                args.output_dir,
                f"{folder}_outputs_sample.mp4",
            )
            png_out_file = os.path.join(args.output_dir, "animated_images_outputs_sample")
        elif args.mode == 'poses':
            out_file = os.path.join(
                args.output_dir,
                f"{folder}_outputs_raw.mp4",
            )
            png_out_file = os.path.join(args.output_dir, "animated_images_outputs_raw")
        elif args.mode == 'smooth_gt_images':
            out_file = os.path.join(
                args.output_dir,
                f"{folder}_outputs_smooth_gt.mp4",
            )
            png_out_file = os.path.join(args.output_dir, "animated_images_outputs_smooth_gt")
        if os.path.exists(out_file):
            continue
        print(f"Start to process {folder}")

        #########################################   

        validation_image_path = args.validation_image
        validation_image = Image.open(args.validation_image).convert('RGB')
        validation_control_images = load_images_from_folder(args.validation_control_folder, width=args.width, height=args.height)

        num_frames = len(validation_control_images)
        face_model.face_helper.clean_all()
        validation_face = cv2.imread(validation_image_path)
        validation_image_bgr = cv2.cvtColor(validation_face, cv2.COLOR_RGB2BGR)
        validation_image_face_info = face_model.app.get(validation_image_bgr)
        if len(validation_image_face_info) > 0:
            validation_image_face_info = sorted(validation_image_face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
            validation_image_id_ante_embedding = validation_image_face_info['embedding']
        else:
            validation_image_id_ante_embedding = None

        if validation_image_id_ante_embedding is None:
            face_model.face_helper.read_image(validation_image_bgr)
            face_model.face_helper.get_face_landmarks_5(only_center_face=True)
            face_model.face_helper.align_warp_face()

            if len(face_model.face_helper.cropped_faces) == 0:
                validation_image_id_ante_embedding = np.zeros((512,))
            else:
                validation_image_align_face = face_model.face_helper.cropped_faces[0]
                print('fail to detect face using insightface, extract embedding on align face')
                validation_image_id_ante_embedding = face_model.handler_ante.get_feat(validation_image_align_face)

        # generator = torch.Generator(device=accelerator.device).manual_seed(23123134)

        decode_chunk_size = args.decode_chunk_size
        video_frames = pipeline(
            image=validation_image,
            image_pose=validation_control_images,
            height=args.height,
            width=args.width,
            num_frames=num_frames,
            tile_size=args.tile_size,
            tile_overlap=args.frames_overlap,
            decode_chunk_size=decode_chunk_size,
            motion_bucket_id=127.,
            fps=7,
            min_guidance_scale=args.guidance_scale,
            max_guidance_scale=args.guidance_scale,
            noise_aug_strength=args.noise_aug_strength,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            output_type="pil",
            validation_image_id_ante_embedding=validation_image_id_ante_embedding,
        ).frames[0]


        for i in range(num_frames):
            img = video_frames[i]
            video_frames[i] = np.array(img)

        
        os.makedirs(png_out_file, exist_ok=True)
        save_frames_as_png(video_frames, png_out_file)
        save_vid_side_by_side([video_frames], validation_control_images, out_file, 25)
