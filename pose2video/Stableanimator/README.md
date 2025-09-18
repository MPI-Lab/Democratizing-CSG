# Environments
  ```python 
  conda create -n stableani python=3.10
  pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
  pip install torch==2.5.1+cu124 xformers --index-url https://download.pytorch.org/whl/cu124
  pip install -r requirements.txt
  pip install onnxruntime-gpu==1.19.2 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
  pip install moviepy==1.0.3
  ```

# Model CheckPoints Download
  Download the model weights here: [stableanimator_weights](https://huggingface.co/Moon-jack/Democratize-CSG) and the overall file structure will be:
  ```shell
  StableAnimator/
    ├── animation
    ├── commands
    ├── DWPose
    ├── checkpoints
    │   ├── DWPose
    │   │   ├── dw-ll_ucoco_384.onnx
    │   │   └── yolox_l.onnx
    │   ├── Animation
    │   │   ├── pose_net.pth
    │   │   ├── face_encoder.pth
    │   │   └── unet.pth
    │   ├── stable-video-diffusion-img2vid-xt
    │   │   ├── feature_extractor
    │   │   ├── image_encoder
    │   │   ├── scheduler
    │   │   ├── unet
    │   │   ├── vae
    │   │   ├── model_index.json
    │   │   ├── svd_xt.safetensors
    │   │   └── svd_xt_image_decoder.safetensors
    ├── models
    │   │   └── antelopev2
    │   │       ├── 1k3d68.onnx
    │   │       ├── 2d106det.onnx
    │   │       ├── genderage.onnx
    │   │       ├── glintr100.onnx
    │   │       └── scrfd_10g_bnkps.onnx
    ├── scripts
    ├── utils
    ├── ge_kp2kpvideo.py
    ├── inference_use.py
    ├── README.md
    ├── requirement.txt 
  ```
Note: The model weights we provide have been fine-tuned on our dataset. The original stableanimator weights can be downloaded here: [original_stableanimator](https://huggingface.co/FrancisRing/StableAnimator). If you want to use the original stableanimator, you can replace the weights here.

# Inference:
  Refer to [commands/infer_batch.sh](https://github.com/MPI-Lab/Democratizing-CSG/blob/main/pose2video/Stableanimator/commands/infer_batch.sh)
  ```shell
  Usage: bash commands/infer_batch.sh <dataset_path> <output_path> <save_stage1_path> [devices] [cfg]
  Parameter description:
  <dataset_path>: The path of dataset, which contains the test.txt and mp4_cropped folders.
  <output_path>: The path used to save all outputs.
  <save_stage1_path>: The saved path of stage 1 output, which contains the sample_pose folder.
  <cfg>: Optional parameter. Default 3.5.
  <devices>: Optional parameters.Default 0,1,2,3,4,5,6,7.
  ```

# Acknowledgements
Thanks to [StableAnimator](https://github.com/Francis-Rings/StableAnimator), our code is built upon their work.