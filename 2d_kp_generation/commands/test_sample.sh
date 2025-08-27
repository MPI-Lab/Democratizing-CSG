ckpt_base_path="./pretrained_weights/model_weights/"
ckpt_step=("3000000.pth")

CUDA_VISIBLE_DEVICES=0 python -m train.test_saved_ckpt \
--save_dir save_sample_dataset/test_from_3000000_steps \
--dataset_config data_loaders.audio_to_2dkp.config.train_from_3000000_steps \
--cond audio_concat,long_video \
--arch trans_enc \
--emb_trans_dec True \
--train_platform_type TensorboardPlatform \
--diffusion_steps 100 \
--cond_mask_prob 0.1 \
--lr 5e-5 \
--train_batch_size 32 \
--layers 16 \
--num_heads 16 \
--latent_dim 1024 \
--ff_size 4096 \
--device 0 \
--num_steps 2000000 \
--save_interval 100000 \
--loss_weight mean \
--overwrite_model \
--if_train_vis \
--if_val_vis \
--resume_checkpoint "${ckpt_base_path}/checkpoint_${ckpt_step}"



echo "所有任务已完成。"
