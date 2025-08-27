import itertools
import json
import os
import torch
from tqdm import tqdm
from sample.sampler import model_kwargs_to_device
from train.train_platforms import NoPlatform, TrainPlatform
from utils import dist_utils
from torch.optim import AdamW
from torch.nn import Module
import torch.distributed as dist
import json
import random

class Trainer:
    def __init__(self, args, model: Module, train_data, test_data, train_platform: TrainPlatform = NoPlatform(None)):
        self.args = args
        self.train_data = train_data
        self.test_data = test_data
        self.kp_loss_weight = self.args.loss_weight

        if dist.is_available() and dist.is_initialized():
            self.use_ddp = True
            self.model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[dist_utils.dev()],
                output_device=dist_utils.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=True,
            )

        else:
            self.use_ddp = False
            self.model = model
        
        self.save_sample_progressive = self.args.save_sample_progressive
        
        self.train_vis_steps = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        self.train_vis_save = os.path.join(self.args.save_dir, f"train_samples")
        self.if_train_vis = self.args.if_train_vis

        self.val_steps = [1000, 5000, 10000]
        self.if_val_vis = self.args.if_val_vis

        self.train_platform = train_platform
        self.optimizer = AdamW(model.parameters(), lr=args.lr)
        self.step = 0
        self.setup_device()
        self.setup_dir()
        self.loss = None
        self.mean_loss = None


        if self.args.resume_checkpoint:
            self.load_checkpoint()

    def load_checkpoint(self):
        checkpoint = torch.load(self.args.resume_checkpoint, map_location=self.device)
        if self.use_ddp:
            self.model.module.load_state_dict(checkpoint["model"], strict=False)
        elif not self.use_ddp:
            self.model.load_state_dict(checkpoint["model"], strict=False)
        # self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step = checkpoint["step"] + 1
        self.mean_loss = checkpoint["mean_loss"]

        if self.use_ddp:
            dist.barrier()
        print(f"Loaded checkpoint: [{self.args.resume_checkpoint}]")

    def setup_device(self):
        self.device = dist_utils.dev()

    def setup_dir(self):
        if os.path.exists(self.args.save_dir) and not self.args.overwrite_model and (os.path.dirname(self.args.resume_checkpoint) != self.args.save_dir):
            raise FileExistsError(f"Save dir [{self.args.save_dir}] already exists.")
        elif not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)
        self.args_path = os.path.join(self.args.save_dir, "args.json")
        with open(self.args_path, "w") as fw:
            json.dump(vars(self.args), fw, indent=4, sort_keys=True)

    def calculate_loss(self, motion, cond) -> torch.Tensor:
        pass  # To be overridden

    def evaluate(self):
        pass  # To be overridden

    def train(self):
        for epoch in itertools.count(start=1):
            if self.use_ddp:
                self.train_data.sampler.set_epoch(epoch)
                self.test_data.sampler.set_epoch(epoch)
            iter_train = tqdm(self.train_data)
            for motion, cond in iter_train:
                motion = motion.to(self.device)
                cond = model_kwargs_to_device(cond, self.device)
                self.optimizer.zero_grad()
                self.loss = self.calculate_loss(motion, cond)
                self.mean_loss = self.mean_loss * self.step / (self.step + 1) + self.loss.item() / (self.step + 1) if self.mean_loss else self.loss.item()

                self.loss.backward()
                self.optimizer.step()
                
                if dist_utils.is_main_process():
                    iter_train.set_description(f"epoch: {epoch:<3} | step: {self.step:<6} | loss: {self.loss:.4f}　| mean_loss: {self.mean_loss:.4f}")
                    self.train_platform.report_scalar(name="train/mean_loss", value=self.mean_loss, iteration=self.step, group_name="Loss")
                    self.train_platform.report_scalar(name="train/loss", value=self.loss, iteration=self.step, group_name="Loss")



                if self.if_train_vis and (self.step in self.train_vis_steps or self.step % 5000 == 0):
                    if dist_utils.is_main_process():
                        self.model.eval()
                        with torch.no_grad():
                            if self.args.cond_mask_prob > 0 and self.step % 50000 == 0:
                                for scale in [1,3.5,6,8.5]:
                                    self.sample_and_visualize_unpaired(motion, cond, save_num=2, save_path=self.train_vis_save, prefix=f"train_{self.step}_unpaired_cfg_{scale}_", scale=scale)
                                    self.sample_and_visualize(motion, cond, save_num=2, save_path=self.train_vis_save, prefix=f"train_{self.step}_cfg_{scale}_", scale=scale)
                            else:
                                self.sample_and_visualize_unpaired(motion, cond, save_num=2, save_path=self.train_vis_save, prefix=f"train_{self.step}_unpaired_")
                                self.sample_and_visualize(motion, cond, save_num=2, save_path=self.train_vis_save, prefix=f"train_{self.step}_")
                        self.model.train()
                    if self.use_ddp:
                        dist.barrier()

                # torch.cuda.empty_cache()
                if self.step in self.val_steps or self.step % self.args.save_interval == 0 and self.step != 0:
                    self.save_checkpoint()
                    if self.if_val_vis:
                        self.test_saved_ckpt()
                    if self.args.eval_during_training:
                        self.evaluate()
                self.step += 1
                if self.use_ddp:
                    dist.barrier()
                if self.step > self.args.num_steps:
                    return



    def test_saved_ckpt(self):
        print("Evaluating in test data...")

        self.model.eval()
        test_step = 0
        test_loss = 0.0
        test_mean_loss = None
        save_path = os.path.join(self.args.save_dir, f"val_samples_checkpoints_{self.step}")
        os.makedirs(save_path, exist_ok=True)
        with torch.no_grad():
            iter_test = tqdm(self.test_data, disable=not dist_utils.is_main_process())
            for motion, cond in iter_test:
                motion = motion.to(self.device)
                cond = model_kwargs_to_device(cond, self.device)
                
                test_loss = self.calculate_loss(motion, cond)
                test_mean_loss = test_mean_loss * test_step / (test_step + 1) + test_loss.item() / (test_step + 1) if test_mean_loss else test_loss.item()

                iter_test.set_description(f"global_step: {self.step:<6} | test_step: {test_step:<6} | test_loss: {test_loss:.4f} | test_mean_loss: {test_mean_loss:.4f}")

                save_num = round(15 / dist.get_world_size()) if self.use_ddp else 15
                # save_num = 1000
                if test_step % 2 == 0:
                # if test_step % 1 == 0:
                    if self.args.cond_mask_prob > 0 and random.random() < 0.3:
                        for scale in [1,3.5,6,8.5]:
                            if "long_video" in self.args.cond:
                                self.sample_and_visualize_long(motion, cond, save_num=save_num, save_path=save_path, prefix=f"long_cfg_{scale}_", scale=scale)
                                self.sample_and_visualize_long_unpaired(motion, cond, save_num=save_num, save_path=save_path, prefix=f"long_unpaired_cfg_{scale}_", scale=scale)
                            self.sample_and_visualize_unpaired(motion, cond, save_num=save_num, save_path=save_path, prefix=f"unpaired_cfg_{scale}_", scale=scale)
                            self.sample_and_visualize(motion, cond, save_num=save_num, save_path=save_path, prefix=f"cfg_{scale}_", scale=scale)
                    else:
                        if "long_video" in self.args.cond:
                            self.sample_and_visualize_long(motion, cond, save_num=save_num, save_path=save_path, prefix=f"long_")
                            self.sample_and_visualize_long_unpaired(motion, cond, save_num=save_num, save_path=save_path, prefix=f"long_unpaired_")
                        self.sample_and_visualize_unpaired(motion, cond, save_num=save_num, save_path=save_path, prefix=f"unpaired_")
                        self.sample_and_visualize(motion, cond, save_num=save_num, save_path=save_path, prefix=f"")
                test_step += 1

        if self.use_ddp:
            dist.barrier()
            # Gather mean_test_loss from all processes in DDP mode
            test_mean_loss_tensor = torch.tensor(test_mean_loss, device=self.device)
            dist.all_reduce(test_mean_loss_tensor, op=dist.ReduceOp.SUM)
            test_mean_loss = test_mean_loss_tensor.item() / dist.get_world_size()

        if dist_utils.is_main_process():
            # Log the mean test loss
            print(f"Mean test loss: {test_mean_loss:.4f}")
            self.train_platform.report_scalar(name="test/loss", value=test_mean_loss, iteration=self.step, group_name="Loss")
        
        self.model.train()

    def save_checkpoint(self):
        if self.use_ddp:
            if dist_utils.is_main_process():
                model_state_dict = self.model.module.state_dict()
                # 移除 aud_m 部分的权重
                keys_to_remove = [key for key in model_state_dict if key.startswith('aud_m')]
                for key in keys_to_remove:
                    del model_state_dict[key]
                checkpoint = {
                    "model": model_state_dict,
                    "optimizer": self.optimizer.state_dict(),
                    "step": self.step,
                    "mean_loss": self.mean_loss,
                }
                checkpoint_path = os.path.join(self.args.save_dir, f"checkpoint_{self.step}.pth")
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint to [{checkpoint_path}]")
        else:
            model_state_dict = self.model.state_dict()
            # 移除 aud_m 部分的权重
            keys_to_remove = [key for key in model_state_dict if key.startswith('aud_m')]
            for key in keys_to_remove:
                del model_state_dict[key]
            checkpoint = {
                "model": model_state_dict,
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
                "mean_loss": self.mean_loss,
            }
            checkpoint_path = os.path.join(self.args.save_dir, f"checkpoint_{self.step}.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to [{checkpoint_path}]")

