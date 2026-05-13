# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GaussCtrl Pipeline and trainer"""

import os
from dataclasses import dataclass, field
from itertools import cycle
from typing import Optional, Type, List
from rich.progress import Console
from copy import deepcopy
import numpy as np 
from PIL import Image
import mediapy as media
from lang_sam import LangSAM

import torch, random
from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.viewer.server.viewer_elements import ViewerNumber, ViewerText
from diffusers.schedulers import DDIMScheduler, DDIMInverseScheduler
from gaussctrl.gc_datamanager import (
    GaussCtrlDataManagerConfig,
)
from diffusers.models.attention_processor import AttnProcessor
from gaussctrl import utils
from nerfstudio.viewer_legacy.server.utils import three_js_perspective_camera_focal_length
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils import colormaps

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler, DDIMInverseScheduler

CONSOLE = Console(width=120)

@dataclass
class GaussCtrlPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: GaussCtrlPipeline)
    """target class to instantiate"""
    datamanager: GaussCtrlDataManagerConfig = GaussCtrlDataManagerConfig()
    """specifies the datamanager config"""
    render_rate: int = 500
    """how many gauss steps for gauss training"""
    edit_prompt: str = ""
    """Positive Prompt"""
    reverse_prompt: str = "" 
    """DDIM Inversion Prompt"""
    langsam_obj: str = ""
    """The object to be edited"""
    guidance_scale: float = 5
    """Classifier Free Guidance"""
    num_inference_steps: int = 20
    """Inference steps"""
    chunk_size: int = 1
    """Batch size for image editing. Keep at 1 for GPUs with ≤ 12 GB VRAM."""
    attention_chunk_size: int = 256
    """Query-token chunk size for custom attention. Lower uses less VRAM but is slower."""
    offload_model_during_edit: bool = False
    """Move the Gaussian model to CPU while Stable Diffusion edits images."""
    unload_diffusion_after_edit: bool = True
    """Delete the diffusion pipeline after editing to free VRAM before Gaussian training."""
    ref_view_num: int = 4
    """Number of reference frames"""
    diffusion_ckpt: str = 'CompVis/stable-diffusion-v1-4'
    """Diffusion checkpoints"""
    

class GaussCtrlPipeline(VanillaPipeline):
    """GaussCtrl pipeline"""

    config: GaussCtrlPipelineConfig

    def __init__(
        self,
        config: GaussCtrlPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)
        self.test_mode = test_mode
        self.langsam = LangSAM()
        
        self.edit_prompt = self.config.edit_prompt
        self.reverse_prompt = self.config.reverse_prompt
        self.pipe_device = 'cuda:0'
        self.ddim_scheduler = DDIMScheduler.from_pretrained(self.config.diffusion_ckpt, subfolder="scheduler")
        self.ddim_inverser = DDIMInverseScheduler.from_pretrained(self.config.diffusion_ckpt, subfolder="scheduler")
        
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.config.diffusion_ckpt, controlnet=controlnet
        ).to(self.device).to(torch.float16)
        self.pipe.to(self.pipe_device)

        # Attention slicing reduces peak VRAM for the VAE encoder/decoder.
        # NOTE: do NOT call enable_model_cpu_offload() here — it installs
        #       device-move hooks that conflict with the manual .to('cpu') /
        #       .to(device) calls used in edit_images() and render_reverse().
        #       Mixing both causes memory to increase, not decrease.
        try:
            self.pipe.enable_attention_slicing()
        except Exception:
            CONSOLE.print("Warning: enable_attention_slicing() failed, VRAM may be high. Install the latest version of diffusers for this feature.", style="bold red")
            pass

        # Use xformers memory-efficient attention for the VAE and any attention
        # layers that are NOT overridden by CrossViewAttnProcessor (e.g. the
        # text encoder cross-attention in the UNet).  This is a no-op if
        # xformers is not installed.
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            CONSOLE.print("xformers memory-efficient attention enabled.", style="bold green")
        except Exception:
            CONSOLE.print(
                "xformers not available — using chunked attention fallback. "
                "Install xformers for extra speed: pip install xformers",
                style="bold yellow",
            )

        added_prompt = 'best quality, extremely detailed'
        self.positive_prompt = self.edit_prompt + ', ' + added_prompt
        self.positive_reverse_prompt = self.reverse_prompt + ', ' + added_prompt
        self.negative_prompts = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
        
        view_num = len(self.datamanager.cameras) 
        anchors = [(view_num * i) // self.config.ref_view_num for i in range(self.config.ref_view_num)] + [view_num]
        
        random.seed(13789)
        self.ref_indices = [random.randint(anchor, anchors[idx+1]) for idx, anchor in enumerate(anchors[:-1])] 
        self.num_ref_views = len(self.ref_indices)

        self.num_inference_steps = self.config.num_inference_steps
        self.guidance_scale = self.config.guidance_scale
        self.controlnet_conditioning_scale = 1.0
        self.eta = 0.0
        self.chunk_size = self.config.chunk_size

    def _log_cuda_memory(self, prefix: str) -> None:
        try:
            CONSOLE.print(utils.cuda_memory_summary(prefix), style="dim")
        except Exception:
            pass

    def _move_gaussian_model(self, device) -> None:
        self._model.to(device)

    def render_reverse(self):
        '''Render rgb, depth and reverse rgb images back to latents'''
        for cam_idx in range(len(self.datamanager.cameras)):
            CONSOLE.print(f"Rendering view {cam_idx}", style="bold yellow")
            current_cam = self.datamanager.cameras[cam_idx].to(self.device)
            if current_cam.metadata is None:
                current_cam.metadata = {}
            current_cam.metadata["cam_idx"] = cam_idx
            rendered_image = self._model.get_outputs_for_camera(current_cam)

            rendered_rgb = rendered_image['rgb'].to(torch.float16) # [512 512 3] 0-1
            rendered_depth = rendered_image['depth'].to(torch.float16) # [512 512 1]

            # reverse the images to noises
            self.pipe.unet.set_attn_processor(processor=AttnProcessor())
            self.pipe.controlnet.set_attn_processor(processor=AttnProcessor()) 
            init_latent = self.image2latent(rendered_rgb)
            disparity = self.depth2disparity_torch(rendered_depth[:,:,0][None]) 
            
            self.pipe.scheduler = self.ddim_inverser
            latent, _ = self.pipe(prompt=self.positive_reverse_prompt, #  placeholder here, since cfg=0
                                num_inference_steps=self.num_inference_steps, 
                                latents=init_latent, 
                                image=disparity, return_dict=False, guidance_scale=0, output_type='latent')

            # LangSAM is optional
            if self.config.langsam_obj != "":
                langsam_obj = self.config.langsam_obj
                langsam_rgb_pil = Image.fromarray((rendered_rgb.cpu().numpy() * 255).astype(np.uint8))
                masks, _, _, _ = self.langsam.predict(langsam_rgb_pil, langsam_obj)
                try:
                    mask_npy = masks.clone().cpu().numpy()[0] * 1
                except:
                    # There is a chance that nothing is detected
                    mask_npy = None

            if self.config.langsam_obj != "":
                self.update_datasets(cam_idx, rendered_rgb.cpu(), rendered_depth, latent, mask_npy)
            else: 
                self.update_datasets(cam_idx, rendered_rgb.cpu(), rendered_depth, latent, None)

            del current_cam, rendered_image, rendered_rgb, rendered_depth, init_latent, disparity, latent
            utils.free_cuda_memory()

        self.pipe.to('cpu')
        utils.free_cuda_memory()
        self._log_cuda_memory("After render_reverse cleanup")
        
    def edit_images(self):
        '''Edit images with ControlNet and AttnAlign''' 
        # Set up ControlNet and AttnAlign
        self.pipe.scheduler = self.ddim_scheduler
        self.pipe.unet.set_attn_processor(
                        processor=utils.CrossViewAttnProcessor(self_attn_coeff=0.6,
                        unet_chunk_size=2,
                        attention_chunk_size=self.config.attention_chunk_size))
        self.pipe.controlnet.set_attn_processor(
                        processor=utils.CrossViewAttnProcessor(self_attn_coeff=0,
                        unet_chunk_size=2,
                        attention_chunk_size=self.config.attention_chunk_size))
        CONSOLE.print("Done Resetting Attention Processor", style="bold blue")
        self._log_cuda_memory("Before editing cleanup")

        # Stable Diffusion does all editing work here; the Gaussian model can
        # leave CUDA until training resumes.
        model_offloaded = False
        if self.config.offload_model_during_edit:
            self._move_gaussian_model('cpu')
            model_offloaded = True
        utils.free_cuda_memory()
        self._log_cuda_memory("After editing cleanup")
        
        print("#############################")
        CONSOLE.print("Start Editing: ", style="bold yellow")
        CONSOLE.print(f"Reference views are {[j+1 for j in self.ref_indices]}", style="bold yellow")
        print("#############################")
        ref_disparity_list = []
        ref_z0_list = []
        for ref_idx in self.ref_indices:
            ref_data = deepcopy(self.datamanager.train_data[ref_idx]) 
            ref_disparity = self.depth2disparity(ref_data['depth_image']) 
            ref_z0 = ref_data['z_0_image']
            ref_disparity_list.append(ref_disparity)
            ref_z0_list.append(ref_z0) 
            
        ref_disparities = np.concatenate(ref_disparity_list, axis=0)
        ref_z0s = np.concatenate(ref_z0_list, axis=0)
        # keep reference tensors on CPU (move to GPU per-chunk to save persistent VRAM)
        ref_disparity_torch = torch.from_numpy(ref_disparities.copy()).to(torch.float16)
        ref_z0_torch = torch.from_numpy(ref_z0s.copy()).to(torch.float16)

        try:
            # Edit images in chunk
            for idx in range(0, len(self.datamanager.train_data), self.chunk_size):
                chunked_data = self.datamanager.train_data[idx: idx+self.chunk_size]

                indices = [current_data['image_idx'] for current_data in chunked_data]
                mask_images = [current_data['mask_image'] for current_data in chunked_data if 'mask_image' in current_data.keys()]
                unedited_images = [current_data['unedited_image'] for current_data in chunked_data]
                CONSOLE.print(f"Generating view: {indices}", style="bold yellow")

                depth_images = [self.depth2disparity(current_data['depth_image']) for current_data in chunked_data]
                disparities = np.concatenate(depth_images, axis=0)
                disparities_torch = torch.from_numpy(disparities.copy()).to(torch.float16)

                z_0_images = [current_data['z_0_image'] for current_data in chunked_data] # list of np array
                z0s = np.concatenate(z_0_images, axis=0)
                latents_torch = torch.from_numpy(z0s.copy()).to(torch.float16)

                disp_ctrl_cpu = torch.cat((ref_disparity_torch, disparities_torch), dim=0)
                latents_cpu = torch.cat((ref_z0_torch, latents_torch), dim=0)
                del disparities_torch, latents_torch

                utils.free_cuda_memory()
                self._log_cuda_memory(f"Before diffusion chunk {indices}")
                CONSOLE.print("Running Stable Diffusion ControlNet Pipeline for this chunk...", style="bold yellow")

                latents_chunk = None
                disp_ctrl_chunk = None
                chunk_edited = None
                try:
                    self.pipe.to(self.pipe_device)
                    try:
                        self.pipe.enable_attention_slicing()
                    except Exception:
                        pass

                    disp_ctrl_chunk = disp_ctrl_cpu.to(self.pipe_device)
                    latents_chunk = latents_cpu.to(self.pipe_device)

                    chunk_edited = self.pipe(
                                        prompt=[self.positive_prompt] * (self.num_ref_views+len(chunked_data)),
                                        negative_prompt=[self.negative_prompts] * (self.num_ref_views+len(chunked_data)),
                                        latents=latents_chunk,
                                        image=disp_ctrl_chunk,
                                        num_inference_steps=self.num_inference_steps,
                                        guidance_scale=self.guidance_scale,
                                        controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                                        eta=self.eta,
                                        output_type='pt',
                                    ).images[self.num_ref_views:].cpu()
                finally:
                    del latents_chunk, disp_ctrl_chunk
                    utils.free_cuda_memory()
                    self._log_cuda_memory(f"After diffusion chunk {indices}")

                del disp_ctrl_cpu, latents_cpu, z0s, disparities

                # Insert edited images back to train data for training
                for local_idx, edited_image in enumerate(chunk_edited):
                    global_idx = indices[local_idx]

                    bg_cntrl_edited_image = edited_image
                    if mask_images != []:
                        mask = torch.from_numpy(mask_images[local_idx])
                        bg_mask = 1 - mask

                        unedited_image = unedited_images[local_idx].permute(2,0,1)
                        bg_cntrl_edited_image = edited_image * mask[None] + unedited_image * bg_mask[None]

                    self.datamanager.train_data[global_idx]["image"] = bg_cntrl_edited_image.permute(1,2,0).to(torch.float32) # [512 512 3]
                del chunk_edited
        finally:
            if self.config.unload_diffusion_after_edit:
                del self.pipe
            if model_offloaded:
                self._move_gaussian_model(self.device)
            utils.free_cuda_memory()
            self._log_cuda_memory("After edit_images cleanup")
        print("#############################")
        CONSOLE.print("Done Editing", style="bold yellow")
        print("#############################")

    @torch.no_grad()
    def image2latent(self, image):
        """Encode images to latents"""
        image = image * 2 - 1
        image = image.permute(2, 0, 1).unsqueeze(0) # torch.Size([1, 3, 512, 512]) -1~1
        latents = self.pipe.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        return latents

    def depth2disparity(self, depth):
        """
        Args: depth numpy array [1 512 512]
        Return: disparity
        """
        disparity = 1 / (depth + 1e-5)
        disparity_map = disparity / np.max(disparity) # 0.00233~1
        disparity_map = np.concatenate([disparity_map, disparity_map, disparity_map], axis=0)
        return disparity_map[None]
    
    def depth2disparity_torch(self, depth):
        """
        Args: depth torch tensor
        Return: disparity
        """
        disparity = 1 / (depth + 1e-5)
        disparity_map = disparity / torch.max(disparity) # 0.00233~1
        disparity_map = torch.concatenate([disparity_map, disparity_map, disparity_map], dim=0)
        return disparity_map[None]

    def update_datasets(self, cam_idx, unedited_image, depth, latent, mask):
        """Save mid results"""
        self.datamanager.train_data[cam_idx]["unedited_image"] = unedited_image 
        self.datamanager.train_data[cam_idx]["depth_image"] = depth.permute(2,0,1).cpu().to(torch.float32).numpy()
        self.datamanager.train_data[cam_idx]["z_0_image"] = latent.cpu().to(torch.float32).numpy()
        if mask is not None:
            self.datamanager.train_data[cam_idx]["mask_image"] = mask 

    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict and performs image editing.
        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, batch = self.datamanager.next_train(step) # camera, data
        model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
        
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError
