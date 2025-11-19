# -------------  cl/training/trainer.py  --------------------
#─────────────────────────────────────────────────────────────────────────────
# Monkey-patch for PyTorch < 2.2, providing get_default_device() so
# diffusers/transformers pipelines don’t crash on torch.get_default_device()
import torch
if not hasattr(torch, "get_default_device"):
    torch.get_default_device = lambda: ("cuda" if torch.cuda.is_available() else "cpu")
#─────────────────────────────────────────────────────────────────────────────

# optimize PyTorch for speed
torch.backends.cuda.matmul.allow_tf32 = True  # if you want speed > accuracy; enable TF32 on mats
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True  # enable TF32 on convolutions
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = False  # Faster but less deterministic
torch.set_float32_matmul_precision("high")     # trade a bit of FP32 precision for speed , for ~2–4× GEMM speedup on Ampere.

# Enable optimized attention
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

# Optimize CUDA allocator
import os, gc
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,roundup_power2_divisions:16"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,expandable_segments:True"
# Aggressive memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,expandable_segments:True,roundup_power2_divisions:16"

torch.cuda.empty_cache()
gc.collect()

import torch.multiprocessing as _mp
_mp.set_sharing_strategy("file_system")   # <— avoid /dev/shm exhaustion on many workers


import shutil
# Update LD_LIBRARY_PATH to include where libcuda.so actually is
os.environ["LD_LIBRARY_PATH"] = "/usr/local/nvidia/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
# Set torch compile backend
os.environ["TORCH_COMPILE_BACKEND"] = "inductor"
# force‐load the real driver

# disable Dynamo/inductor checks
import types
import math
from PIL import Image

# Before running your training script
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:128"

import datetime, time, types
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional

import torch.nn.functional as F
from torchvision import models
import bitsandbytes as bnb
from transformers import CLIPProcessor, CLIPModel
# from peft import get_peft_model, LoraConfig #, prepare_model_for_kbit_training

import wandb

from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs, DeepSpeedPlugin
# from diffusers import StableDiffusionPipeline

from data.latent_webdataset import get_dataloader
from model.architecture   import CubeDiffModel

from diffusers import UNet2DConditionModel, DDPMScheduler # DDIMScheduler # DPMSolverMultistepScheduler # EulerAncestralDiscreteScheduler
# from transformers import get_cosine_schedule_with_warmup, get_wsd_schedule # (2025-5-18 this made LR be exactly 0 after 29 steps, why ?)
import tarfile

from inference.pipeline import CubeDiffPipeline
from model.normalization import replace_group_norms

import tarfile, io, torch
from torch.utils.data import TensorDataset

# Disable FlashAttention/CUTLASS and force PyTorch SDPA because:
# diffusers >= 0.25 + transformers >= 4.35 now use:
# scaled_dot_product_attention (SDPA) with Flash/CUTLASS backends
# L4 GPU does not have a kernel implementation for the requested attention mode.
                                                 
import os
os.environ["PYTORCH_CUDA_ALLOW_FP16_REDUCED_PRECISION_REDUCTION"] = "1"
os.environ["ATTENTION_BACKEND"] = "SDPA"
os.environ["USE_FLASH_ATTENTION"] = "0"
os.environ["USE_MEMORY_EFFICIENT_ATTENTION"] = "0"
os.environ["PYTORCH_SDPA_ENABLE_FLASH"] = "0"
os.environ["PYTORCH_SDPA_ALLOW_FLASH"] = "0"
os.environ["PYTORCH_SDPA_ALLOW_MEM_EFFICIENT"] = "0"
os.environ["PYTORCH_SDPA_FORCE_FALLBACK"] = "1"

import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


def load_tar_to_dataset(tar_path: str):
    """
    Reads every *.pt file from the tar, loads it with torch.load,
    and returns a TensorDataset(latents, text_embs).
    Assumes each .pt contains {'latents':Tensor[6,4,64,64], 'text_emb':Tensor[L,D]}.
    """
    latents_list, emb_list = [], []
    with tarfile.open(tar_path, "r") as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name.endswith(".pt"):
                f = tar.extractfile(m)
                sample = torch.load(io.BytesIO(f.read()), map_location="cpu")
                latents_list .append(sample["latents"])
                emb_list     .append(sample["text_emb"])
    all_latents = torch.stack(latents_list, dim=0)   # [N,6,4,64,64]
    all_embs    = torch.stack(emb_list,    dim=0)   # [N,L,dim]
    return TensorDataset(all_latents, all_embs)

# -----------------------------------------------------------
# — Monkey-patch UNet2DConditionModel.forward to swallow any extra kwargs —
orig_unet_forward = UNet2DConditionModel.forward
def _patched_unet_forward(self, sample, timestep, encoder_hidden_states, *args, **kwargs):
    """
    Accept the three required args, ignore anything else.
    Guards against 'decoder_input_ids', 'use_cache', etc.
    """
    return orig_unet_forward(self, sample, timestep, encoder_hidden_states)

UNet2DConditionModel.forward = _patched_unet_forward
# -------------------------------------------------------------
class CubeDiffTrainer:
    def __init__(self, config,
                 pretrained_model_name="runwayml/stable-diffusion-v1-5",
                 output_dir="./outputs",
                 mixed_precision= "bf16", # "fp16", automatically wrap your forward/backward in torch.cuda.amp and cut activation/optimizer memory in half, for another ~2× speedup on A100s 
                 gradient_accumulation_steps=1):
        self.config  = config
        self.output_dir = Path(output_dir); self.output_dir.mkdir(exist_ok=True, parents=True)
        self.images_dir = self.output_dir / "samples"; self.images_dir.mkdir(exist_ok=True)
        self.logs_dir   = self.output_dir / "logs";    self.logs_dir.mkdir(exist_ok=True)
        self.mixed_precision = mixed_precision
        self.model_dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16

        self.micro_batch = self.config["batch_size"]      # = 2
        self.acc_steps   = self.config["gradient_accum_steps"]     # e.g. 1 or 4

        self._last_sampled_ckpt = None

        # 1) Create a DDP kwargs handler that turns on unused-parameter detection
        # Ensure DeepSpeed Activation
        ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=True)
        # in order to get more memory headroom for even larger batches, we can shard optimizer states (Stage 1/2/3) with DeepSpeed ZeRO
        # Accelerate will wire DeepSpeed ZeRO under the hood, cut the per-GPU memory by up to 3×, and let us bump batch sizes further
        # ds_config = {
        #             "zero_optimization": {
        #                 "stage": 2,                   # stag 2: shard optimizer + gradients; stage 3: shards parameters, gradients, and optimizer state → reduces each all-reduce by ~8×.
        #                 "contiguous_gradients": True,
        #                 "overlap_comm": True
        #             },
        #             "bf16": {"enabled": True},       # offload full precision if needed
        #             # tell DeepSpeed how many samples each GPU sees per step:
        #             "train_micro_batch_size_per_gpu": self.config["batch_size"],
        # }
        # ds_plugin = DeepSpeedPlugin(hf_ds_config=ds_config)
        # use the global DS config defined in accelerate_config.yaml.
        self.accelerator = Accelerator(
                                    # mixed_precision=mixed_precision,
                                    #    gradient_accumulation_steps=self.acc_steps,
                                    #    deepspeed_plugin=ds_plugin, 
                                        kwargs_handlers=[ddp_handler], 
                                    )
        
        # Debug: Print what backend we're actually using
        print(f"trainer.py - init - Rank {self.accelerator.process_index}: Accelerator state: {self.accelerator.state}")
        print(f"trainer.py - init - Distributed type: {self.accelerator.distributed_type}")
        print(f"trainer.py - init - Using DeepSpeed: {self.accelerator.state.deepspeed_plugin is not None}")
        if self.accelerator.state.deepspeed_plugin is not None:
            print(f"trainer.py - init - DeepSpeed config: {self.accelerator.state.deepspeed_plugin.deepspeed_config}")
        
        # device_str = f"cuda:{self.accelerator.local_process_index}"
        # Debug: Print device assignment for each process
        print(f"Rank {self.accelerator.process_index}: "
            f"local_rank={self.accelerator.local_process_index}, "
            f"device={self.accelerator.device}")
        
        device_str = self.accelerator.device  # This will be cuda:0, cuda:1, cuda:2, or cuda:3
        print(f"Moving all components to {device_str}...")
        
        self.pipe = CubeDiffPipeline(
             pretrained_model_name="runwayml/stable-diffusion-v1-5",
             config=self.config,
        )
        
        # 4) Move the cube‐diff wrapper model (which holds base_unet and its time‐embedding, face‐embedding layers) onto the same GPU
        self.pipe.model = self.pipe.model.to(device_str)
        
        # 2) THE ONE MODEL: pipeline.model is a CubeDiffModel(base_unet=HF U-Net)
        #    that already has inflation logic in its __init__
        self.model = self.pipe.model
        
        # 3) we only ever generate in eval mode
        self.pipe.model.eval()

        # 3) Move the VAE and text_encoder if they aren’t already included in pipeline.to()
        #    (Sometimes pipeline.to(device) will already cascade to its submodules,
        #    but we call them explicitly to be safe.)
        self.pipe.vae            = self.pipe.vae.to(device_str)
        self.pipe.text_encoder   = self.pipe.text_encoder.to(device_str)
        
        # EXTRA: Explicitly move positional encoding if it exists
        if hasattr(self.pipe.model, 'positional_encoding'):
            self.pipe.model.positional_encoding = self.pipe.model.positional_encoding.to(device_str)
            print(f"Explicitly moved positional_encoding to {device_str}")
        
        # StableDiffusionPipeline does not have .eval(), so do each submodule:
        self.pipe.vae.eval()
        self.pipe.text_encoder.eval()
        
        # Now the CubeDiffModel wrapper (which holds positional_encoding + base_unet + etc):
        self.pipe.model.eval()            # or at least self.pipe.model.base_unet.eval()

        # 6) Now build a properly‐shaped “fake” tensor, all on device_str.
        #    CubeDiffModel.forward expects:
        #      - latents: [B, 6, 4, H, W]
        #      - timesteps: [B]  (internally broadcast to [B*6])
        #      - encoder_hidden_states: [B, hidden_size] (internally broadcast to [B*6, hidden_size])
        #    The patched forward will reshape it into the 15‐channel input that the underlying Conv2D layers want.

        #    (a) Fake latents: B=1, F=6 faces, C=4 latent channels, H=W=64
        fake_latents = torch.zeros(
            (1,  # batch
            6,  # faces
            4,  # latent channels
            64, # height
            64),# width
            dtype=torch.bfloat16,
            device=device_str,
        )

        #    (b) Fake timesteps: [0] on device_str
        fake_timesteps = torch.tensor([0], dtype=torch.long, device=device_str)

        #    (c) Fake text embedding: [1, hidden_size] on device_str
        hidden_size = self.pipe.text_encoder.config.hidden_size  # e.g. 768
        fake_txt_emb = torch.zeros(
            (1, 1, hidden_size),       # 3-D: [B=1, seq_len=1, hidden_size]
            dtype=torch.bfloat16,
            device=device_str,
        )
        
        # Warmup forward pass with explicit device checking
        print("Running warmup forward pass...")
                
        # 5) Run one forward pass so all kernels get JIT‐compiled (no_grad + bf16) to make the pipe warm
        try: 
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=self.model_dtype):
                # RIGHT: get from config so it matches exactly
                _ = self.pipe.model(
                    fake_latents,        # [1,6,4,64,64]
                    fake_timesteps,      # [1]
                    fake_txt_emb         # [1, hidden_size]
                )
            # Block until GPU work actually finishes, so timing is accurate
            torch.cuda.synchronize(device_str)
        except RuntimeError as e:
            print(f"✗ Warmup failed: {e}")
            # Debug: Print device locations
            print("Debugging device locations:")
            print(f"  fake_latents.device: {fake_latents.device}")
            print(f"  fake_timesteps.device: {fake_timesteps.device}")
            print(f"  fake_txt_emb.device: {fake_txt_emb.device}")
            print(f"  model device: {next(self.pipe.model.parameters()).device}")
            if hasattr(self.pipe.model, 'positional_encoding'):
                if hasattr(self.pipe.model.positional_encoding, 'face_emb'):
                    face_emb_device = next(self.pipe.model.positional_encoding.face_emb.parameters()).device
                    print(f"  face_emb device: {face_emb_device}")
            raise

        # --------------------------------------------
        # From here on, the pipeline is “pre‐warmed” and every submodule is guaranteed
        # to be on device_str.  You can proceed with the rest of __init__.
        # --------------------------------------------

        self.num_gpus    = self.accelerator.num_processes                 
        # samples seen per optimizer.step()
        self.global_batch_size = self.micro_batch * self.num_gpus * self.acc_steps  # = 8*8*4 = 256 
        self.eval_cycle = self.config["eval_cycle"] # self.global_batch_size * self.eval_every_n_global_batch  # eval cycle in global batches
        # optional offline-wandb
        if "use_wandb" not in self.config:
            wandb.init(dir=str(self.logs_dir/"wandb"),
                       project=self.config.get("wandb_project","cubediff"),
                       name   =self.config.get("wandb_run_name", "cubediff_"+datetime.datetime.now().strftime("%H%M%S")),
                       mode="offline",
                       config=dict(self.config))

        # Maintain an EMA copy of your adapter weights to stabilize both loss and samples.
        # Maintain an exponential moving average of your inflated-attention weights and use that for sampling. 
        # EMA smooths out per-step jitter and almost always yields lower and steadier losses, as well as cleaner images.
        self.ema_model: Optional[CubeDiffModel] = None

        # build model / VAE / text-enc once
        self.use_safetensors = True
        self.setup_model(pretrained_model_name)
        print(f"trainer.py - CubeDiffTrainer - init - setup_model {pretrained_model_name} done\n")
        
        from torchvision.models import VGG16_Weights

        def safe_vgg16(pretrained=True, **kwargs):
            # 1) try loading from local torchvision cache
            try:
                # new-style API will look in ~/.cache/torch/hub/checkpoints first
                return models.vgg16(weights=VGG16_Weights.DEFAULT, **kwargs)
            except Exception:
                # 2) fallback to original pretrained=True (which may download)
                return models.vgg16(pretrained=pretrained, **kwargs)

        self.perceptual_net = safe_vgg16().features[:16].eval().to(torch.float32)

        for p in self.perceptual_net.parameters():
            p.requires_grad = False
        self.l1 = torch.nn.L1Loss()
        print(f"trainer.py - CubeDiffTrainer - init - vgg16 loading done")
        self.global_iter = 0

        # get CLIP socre for pano image and text prompt 
        # 1) Try loading CLIPProcessor from local cache, else fall back to Hub
        CACHE = Path.home() / ".cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots"
        print(f"train.py - Try loading CLIPProcessor from local cache snapshots : {CACHE}")
        def _find_snapshot(use_safetensors: bool) -> str:
            want = "model.safetensors" if use_safetensors else "pytorch_model.bin"
            for rev in CACHE.iterdir():
                if (rev / want).exists():
                    return str(rev)
            raise FileNotFoundError(f"no snapshot with {want}")
        
        model_path = _find_snapshot(self.use_safetensors)
        print(f"trainer.py - Try loading CLIPProcessor from local cache subfolder: {model_path}")
        try:
            self.clip_processor = CLIPProcessor.from_pretrained(
                model_path, # "openai/clip-vit-base-patch32",
                local_files_only=True,
                use_fast=False, #True,    
            )            
            print("trainer.py - ✅ Loaded CLIPProcessor from local cache")
        except Exception as e:
            print(f"trainer.py - ⚠️ CLIPProcessor cache load failed ({e}), downloading…")
            self.clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32",
                use_fast=False, # True,    
            )
        # 2) Try loading CLIPModel (safetensors) from cache, else fall back
        try:
            self.clip_model = CLIPModel.from_pretrained(
                model_path, # "openai/clip-vit-base-patch32",
                use_safetensors=self.use_safetensors,
                local_files_only=True
            )
            print(f"trainer.py - ✅ Loaded CLIPModel from local cache (safetensors) use_safetensors as {self.use_safetensors}")
        except Exception as e:
            print(f"trainer.py - ⚠️ CLIPModel cache load failed ({e}), downloading…")
            self.clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                use_safetensors=self.use_safetensors
            )
        
        # 3) Finalize
        self.clip_model = self.clip_model.eval().to(self.accelerator.device)

        # prepare a place to record
        self.clip_score_history = []

        # ─── helper to count samples in a .tar ──────────────────────────
        def count_tar_samples(path: str) -> int:
            cnt = 0
            with tarfile.open(path, "r") as tar:
                for m in tar.getmembers():
                    # assume each .pt is one sample
                    if m.name.endswith(".pt"):
                        cnt += 1
            return cnt
        
        self.train_size = count_tar_samples(self.config["dataset"])
        print(f"  ▶ Train samples: {self.train_size}")
        
        self.val_size = count_tar_samples(self.config["val_dataset"])
        print(f"  ▶ Val   samples: {self.val_size}")
                
        self.total_updates = self.config["max_train_steps"]
        world_size      = self.accelerator.num_processes
        samples_per_rank = self.train_size // world_size
        self.batch_size      = self.config["batch_size"] # 8
        batch_num_per_rank = samples_per_rank // self.batch_size
        micro_batch_size = self.config["batch_size"] * world_size # e.g. 8 * 8 = 64
        accum_steps = self.config["gradient_accum_steps"]  # e.g. 4
        accum_batch_size_per_rank = accum_steps * self.batch_size # e.g. 4 * 8 = 32
        sample_size_per_update = accum_batch_size_per_rank * world_size  # e.g. 32 * 8 = 256
        epochs_per_trn_datasize = math.ceil(self.train_size/sample_size_per_update)# e.g. 630 / 256 = 2.46, so 3 "epoch"
        self.total_steps  = self.total_updates # use the max_steps directly # e.g. 30000 / 2.46 = 12195.12 steps ( 30000 // 3 = 10000 "epoch" steps)
        self.total_required_samples = self.total_steps * sample_size_per_update # e.g. 10000 * 256 = 2,560,000 samples totally

        print(f"trainer.py - CubeDiffTrainer - __init__ - train data - \
                self.train_size is {self.train_size}, max_train_steps (total_updates) is {self.total_updates}, \
                epochs_per_trn_datasize is {epochs_per_trn_datasize}, total_steps is {self.total_steps},  \
                self.total_required_samples is {self.total_required_samples}, \
                world_size is {world_size}, batch_size is {self.batch_size}, micro_batch_size is {micro_batch_size}, \
                batch_num_per_rank is {batch_num_per_rank}, samples_per_rank is {samples_per_rank}, \
                accum_steps is {accum_steps}, accum_batch_size_per_rank is {accum_batch_size_per_rank}, \
                sample_size_per_update_step (actual batch size for gradients update) is {sample_size_per_update}, eval_cycle is {self.eval_cycle}\n")
        
        
        # LR schedule -------------------------------------
        from diffusers import get_scheduler  # for linear warmup + cosine
        
        # follow CubeDiff Sec.5.1.1: 30k update steps, warm up first 10k
        warmup     = int(self.config.get("warmup_ratio", 0.1) * self.total_steps)
        self.lr_scheduler = get_scheduler(
            name="cosine",                    # no “_with_restarts”
            optimizer=self.optimizer,
            num_warmup_steps = warmup,  # 10% warmup
            num_training_steps = self.total_steps,
        )
        
    def setup_model(self, pretrained_model_name: str):
        torch.backends.cuda.matmul.allow_tf32 = True  # if you want speed > accuracy
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.enabled = True

        self.vae          = self.pipe.vae.eval().requires_grad_(False)
        
        # 2) Noise scheduler (bfloat16)
        # self.noise_scheduler = DDPMScheduler.from_pretrained(
        #     pretrained_model_name, 
        #     subfolder="scheduler",
        #     prediction_type="v_prediction"  # Specify v-prediction objective
        # )

        # self.noise_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
        #     pretrained_model_name,
        #     subfolder="scheduler",
        #     prediction_type="v_prediction"
        # )
        # self.noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(
        #     pretrained_model_name,
        #     subfolder="scheduler",
        #     prediction_type="v_prediction"
        # )

        # self.noise_scheduler = DDIMScheduler.from_pretrained(
        #     pretrained_model_name,
        #     subfolder="scheduler",
        #     prediction_type="v_prediction",
        #     use_safetensors=self.use_safetensors
        # )

        # ── revert to ε-prediction DDIM ──
        # self.noise_scheduler  = DDIMScheduler.from_pretrained(
        #     pretrained_model_name,
        #     subfolder="scheduler",
        #     prediction_type="epsilon",      # ← explicit ε-prediction
        #     clip_sample=False,               # keep raw model output
        # )
        
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name,
            subfolder="scheduler",
            prediction_type="epsilon",   # UNet was trained to predict ε
            clip_sample=False,
        )
        
        for k, v in self.noise_scheduler.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self.noise_scheduler, k, v.to(torch.bfloat16))

        # self.model = self.pipe.model
        
        # That will reproduce the CubeDiff authors’ ∼17 M trainable parameters and ensure meaningful gradients.
        # to enable full-rank tuning on only the inflated-attn layers
        # Keep **only** the inflated‐attention layers trainable:
        print(f"trainer.py - CubeDiffTrainer - setup_model - UNet model parameters set to requires_grad for inflated-attn layers only\n")
        # for name, p in self.model.base_unet.named_parameters():
        #     print(f"trainer.py - CubeDiffTrainer - setup_model - self.model.base_unet.named_parameters: {name} - requires_grad {p.requires_grad} - shape {p.shape}")
        #     # p.requires_grad = ("inflated_attn" in name)
        #     p.requires_grad = ("is_inflated" in name)
        #     if p.requires_grad:
        #         print(f"trainer.py - CubeDiffTrainer - setup_model - self.model.base_unet.named_parameters: {name} is trainable")

        # 1) Build a mapping from each Parameter’s id → its full name
        param_to_name = {
            id(p): name
            for name, p in self.model.base_unet.named_parameters()
        }

        # 2) First, freeze _all_ parameters
        for p in self.model.base_unet.parameters():
            p.requires_grad = False

        # 3) Now un-freeze every parameter that lives under an inflated module
        inflated_count = 0
        for module in self.model.base_unet.modules():
            if hasattr(module, "is_inflated") and module.is_inflated:
                print(f"[TRAINABLE] trainer.py - setup_model - inflated module {module.__class__.__name__} at {module} with is_inflated = True")
                for p in module.parameters():
                    # recurse=False ensures we only grab this module's own weights,
                    # but if you want submodules too, drop recurse=False.
                    p.requires_grad = True
                    inflated_count += 1
                    # optional debug print
                    name = param_to_name.get(id(p), "<unknown>")
                    print(f"[TRAINABLE] trainer.py - setup_model - inflated layer {name}, requires_grad = True")

        print(f">>> trainer.py - setup_model - {inflated_count} parameters un-frozen for fine-tuning (inflated modules)")

        # quick sanity check for trainable parameters of unet
        total, trainable = 0, 0
        for p in self.model.base_unet.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        print(f"👉 trainer.py - CubeDiffTrainer - setup_model - UNet model parameters set to requires_grad for Full-rank tuning: {trainable/1e6:.2f}M / {total/1e6:.1f}M params")

        self.vae = self.vae.to(dtype=self.model_dtype)
        replace_group_norms(self.vae.encoder, in_place=True)
        replace_group_norms(self.vae.decoder, in_place=True)

        # Cast scheduler tensors
        for k, v in self.noise_scheduler.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self.noise_scheduler, k, v.to(self.model_dtype))

        print(f"trainer.py - CubeDiffTrainer- CubeDiff Model components cast to {self.model_dtype}")

        # Enable gradient checkpoints on the U-Net backbone only and circular padding
        # if diffusers>=0.18, which shards attention internals to slash peak usage.
        #    (saves ~30–40% memory at the cost of ~10–20% extra compute)
        print(f"trainer.py - CubeDiffTrainer - CubeDiff Model enabled gradient checkpointing\n")
        self.model.base_unet.enable_gradient_checkpointing()
        
        print(f"trainer.py - CubeDiffTrainer - CubeDiff Model enabled xformers\n")

        # a direct incompatibility between torch.compile and the explicit xformers memory-efficient attention mechanism 
        # torch.compile attempts to trace the model's operations to create an optimized computation graph. 
        # However, it doesn't know how to handle the low-level, custom CUDA kernel from the xformers library
        # The modern and recommended approach is to let torch.compile manage the attention optimization. 
        # It's designed to automatically use the best available backend, including FlashAttention, through PyTorch 2.0's native scaled_dot_product_attention
        # self.model.base_unet.enable_xformers_memory_efficient_attention()
        # self.model = self.model.to(memory_format=torch.channels_last) # Switch to channels-last-3d memory format
        for m in self.model.base_unet.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.padding_mode = "circular"

        tot   = sum(p.numel() for p in self.model.parameters())
        train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f">>> Training {len(trainable_params)} tensors, totalling {sum(p.numel() for p in trainable_params):,} params")
        self.optimizer = bnb.optim.AdamW8bit(
            params        = trainable_params,
            lr            = self.config["learning_rate"],
            betas         = (0.9, 0.95),
            weight_decay  = self.config.get("weight_decay", 0.01),
        )
        print(f"trainer.py - CubeDiffTrainer - setup_model done - Total params {tot/1e6:.1f}M — trainable {train/1e6:.2f}M")

        # ➌ create EMA model
        import copy
        self.ema_model = copy.deepcopy(self.model)
        for p in self.ema_model.parameters():
            p.requires_grad = False  # EMA is not trained directly

        # print("🚀 Compiling model with torch.compile for 2-4x speedup...")
        # self.model = torch.compile(
        #     self.model,
        #     mode="max-autotune",     # Aggressive optimization
        #     fullgraph=True,          # Compile entire model as one graph
        #     backend="inductor"       # Use PyTorch's native backend
        # )

        # --------------------------------------------------------
        # After all model setup is complete, add compilation
        time_s_comp = time.time()
        try:
            print("🚀 Compiling model with optimized settings for L4 GPUs...")
            
            # Compile with settings optimized for training stability
            self.model = torch.compile(
                self.model,
                mode="reduce-overhead",  # Better for training loops
                fullgraph=False,         # More stable with complex models
                backend="inductor",
                options={
                    "triton.cudagraphs": True,      # Enable CUDA graphs
                    "max_autotune": True,           # Aggressive optimization
                    "epilogue_fusion": True,        # Fuse operations
                    "max_autotune_gemm": True,      # Optimize matrix ops
                }
            )
            print("trainer.py - CubeDiffTrainer - setup_model - ✅ Model compilation successful - expect 2-4x speedup")
            
        except Exception as e:
            print(f"⚠️ Model compilation failed: {e}")
            print("Training will continue without compilation...")
            # Don't raise - continue without compilation

        print(f"trainer.py - CubeDiffTrainer - setup_model - torch.compile done, cost {time.time()-time_s_comp:0.4f} seconds")
        # --------------------------------------------------------

    # --------------------------------------------------
    # New dataloader creator  (latents, no JPEG) 
    # --------------------------------------------------    
    def build_dataloaders(self):
        print(f"trainer.py - CubeDiffTrainer - Building dataloaders with config: {self.config}")

        try:            
            self.train_dataloader = get_dataloader(
                wds_path=self.config["dataset"],
                batch_size=self.config["batch_size"],
                data_size=self.train_size,
                num_workers=self.config["num_workers"],
                is_eval=False,
                rank=self.accelerator.local_process_index,  # pass local rank for distributed training
                world_size=self.accelerator.num_processes,  # pass world size for distributed training
            )
            print(f"Train dataloader created successfully")
            
            # Verify the dataloader by trying to get one batch
            try:
                batch_iter = iter(self.train_dataloader)
                first_batch = next(batch_iter)
                print(f"Successfully loaded a sample batch with keys: {first_batch.keys()}")
            except StopIteration:
                print(f"Warning: Dataloader is empty, no samples found - at rank {self.accelerator.local_process_index}")
            except Exception as e:
                print(f"Warning: Failed to load sample batch: {e}")
            
            if "val_dataset" in self.config:
                self.val_dataloader = get_dataloader(
                    self.config["val_dataset"],   
                    batch_size=self.config["eval_batch_size"], # larger eval batch size halves or quarter the number of forward calls.
                    data_size=self.val_size,
                    num_workers=self.config["num_workers"],
                    is_eval=True,
                    rank=self.accelerator.local_process_index,  # pass local rank for distributed training
                    world_size=self.accelerator.num_processes,  # pass world size for distributed training
                )
                print("Val dataloader created successfully")

                # Verify the val loader can yield at least one batch
                try:
                    batch_iter = iter(self.val_dataloader)
                    first_batch = next(batch_iter)
                    print(f"Successfully loaded a val sample batch with keys: {first_batch.keys()}")
                except StopIteration:
                    print(f"Warning: Val dataloader is empty at rank {self.accelerator.local_process_index}")
                except Exception as e:
                    print(f"Warning: Failed to load val sample batch: {e}")
            else:
                self.val_dataloader = None
            
            # Call list(raw_val_dataloader) before accelerator.prepare(...) and store those mini-batches in a local (per-rank) list.
            self.val_dataloader_list = list(self.val_dataloader) if self.val_dataloader is not None else []
            print(f"trainer.py - CubeDiffTrainer - build_dataloaders - val_dataloader_list created as {len(self.val_dataloader_list)} batches per rank\n")
            # Set random seed
            set_seed(self.config.get("seed", 42))

            print("Preparing model and dataloader with accelerator")            
            # NOTE: the first input arguments ordering must be the model, then optimizer, then lr_scheduler in order to get self.model as a deepspeed engine .
            if self.val_dataloader is not None:
                self.model, self.optimizer, self.lr_scheduler, self.train_dataloader, self.val_dataloader = self.accelerator.prepare(
                    self.model,
                    self.optimizer,
                    self.lr_scheduler,
                    self.train_dataloader,
                    self.val_dataloader,                   
                )
            else:
                self.model, self.optimizer, self.lr_scheduler, self.train_dataloader = self.accelerator.prepare(
                    self.model,
                    self.optimizer,
                    self.lr_scheduler,
                    self.train_dataloader,                    
                )
            print(f"trainer.py - cubedifftrainer - build_datyaloader - Model type after prepare: {type(self.model)}")
            print(f"trainer.py - cubedifftrainer - build_datyaloader - Has save_checkpoint: {hasattr(self.model, 'save_checkpoint')}")
            print(f"trainer.py - cubedifftrainer - build_datyaloader - Accelerator distributed type: {self.accelerator.distributed_type}")
            print(f"trainer.py - cubedifftrainer - build_datyaloader -  DeepSpeed plugin: {self.accelerator.state.deepspeed_plugin}")

            self.model.to(self.accelerator.device)
            # allow Conv2d layers to use NHWC layout under the hood
            self.model.to(memory_format=torch.channels_last)

            # Warm up every inflated-attn adapter so DDP thinks they’re all used. Otherwise, accelerator will complain about unused parameters. 
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=self.model_dtype):
                # B=1 image, F=6 faces, C0=4 latent channels, H=W=64 (or whatever your VAE latent size is)
                dummy_latents = torch.randn((1, 6, 4, 64, 64),
                                            device=self.accelerator.device,
                                            dtype=self.model_dtype)
                dummy_timesteps = torch.zeros(1,
                                            device=self.accelerator.device,
                                            dtype=torch.long)
                # per-image text embeddings: [B, seq_len, D]
                dummy_text = torch.randn((1, 77, 768),
                                        device=self.accelerator.device,
                                        dtype=self.model_dtype)
                # invoke your model’s forward so every inflated module runs once
                _ = self.model(latents=dummy_latents,
                            timesteps=dummy_timesteps,
                            encoder_hidden_states=dummy_text)

            # Move components to device (preserve FP32 for perceptual_net)
            # U-Net, VAE, and text encoder benefit from BF16 for speed/memory.
            # Move components to device (preserve FP32 for perceptual_net)
            device = self.accelerator.device
            print(f"Moving components to {device}; keeping perceptual_net in FP32")
            self.perceptual_net = self.perceptual_net.to(device)                  # keep FP32
            print("Dataloader building completed successfully")
        except Exception as e:
            print(f"Error building dataloaders: {e}")
            import traceback
            traceback.print_exc()
            raise


    def boundary_loss(self, x):
        """
        x: either
        [B, F, C, H, W]   (5-D)
        or
        [B*F, C, H, W]    (4-D)
        Returns a scalar: average L1 across all seams.
        """
        # detect & reshape into [B, F, C, H, W]
        if x.dim() == 5:
            B, Face, C, H, W = x.shape
            x5 = x
        elif x.dim() == 4:
            Bf, C, H, W = x.shape
            Face = self.config.get("num_faces", 6)
            B = Bf // Face
            x5 = x.view(B, Face, C, H, W)
        else:
            raise ValueError(f"boundary_loss: unexpected tensor dim {x.dim()}")

        losses = []
        for i in range(Face):
            # right edge of face i vs left edge of face (i+1)%F
            r = x5[:, i, :, :, -1]      # [B, C, H]
            l = x5[:, (i+1) % Face, :, :,  0]  # [B, C, H]
            losses.append(self.l1(r, l))
        return sum(losses) / Face
    
    # --------------------------------------------------
    #  Training loop  (shortened & adapted to latent input)
    # --------------------------------------------------
    def train(self):
        self.build_dataloaders()
        train_iter = iter(self.train_dataloader)    # ← one persistent iterator, shared by all gstep loops        
        
        # the LR will decay smoothly over exactly the number of updates you actually perform,         
        warmup     = int(self.config.get("warmup_ratio", 0.1) * self.total_steps)

        # ─── DEBUG: print out LR values ───
        if self.accelerator.is_main_process:
            from diffusers import get_scheduler
        
            # 1) Create a tiny dummy parameter & optimizer, so we can build an independent scheduler
            dummy_param = torch.nn.Parameter(torch.zeros(1))
            dummy_opt   = torch.optim.AdamW(
                params=[dummy_param],
                lr=self.config["learning_rate"],
                betas=(0.9, 0.95),
                weight_decay=self.config.get("weight_decay", 0.01),
            )
        
            # 2) Instantiate a fresh scheduler with the same hyper-params
            
            tmp_sched = get_scheduler(
                name="cosine",                    # same as your real scheduler
                optimizer=dummy_opt,
                num_warmup_steps=warmup,          # same warmup count
                num_training_steps=self.total_steps,
            )
            # import copy
            # tmp = copy.deepcopy(self.lr_scheduler)
            # print(f"debug LR - step 0: {tmp.get_last_lr()[0]:.3e}")  # before any .step()

            # 3) Now we can safely step through and print ALL the LR values:
            print(f"debug LR - step  0: {tmp_sched.get_last_lr()[0]:.3e}")
            for i in range(1, self.total_steps + 1):
                tmp_sched.step()   # advances the schedule
                print(f"debug LR - step {i:>3}: {tmp_sched.get_last_lr()[0]:.3e}")

            # print learning rates for all steps 
            # for i in range(1, self.total_steps + 1):
            #     tmp.step()
            #     # if i in check:
                # print(f"step {i:>3}: {tmp.get_last_lr()[0]:.3e}")
        # ------------------------------------------------

        # ───── DDP warm-up ─────
        # run one dummy forward on the wrapped (DDP) model so every param is “used”; otherwise, DDP (accelerator) will complain about unused parameters.
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=self.model_dtype):
            # 1 image, 6 faces, 4 latent channels, 64×64 spatial
            dummy_lat = torch.randn((1, 6, 4, 64, 64),
                                     device=self.accelerator.device,
                                     dtype=self.model_dtype)
            dummy_ts  = torch.zeros((1,),
                                     device=self.accelerator.device,
                                     dtype=torch.long)
            # 77 tokens, 768-dim CLIP
            dummy_txt = torch.randn((1, 77, 768),
                                     device=self.accelerator.device,
                                     dtype=self.model_dtype)
            _ = self.model(
                latents=dummy_lat,
                timesteps=dummy_ts,
                encoder_hidden_states=dummy_txt,
            )
        # ────────────────────────
        
        print(f"trainer.py - CubeDiffTrainer - train - lr_scheduler created - self.total_steps {self.total_steps}, warmup {warmup}, warmup_ratio {self.config.get('warmup_ratio', 0.3)}\n")
        # now wrap both optimizer and scheduler exactly once
        # self.optimizer, self.lr_scheduler = self.accelerator.prepare(
        #     self.optimizer,
        #     self.lr_scheduler
        # )        

        sample_prompts        = ["A beautiful mountain lake at sunset with snow-capped peaks"]
        print(f"trainer.py - CubeDiffTrainer - train - sample_prompts is {sample_prompts}\n")

        gstep             = 0 # udpated when sync_gradients=True, i.e. once per optimizer step
        train_losses, val_losses = [], []
        ep_g_start_tm = time.time()
        real_sample_size_per_rank = 0
        rank = self.accelerator.local_process_index  
        self.total_processed_samples = 0
        batch_indx = -1
        acc_counter = 0
        while gstep < self.total_steps:
            g_start_tm = time.time()
            print(f"▶️ rank {rank} - Starting gstep {gstep}/{self.total_steps}")
            self.accelerator.wait_for_everyone()  # wait for all processes to be ready
            # Pull exactly one batch from train_iter. If it is exhausted, recreate it.
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)
            batch_indx += 1
            real_batch_size = batch["latent"].size(0)
            real_sample_size_per_rank += real_batch_size
            acc_counter += 1
            start_tm_accum = time.time()
            with self.accelerator.accumulate(self.model):
                latents = batch["latent"].to(self.accelerator.device, dtype=self.model_dtype)              # [B,6,4,64,64]
                txt_emb = batch["encoder_hidden_states"].to(self.accelerator.device, dtype=self.model_dtype)
                mask = batch["attention_mask"].to(self.accelerator.device)
                print(f"\tRank {rank} - gstep {gstep} - batch_indx {batch_indx} - lat shape is {latents.shape}, txt_emb shape is {txt_emb.shape}, mask shape is {mask.shape}")
                
                # ──────────────────────────────────────────────────────────────────────────────
                # 1) Sample timesteps with your squared‐weight distribution
                # ──────────────────────────────────────────────────────────────────────────────
                # 1) number of diffusion steps (e.g. 1 000)
                T = self.noise_scheduler.config.num_train_timesteps
                # weights = torch.arange(1, T + 1, device=self.accelerator.device, dtype=torch.float32) ** 2
                # probs   = weights / weights.sum()
                # sample with replacement according to probs
                # timesteps = torch.multinomial(probs, self.batch_size, replacement=True)
                
                # new: uniform sampling over all timesteps
                # 2) uniformly choose one timestep t ∈ {0,…,T–1} for each item in the batch
                timesteps = torch.randint(0, T, (real_batch_size,), device=self.accelerator.device)

                # ──────────────────────────────────────────────────────────────────────────────
                # 2) Create noise and compute the noised latents x_t
                # ──────────────────────────────────────────────────────────────────────────────
                # sample “pure” Gaussian noise, same shape as your latents
                noise     = torch.randn_like(latents)
                # call the scheduler’s forward‐noise helper:
                noisy_lat = self.noise_scheduler.add_noise(latents, noise, timesteps)

                # Classifier-Free Guidance (CFG) drops (§4.5):
                #   10% drop text, 10% drop image, 80% full cond
                bs = txt_emb.size(0)
                rnd = torch.rand(bs, device=txt_emb.device)
                # drop text embeddings
                drop_txt = rnd < 0.1
                if drop_txt.any():
                    txt_emb[drop_txt] = 0
                # drop image conditioning mask
                drop_img = (rnd >= 0.1) & (rnd < 0.2)
                if drop_img.any():
                    # zero out mask[i] = 0, that sample truly runs without text conditioning—completing the multimodal CF dropout.
                    mask[drop_img] = 0
                # print(f"\tRank {rank} - gstep {gstep} - batch_indx {batch_indx} - drop_txt {drop_txt.sum()} samples, drop_img {drop_img.sum()} samples")
                temp_s_time_pred = time.time()

                # ──────────────────────────────────────────────────────────────────────────────
                # 3) Forward pass through your inflated UNet (ε-prediction mode)
                # ────────────────────────────────────────────────────────────────────────────
                with self.accelerator.autocast(): # 2025-7-4 cost 6+ secs on 4 A100, 8bs, 1 accumu; 2025-06-29 5 secs on 8 L4, 8 batch size, gradient accumulation step 4
                    # Convert to channels_last for better tensor core utilization; using the most efficient memory format for GPUs
                    # noisy_lat = noisy_lat.to(memory_format=torch.channels_last)
                    model_out = self.model(
                        latents=noisy_lat,
                        timesteps=timesteps, # shape: (batch,)
                        encoder_hidden_states=txt_emb,
                        encoder_attention_mask=mask,
                    )
                    # unwrap the actual noise prediction tensor
                    pred = model_out["sample"]
                    
                    temp_e_time_pred = time.time()
                    print(f"\tRank {rank} - gstep {gstep} - batch_indx {batch_indx} - noisy_lat shape is {noisy_lat.shape}, prediction done, cost {temp_e_time_pred - temp_s_time_pred:.4f} seconds")
                    # Align spatial dims if needed
                    if noise.size(2) != pred.size(2): # cost - 0 secs
                        noise = noise[:, :, : pred.size(2), :, :]
                    
                    # 4) Train to predict the *added* Gaussian noise; Compute loss only once, in bf16
                    # loss = F.mse_loss(pred.float(), noise.float(), reduction="mean").float()
                    loss = F.mse_loss(pred, noise, reduction="mean")
                
                print(f"\trank{rank} - gstep {gstep} - batch_indx {batch_indx} - check noise std mean - noise std:", noise.std().item(), "  mean:", noise.mean().item())
                print(f"\trank{rank} - gstep {gstep} - batch_indx {batch_indx} - check pred std mean  - pred std:", pred.std().item(), "  mean:", pred.mean().item())

                # collect loss 
                temp_s_time_4 = time.time()
                self.accelerator.backward(loss)  # 2025-7-4 cost 12+ secs on 4 A100, 8bs; 2025-6-29 compute gradients cost - 12 - 15 secs for 8 L4, 8 bs, accumu 4
                temp_e_time_4 = time.time()
                print(f"\tRank {rank} - gstep {gstep} - batch_indx {batch_indx} - backward done, cost {temp_e_time_4 - temp_s_time_4:.2f} seconds - loss type is {type(loss)} shape is {loss.shape}, loss is {loss} ")
                
                # Only once per actual optimizer step (i.e. when sync_gradients=True):
                # if self.accelerator.sync_gradients:
                if acc_counter % self.acc_steps == 0:
                    # here all replicas have the fully synchronized, accumulated grads, see each adapter’s true gradient (after accumulation + all‐GPU sync) exactly once per update, 
                    # rather than on every micro‐batch.
                    # Clip gradients: This prevents any single batch from sending the loss curve on a wild ride.
                    # Gradient clipping (1.0) is standard in diffusion training to prevent occasional large updates, smoothing out “bounces” .
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                    # Apply optimizer update
                    self.optimizer.step() # 2025-6-29 0.01 - 0.02 secs for 8 L4, 8 bs, 4 accumu
                    # Increment per‐rank step counter due to weights update
                    gstep += 1
                    
                    # Step the LR scheduler (once per real update)
                    self.lr_scheduler.step() # 0 secs
                    
                    # EMA update (decay=0.999)
                    # Maintain an exponential moving average of your inflated-attention weights and use that for sampling. 
                    # EMA smooths out per-step jitter and almost always yields lower and steadier losses, as well as cleaner images.
                    with torch.no_grad():
                        for p, ema_p in zip(self.model.parameters(), self.ema_model.parameters()):
                            ema_p.mul_(0.999).add_(p, alpha=1-0.999)

                    # logging
                    # increment by the number of samples that just contributed including accumulated batches
                    self.total_processed_samples += self.global_batch_size 

                    # DeepSpeed ZeRO-2 shards both optimizer states and model parameters across ranks. 
                    # A proper checkpoint requires each rank to write out its local shard and then synchronize via a barrier to ensure consistency.
                    if (gstep>0) and (gstep%self.eval_cycle==0):
                        temp_s_time_chkp = time.time() # 2025-6-29 1.9-3.8 secs for 8 L4, 8 bs, 4 accumu
                        ckpt_folder = self.output_dir / "ds_ckpt" / f"deepspeed_ckpt_step{gstep}"
                        ckpt_folder.mkdir(exist_ok=True, parents=True)
                        
                        # 1) Check if we're actually using DeepSpeed
                        unwrapped_model = self.accelerator.unwrap_model(self.model)
                        
                        # 2) Determine the engine type and save accordingly
                        if hasattr(unwrapped_model, 'save_checkpoint'):
                            # We have a DeepSpeed engine
                            # print(f"trainer.py - train - [rank {rank}] → DeepSpeed engine detected: {type(unwrapped_model)}")
                            unwrapped_model.save_checkpoint(
                                save_dir=str(ckpt_folder),
                                tag=f"{gstep}",
                                client_state={"step": gstep},
                            )
                        else:
                            # Fall back to standard accelerate checkpoint saving
                            self.accelerator.save_state(str(ckpt_folder))
                            print(f"trainer.py - train - [rank {rank}] → Using standard checkpoint (not DeepSpeed): {type(unwrapped_model)}, Standard checkpoint saved to: {ckpt_folder}")
                        
                        print(f"[rank {rank}] → Checkpoint saved, cost {time.time() - temp_s_time_chkp:.4f} seconds")

                    # Clear gradients for the next accumulation
                    self.optimizer.zero_grad()
                        
                    # percent of update‐steps done
                    pct_updates = gstep / self.total_steps * 100.0

                    # percent of total samples seen (out of the 30k×batch_size stated in paper)
                    pct_samples = (self.total_processed_samples / self.total_required_samples) * 100.0

                    # equivalent epochs over the 700-image “tiny” set
                    eq_epochs = self.total_processed_samples / self.train_size
                    
                    # All‐reduce the *scalar* loss (local mean over the micro-batch)
                    if (gstep>0) and (gstep%self.eval_cycle==0):
                        temp_s_time_gloss = time.time() # 2025-6-29 0.003 secs for 8 L4, 8 bs, 4 accumu
                        global_loss = self.accelerator.reduce(loss, reduction="mean")
                        temp_e_time_gloss = time.time()
                        print(f"\tRank {rank} - gstep {gstep} - batch_indx {batch_indx} - global_loss done, cost {temp_e_time_gloss - temp_s_time_gloss:.4f} seconds, global_loss shape is {global_loss.shape}")
                    
                    #–– collect train loss ––#
                    if (gstep>0) and (gstep%self.eval_cycle==0) and self.accelerator.is_main_process:
                        avg_train_loss = global_loss.item()  # only one float moves
                        train_losses.append((self.total_processed_samples, avg_train_loss))
                        
                        lr = self.lr_scheduler.get_last_lr()[0] 
                        print(
                            f"lr and sample and epoch progress: rank {rank} — gstep {gstep}/{self.total_steps} ▶"
                            f" {pct_updates:5.1f}% updates | Samples ▶ {pct_samples:5.1f}%"
                            f" ({self.total_processed_samples}/{self.total_required_samples}) |"
                            f" Equivalent Epochs seen ▶ {eq_epochs:.2f}"
                            f" LR at step {gstep}: {lr:.3e}"
                            f" total_samples = {self.total_processed_samples}"
                        )
                    # ----------------- end of training loop for the current step ----------------------
                    g_end_tm = time.time() # 2025-6-5 8 L4, 8 bs, 1 accum - 30+ secs per step; 2025-6-29 8 L4, 8 bs, 4 accum - 26+ secs per step
                    print(f"Rank {rank} - for current udpate step - gstep {gstep} - processed batch {batch_indx} - real_sample_size_per_rank {real_sample_size_per_rank} samples done, cost {g_end_tm - g_start_tm:.4f} seconds\n")
                    
                self.global_iter += 1
            # if gstep % 5 == 0:  # Every 5 steps
            #     torch.cuda.empty_cache()
            #     gc.collect()
                
            end_tm_accum = time.time() # 2025-6-5 23-30+ secs for 8 L4, 8 bs, 1 accumu
            print(f"\tRank {rank} - gstep {gstep} - out of accumulate - batch_indx {batch_indx} cost {end_tm_accum - start_tm_accum:.2f} seconds, before waiting for all ranks to finish accumulate ...")
            
            self.accelerator.wait_for_everyone()
            print(f"\tRank {rank} - gstep {gstep} - out of accumulate - batch_indx {batch_indx} after waiting for all ranks to finish accumulate, cost {time.time() - end_tm_accum:.2f} seconds, real_sample_size_per_rank is {real_sample_size_per_rank} samples")

            # ---- evaluate --------------------------------------------------------------   
            if (gstep>0 and gstep%self.eval_cycle==0) or (self.total_processed_samples>= self.total_required_samples):
                # --------- plot loss and generate panorama by rank=0 ------------------------------------
                # update sample count on the main rank (only it tracks & saves losses)
                if self.accelerator.is_main_process: 
                    # self._plot_loss_curves(self.total_processed_samples, train_losses, val_losses, gstep, self.total_steps)
                    temp_chkp_path = self.output_dir / "ds_ckpt" / f"deepspeed_ckpt_step{gstep}" 
                    if os.path.exists(temp_chkp_path):
                        # generate panorama from current checkpoints 
                        temp_s_time = time.time() # cost 115-120+ secs on A100 and 30+ secs on L4; 2025-6-5 22-29+ secs on 8 L4, 8 bs, 1 accumu; 2025-6-29 40-42 secs on 8 L4, 8 bs, 4 accumu
                        self.generate_samples(rank, sample_prompts, gstep, self.total_processed_samples)
                        temp_e_time = time.time()
                        print(f"\tRank {rank} - gstep {gstep} - batch_indx {batch_indx} - generate_samples done - cost {temp_e_time - temp_s_time:.2f} seconds")  
                        
                # all ranks evaluate, but only rank 0 saves and returns the results
                temp_s_time_eval_all_ranks = time.time() # 2025-6-29 5.4 secs for 8 L4, 8 bs, 4 accumu; 2025-6-5 0.22+ secs for 8 L4, 8 bs, 1 accumu
                val_loss = self.evaluate(rank, gstep) # this cost some time
                temp_e_time_eval_all_ranks = time.time()
                print(f"\tRank {rank} - gstep {gstep} - evaluate done, cost {temp_e_time_eval_all_ranks - temp_s_time_eval_all_ranks:.4f} seconds, val_loss is {val_loss:.4f}")
                if self.accelerator.is_main_process:  
                    val_losses.append((self.total_processed_samples, val_loss))
                    self._plot_loss_curves(self.total_processed_samples, train_losses, val_losses, self.clip_score_history, gstep, self.total_steps)
            
        ep_g_end_tm = time.time()
        print(f"Rank {rank} - gstep {gstep} - out of the training loop - total_processed_samples is {self.total_processed_samples} -  all updates steps done, cost {ep_g_end_tm - ep_g_start_tm:.4f} seconds\n")

        # ----------------- save final U-net weights ----------------------
        if self.accelerator.is_main_process:
            path = self.output_dir / f"final_unet_adapter_model_gstep{gstep}.bin"
            # pull the real underlying model out of the Accelerator wrapper
            unwrapped = self.accelerator.unwrap_model(self.model)
            # grab just the U-Net’s weights; pulls exactly the adapter weights that have been fine-tuned.
            unet_sd = unwrapped.base_unet.state_dict()
            # move everything to CPU and full precision
            unet_sd = {k: v.detach().cpu().float() for k, v in unet_sd.items()}
            torch.save(unet_sd, path)
            print(f"\nRank {rank} - trainer.py - CubeDiffTrainer - train - move everything to CPU and full precision - ✔ saved U-Net adapter weights to {path}")
            
            # ───────────────────────────────────────────────────────────────
            # after all training, plot train & val curves
            try:
                # unpack
                steps_tr, loss_tr = zip(*train_losses) if train_losses else ([],[])
                steps_va, loss_va = zip(*val_losses)   if val_losses   else ([],[])
                # coerce EVERY element to a host‐side Python scalar
                steps_tr = [
                    int(s.item()) if torch.is_tensor(s) else int(s)
                    for s in steps_tr
                ]
                loss_tr  = [
                    float(l.item()) if torch.is_tensor(l) else float(l)
                    for l in loss_tr
                ]
                steps_va = [
                    int(s.item()) if torch.is_tensor(s) else int(s)
                    for s in steps_va
                ]
                loss_va  = [
                    float(l.item()) if torch.is_tensor(l) else float(l)
                    for l in loss_va
                ]

                plt.figure(figsize=(6,4))
                plt.plot(steps_tr, loss_tr, label="train")
                if steps_va:
                    plt.plot(steps_va, loss_va, label="val")
                plt.xlabel("step")
                plt.ylabel("MSE loss")
                plt.legend()
                plt.title(f"Loss curves for {gstep} gstep and {self.total_processed_samples} samples")
                plt.tight_layout()
                out = self.output_dir / f"loss_curve_gstep{gstep}_total_processed_ds{self.total_processed_samples}.png"
                plt.savefig(out)
                print(f"✔ loss curves saved to {out}")
            except Exception as e:
                print(f"⚠ could not plot loss curves: {e}")
    # ───────────────────────────────────────────────────────────────
     
    # --------------------------------------------------
    # ✱✱✱  generate panorama after N steps for progress  ✱✱✱
    # --------------------------------------------------
    def generate_samples(self, rank, prompts, step, total_processed_samples):
        """
        Updated generate_samples:
        - Only rank 0 runs the denoising (no all-gather).
        - Load adapter weights via DeepSpeed ZeRO: accelerator.load_state().
        - Single BF16 generate() call on GPU 0.
        - Save one PNG per prompt (no redundant images).
        - The HF CubeDiffPipeline self.pipe(...) internally handles building
            [B,6,4,H,W] inputs and calling Model.forward under bf16.
        - We save exactly one PNG per prompt on rank 0.
        """

        # Only main process (rank 0) does anything; other ranks immediately return.
        if not self.accelerator.is_main_process:
            return

        print(f"\trank {rank} → gstep {step} - generate_samples (total_processed_samples={total_processed_samples}, prompts={prompts})")

        # --------------------------------------------------------
        # 1) Load the ZeRO‐sharded checkpoint for this step
        #    (DeepSpeed saved one shard per rank into `deepspeed_ckpt_step{step}`).
        # --------------------------------------------------------
        ckpt_folder = self.output_dir / "ds_ckpt" / f"deepspeed_ckpt_step{step}"
        print(f"\t\t[rank {rank}] loading ZeRO‐sharded checkpoint from {ckpt_folder} ...")
        if not ckpt_folder.exists():
            raise FileNotFoundError(f"Expected checkpoint folder not found: {ckpt_folder}")

        load_start = time.time() 
        # This will automatically rehydrate only the weights
        # into GPU 0 from its shard. Other ranks will load their own shards
        # (but since they return immediately, only rank 0 truly uses them).
        # self.accelerator.load_state(str(ckpt_folder))

        # Unwrap to get the underlying DeepSpeedEngine
        # If self.model is already a DeepSpeedEngine, this is a no-op.
        # engine = self.accelerator.unwrap_model(self.model)
        # Now `engine` is guaranteed to be the DeepSpeedEngine instance that
        # has `load_checkpoint` and `save_zero_checkpoint`.

        # ✅ self.model *is* the DeepSpeedEngine from accelerator.prepare(...)
        # engine = self.model  

        # ── step 1: load only the 16-bit weights shards ──
        # Check if we're using DeepSpeed by examining the wrapped model type
        if hasattr(self.model, 'load_checkpoint'):
            # self.model IS the DeepSpeed engine
            print(f"\t\ttrainer.py - generate_sampes - [rank {rank}] Loading DeepSpeed checkpoint...")
            try:
                load_dir, client_state = self.model.load_checkpoint(
                    str(ckpt_folder),            # positional load_dir
                    tag=str(step),               # your checkpoint tag
                    load_module_strict=True,     # enforce exact name matches
                    load_optimizer_states=False, # skip optimizer
                    load_lr_scheduler_states=False, # skip scheduler
                )
                assert client_state["step"] == step, f"Loaded checkpoint gstep {client_state['step']} but expected gstep {step}"
                print(f"\t\ttrainer.py - generate_sampes - [rank {rank}] DeepSpeed checkpoint loaded successfully")
                assert load_dir.startswith(ckpt_dir), f"trainer.py - generate_samples() - Expected load_dir under {ckpt_dir} but got {load_dir}"

                assert client_state["step"] == step, f"trainer.py - generate_samples() - Loaded checkpoint gstep {client_state['step']} but expected gstep {step}"
            except Exception as e:
                print(f"\t\ttrainer.py - generate_sampes - [rank {rank}] DeepSpeed load failed: {e}, falling back to accelerate load")
                self.accelerator.load_state(str(ckpt_folder))
        else:
            # We're using DDP - use standard accelerate loading
            print(f"\t\ttrainer.py - generate_sampes - [rank {rank}] Loading standard checkpoint...")
            self.accelerator.load_state(str(ckpt_folder))
            print(f"\t\ttrainer.py - generate_sampes - [rank {rank}] Standard checkpoint loaded successfully")
        

        # client_state["step"] should equal step if you passed it above
        ckpt_dir = str(ckpt_folder)
        
        # ── step 2: pull out the raw nn.Module for HF pipeline usage ──
        raw_model = self.accelerator.unwrap_model(self.model)
        # raw_model is your CubeDiffModel with freshly loaded weights.

        # ── step 3: inject into the pipeline and run generate() ──

        # assert isinstance(self.pipe.scheduler, DDIMScheduler)
        assert isinstance(self.pipe.scheduler, DDPMScheduler)
        assert self.pipe.scheduler.prediction_type == "epsilon"

        self.pipe.unet = raw_model.base_unet
        self.pipe.model = raw_model
        # (any other swaps, e.g. ema, scheduler buffers, etc.)

        # ➎ Swap in EMA weights for inference
        ema = self.accelerator.unwrap_model(self.ema_model)
        self.pipe.model = ema
        self.pipe.unet  = ema.base_unet
        self.pipe.model.eval()

        # ❶ Tell the pipeline’s scheduler to expect v-predictions:
        # self.pipe.scheduler = self.noise_scheduler
        # self.pipe.scheduler.config.prediction_type = "v_prediction"
        # self.pipe.scheduler.prediction_type = "v_prediction"

        # ── Move scheduler buffers to GPU so step() will actually run
        for attr in ("alphas_cumprod","alphas_cumprod_prev","betas","one"):
            buf = getattr(self.pipe.scheduler, attr, None)
            if isinstance(buf, torch.Tensor):
                setattr(self.pipe.scheduler, attr, buf.to(self.accelerator.device))

        # 🔍 Debug: make sure this actually took
        print(f"\t\t[rank 0] >>>> Scheduler class: {self.pipe.scheduler.__class__}")
        print(f"\t\t[rank 0] >>>> prediction_type: {self.pipe.scheduler.config.prediction_type!r}")
        # note: .timesteps only exists *after* set_timesteps(…) is called in .generate()
        # but you can at least check the scheduler has the right max steps
        # print(f"\t\t[rank 0] >>>> num_train_timesteps: {self.pipe.scheduler.num_train_timesteps}")
        print(f"\t\t[rank 0] >>>> num_train_timesteps: {self.pipe.scheduler.config.num_train_timesteps}")

        torch.cuda.synchronize()
        # 2025-6-5 cost 0.5-1 secs on 8 L4, 8 bs, 1 accum; 2025-6-29 cost 0.6-1 secs on 8 L4, 8 bs, 4 accum
        print(f"\t\t[rank 0] ZeRO load_state completed in {time.time() - load_start:.3f}s")

        # --------------------------------------------------------
        # 2) Build a list of prompts, one per rank (for compatibility),
        #    but we will only denoise on rank 0.  If the user gave a single
        #    string, we replicate it; if they gave a list of length 1, replicate;
        #    if they gave a list of length = world_size, use directly.
        # --------------------------------------------------------
        world_size = self.accelerator.num_processes

        # Normalize `prompts` into a list
        if isinstance(prompts, str):
            prompt_list = [prompts] * world_size
        elif isinstance(prompts, (list, tuple)) and len(prompts) == 1:
            prompt_list = [prompts[0]] * world_size
        elif isinstance(prompts, (list, tuple)) and len(prompts) == world_size:
            prompt_list = list(prompts)
        else:
            raise ValueError(
                f"generate_samples: Expected 1 or {world_size} prompts, but got {len(prompts)}"
            )

        local_prompt = prompt_list[0]  # only rank 0 will actually use it

        # --------------------------------------------------------
        # 3) Run the diffusion pipeline exactly once on rank 0
        #    Under BF16, no_grad. Pre-warmed pipeline in __init__ ensures minimal JIT overhead.
        # --------------------------------------------------------
        gen_start = time.time() # 2025-6-5 cost 40+ secs on 8 L4, 8 bs, 1 accum; 2025-6-5 19-28+ secs on 8 L4, 8 bs, 1 accum; 2025-6-29 cost 39+ secs on 8 L4, 8 bs, 4 accum
        # make sure the model is in eval
        self.pipe.model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pano = self.pipe.generate(
                    local_prompt,
                    guidance_scale=7.5,
                    num_inference_steps=30,
                    device=self.accelerator.device
            )
        torch.cuda.synchronize()
        gen_time = time.time() - gen_start
        print(f"\t\t[rank {rank}] denoised in {gen_time:.3f}s")

        # Retrieve PIL.Image
        pano_image = Image.fromarray((pano*255).astype("uint8"))

        # compute CLIP similarity to the prompt
        inputs = self.clip_processor(
            text=[local_prompt],
            images=[pano_image],
            return_tensors="pt",
            padding=True
        ).to(self.accelerator.device)

        with torch.no_grad():
            clip_out = self.clip_model(**inputs)
            img_emb = clip_out.image_embeds    # (1, D)
            txt_emb = clip_out.text_embeds     # (1, D)
        clip_sim = torch.nn.functional.cosine_similarity(img_emb, txt_emb, dim=-1).item()

        # record it alongside your losses
        self.clip_score_history.append((self.total_processed_samples, clip_sim))

        print(f"\t\ttrainer.py - [rank {rank}] - gstep {step} - generate_samples - total_processed_samples: {self.total_processed_samples}, CLIP sim: {clip_sim:.4f}")

        # 4) Save the PNG to disk (exactly once)
        out_path = self.images_dir / f"gstep{step}_samples{total_processed_samples}.png"
        pano_image.save(out_path)
        print(f"\t\t[rank {rank}] saved panorama → {out_path}")
        # 2025-6-5 cost 40+ secs on 8 L4, 8 bs, 1 accum; 2025-6-5 29+ secs on 8 L4, 8 bs, 1 accum; 2025-6-29 cost 40+ secs on 8 L4, 8 bs, 4 accum
        print(f"\trank {rank} → generate_samples finished (wall-clock ~{time.time() - load_start:.3f}s)")
    
        # --------------------------------------------------------
        # 5) NEW: Clean up old checkpoints while keeping current one
        # --------------------------------------------------------
        if self.accelerator.is_main_process:
            # Store current checkpoint path for preservation
            current_ckpt = f"deepspeed_ckpt_step{step}"
            
            # Find all checkpoint directories
            ds_ckpt_dir = self.output_dir / "ds_ckpt"
            if ds_ckpt_dir.exists():
                try:
                    # Get all checkpoint folders matching the pattern
                    ckpt_folders = [d for d in ds_ckpt_dir.iterdir() 
                                if d.is_dir() and d.name.startswith("deepspeed_ckpt_step")]
                    
                    
                    def extract_step_num(folder_name):
                        try:
                            return int(folder_name.replace("deepspeed_ckpt_step", ""))
                        except ValueError:
                            return -1
                    
                    # ckpt_folders.sort(key=lambda x: extract_step_num(x.name))
                    ckpt_folder = [extract_step_num(x.name) for x in ckpt_folders]

                    print(f"[rank {rank}] ➖ found {len(ckpt_folders)} checkpoints in {ds_ckpt_dir}, current_ckpt is {current_ckpt}")
                    # Delete all but the current (latest) checkpoint
                    deleted_count = 0
                    for ckpt_folder_path in ckpt_folders:
                        print(f"[rank {rank}] ➖ checking checkpoint folder: {ckpt_folder_path.name}, current_ckpt is {current_ckpt}")
                        if ckpt_folder_path.name != current_ckpt:
                            try:
                                shutil.rmtree(ckpt_folder_path)
                                deleted_count += 1
                                print(f"[rank {rank}] ➖ removed old checkpoint {ckpt_folder_path.name}")
                            except Exception as e:
                                print(f"[rank {rank}] ⚠️ failed to delete {ckpt_folder_path}: {e}")
                    
                    print(f"[rank {rank}] ✅ Checkpoint cleanup completed: kept {current_ckpt}, deleted {deleted_count} old checkpoints")
                    
                except Exception as e:
                    print(f"[rank {rank}] ⚠️ Error during checkpoint cleanup: {e}")
            
            print(f"trainer.py - [rank {rank}] → generate_samples completed successfully with checkpoint management")
        
    # ---------------------------------------------------------------------
    # New: run a full pass over val_dataloader and return avg loss.
    # and decode_latents_to_rgb + self.perceptual_net    
    
    def evaluate(self, rank, gstep):
        """
        Distributed evaluation across all GPUs without dropping samples or deadlocking.

        1) Count exactly how many batches this rank’s val_dataloader will yield.
        2) Use accelerator.gather(...) to find max_batches across ranks.
        3) Loop exactly max_batches times:
            – If this rank has a “real” batch at batch_idx, consume it.
            – Otherwise, contribute zero (no inference).
        4) Perform one blocking all‐reduce at the end to compute global MSE.
        """

        # 0) If no validation loader, return NaN immediately
        if self.val_dataloader is None:
            print(f"rank {rank} – evaluate() returning NaN because val_dataloader is None")
            return float("nan")

        # -----------------------------
        # 1) Count actual number of batches (local_batches) for this rank
        # -----------------------------
        # Rather than using ceil(samples_per_rank / batch_size) (which can be off),
        # we force Python to build a list of all batches so we know exactly how many there are.
        # Because validation is relatively small, converting DataLoader to a list is acceptable.
        val_batches = self.val_dataloader_list # list(self.val_dataloader)  # This exhausts the DataLoader once, may be []
        local_batches = len(val_batches)         # Exactly how many batches this rank has, could be 0
        # Now recreate an iterator over the same list, so we can consume it on the fly.
        val_iter = iter(val_batches) # safe, even if val_batches=[]

        # Logging for debugging:
        print(
            f"\trank {rank} - gstep {gstep} trainer.py - evaluate() - "
            f"Determined local_batches = {local_batches} (actual DataLoader length)"
        )

        # -----------------------------
        # 2) Gather local_batches across all ranks to compute max_batches
        # -----------------------------
        if local_batches == 0:
            print(f"Rank {self.accelerator.local_process_index}: validation dataloader is empty")
            batch_counts = torch.tensor([0], device=self.accelerator.device)
        else:
            batch_counts = torch.tensor([local_batches], device=self.accelerator.device)

        print(f"rank {rank} – evaluate - before accelerator.wait_for_everyone")
        start_e_tm = time.time()  
        self.accelerator.wait_for_everyone()  # make sure all ranks hit gather together
        print(f"rank {rank} – evaluate - after accelerator.wait_for_everyone, cost {time.time() - start_e_tm:.4f} seconds")

        # Important: Accelerate’s `accelerator.gather` (not all_gather) collects a tensor
        # from every rank and concatenates along dim=0. Here each tensor is shape (1,).
        # *** Ensure every rank calls gather() exactly once, even if local_batches is 0. ***
        print(f"rank {rank} – evaluate - calling accelerator.gather(batch_counts={batch_counts.tolist()}")
        start_g_tm = time.time()
        gathered = self.accelerator.gather(batch_counts)
        print(f"rank {rank} – evaluate - after accelerator.gather, cost {time.time()-start_g_tm:.4f} seconds, → gathered = {gathered.tolist()}")

        # After this call, `gathered` is a tensor of shape (world_size,) [
        #   local_batches_rank0, local_batches_rank1, ..., local_batches_rankN ]
        max_batches = int(gathered.max().item())
        # print(f"rank {rank} – evaluate - max_batches = {max_batches}")
        
        # Optional detailed logging:
        val_size_total = self.val_size
        world_size     = self.accelerator.num_processes
        batch_size     = self.config["eval_batch_size"]
        print(
            f"\trank {rank} - gstep {gstep} trainer.py - evaluate() - config: "
            f"val_size_total={val_size_total}, world_size={world_size}, "
            f"batch_size={batch_size}, local_batches={local_batches}, max_batches={max_batches}\n"
        )

        # -----------------------------
        # 3) Loop exactly max_batches times
        # -----------------------------
        total_se      = torch.tensor(0.0, device=self.accelerator.device)  # Sum of (loss * num_samples)
        total_samples = torch.tensor(0,   device=self.accelerator.device)  # Sum of num_samples

        self.model.eval()
        # Debug print to show start of evaluation
        print(
            f"\trank {rank} - gstep {gstep} trainer.py - evaluate() - "
            f"Looping over max_batches={max_batches} steps\n"
        )

        temp_s_time_eval_ba = time.time()  # Start timing the evaluation loop
        for batch_idx in range(max_batches):
            # temp_s_time_eval_b = time.time()  
            if batch_idx < local_batches:
                try:
                    # This rank *does* have a “real” batch to consume at this index:
                    batch = next(val_iter)  # Guaranteed not to raise StopIteration,
                                        # because we know local_batches is len(val_batches).
                    # DEBUGGING: Print batch shapes
                    # print(f"EVAL DEBUG batch shapes:")
                    # print(f"  latent: {batch['latent'].shape}")
                    # print(f"  encoder_hidden_states: {batch['encoder_hidden_states'].shape}")
                    # print(f"  attention_mask: {batch['attention_mask'].shape}")

                    lat     = batch["latent"].to(self.accelerator.device, dtype=self.model_dtype)
                    mask    = batch["attention_mask"].to(self.accelerator.device)
                    txt_emb = batch["encoder_hidden_states"].to(self.accelerator.device,
                                                                dtype=self.model_dtype)

                    # Check if txt_emb needs reshaping for single item
                    if txt_emb.shape[0] == 1 and lat.shape[0] > 1:
                        # print(f"EVAL DEBUG: Expanding txt_emb from {txt_emb.shape} to match latent batch size {lat.shape[0]}")
                        # Expand single embedding to match latent batch size
                        if txt_emb.ndim == 2:
                            txt_emb = txt_emb.expand(lat.shape[0], -1)
                        elif txt_emb.ndim == 3:
                            txt_emb = txt_emb.expand(lat.shape[0], -1, -1)
                            
                except StopIteration:   
                    # This should never happen, but if it does, we skip this batch.
                    print(
                        f"\trank {rank} - gstep {gstep} - evaluate() - "
                        f"Batch index {batch_idx} raised StopIteration, skipping."
                    )
                    continue
                except Exception as e:
                    print(
                        f"\trank {rank} - gstep {gstep} - evaluate() - "
                        f"Batch index {batch_idx} raised an unexpected error: {e}, skipping."
                    )
                    continue

                # Add noise & sample timesteps ------------------------------
                # noise     = torch.randn_like(lat)
                # Use actual batch size from the latents tensor
                actual_batch_size = lat.shape[0]
                
                # mirror train() exactly, using scheduler.add_noise + get_velocity()
                # T = self.noise_scheduler.num_train_timesteps
                T = self.noise_scheduler.config.num_train_timesteps
                weights = torch.arange(1, T+1, dtype=torch.float32, device=self.accelerator.device) ** 2
                probs   = weights / weights.sum()
                # sample with replacement according to probs using ACTUAL batch size
                timesteps = torch.multinomial(probs, actual_batch_size, replacement=True)

                # sample noise + timesteps as before
                noise = torch.randn_like(lat)

                # 2) Build noised latents via the scheduler
                noisy_lat = self.noise_scheduler.add_noise(lat, noise, timesteps)
                # ---------------------------------------------------

                # Forward pass (no gradients, with mixed precision)
                temp_s_time_eval_p = time.time()
                with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    model_out = self.model(
                        latents=noisy_lat,
                        timesteps=timesteps,
                        encoder_hidden_states=txt_emb,
                        encoder_attention_mask=mask
                    )
                    pred = model_out["sample"]  # Extract the actual noise prediction tensor
                    # If the latent’s spatial size differs, trim the noise:
                    if noise.size(2) != pred.size(2):
                        noise = noise[:, :, :pred.size(2), :, :]

                    # batch_loss = F.mse_loss(pred.float(), noise.float(), reduction="mean")
                    batch_loss = F.mse_loss(pred, noise, reduction="mean")
                    
                    num_samples = lat.shape[0]
                temp_e_time_eval_p = time.time()
                print(
                    f"\trank {rank} - gstep {gstep} - evaluate() - in loop - "
                    f"batch_idx {batch_idx}, batch_loss={batch_loss:.4f}, num_samples={num_samples}, "
                    f"cost {temp_e_time_eval_p - temp_s_time_eval_p:.4f} seconds\n"
                )

                total_se      += batch_loss * num_samples
                total_samples += num_samples

                # print(
                #     f"\trank {rank} - gstep {gstep} - evaluate() - in loop - has real data - batch_idx {batch_idx}, "
                #     f"batch_loss={batch_loss:.4f}, num_samples={num_samples}\n"
                # )

            else:
                # This rank has *no* real data at this index → contribute zero.
                # We skip inference entirely to save time.
                print(
                    f"\trank {rank} - gstep {gstep} - evaluate()- in loop - no real data - batch_idx {batch_idx}, "
                    f"no real data (zero loss contribution)\n"
                )
                # => total_se and total_samples remain unchanged
            temp_e_time_eval_b = time.time()
            # print(
            #     f"\trank {rank} - gstep {gstep} - evaluate() - in loop - "
            #     f"batch_idx {batch_idx} done, "
            #     f"cost {temp_e_time_eval_b - temp_s_time_eval_b:.4f} seconds, "
            # )
        temp_e_time_eval_ba = time.time()  # 2025-6-5 cost 0.22 secs on 8 L4, 8 trn bs, 1 accum; 2025-6-29 cost 4.5-5.5 secs on 8 L4, 8 eval bs, 4 accum
        print(
            f"\trank {rank} - gstep {gstep} - evaluate() - all batches done - "
            f"Finished looping over max_batches={max_batches} cost "
            f"{temp_e_time_eval_ba - temp_s_time_eval_ba:.4f} seconds\n"
        )

        # -----------------------------
        # 4) Single blocking reduce to get global sums
        # -----------------------------
        pack = torch.stack([total_se, total_samples], dim=0)
        # This is a blocking collective; all ranks must call this exactly once.
        print(f"\trank {rank} – evaluate - calling accelerator.reduce(pack=[{total_se.item()}, {total_samples.item()}])")
        temp_s_time_eval_r = time.time()
        reduced_pack = self.accelerator.reduce(pack, reduction="sum")
        temp_e_time_eval_r = time.time() # 2025-6-5 cost 0.0003 secs on 8 L4, 8 bs, 1 accum; 2025-6-29 cost 0.0003 secs on 8 L4, 8 bs, 4 accum
        print(f"\trank {rank} – evaluate - after accelerator.reduce → reduced_pack = {reduced_pack.tolist()}, cost {temp_e_time_eval_r - temp_s_time_eval_r:.4f} seconds\n")    

        # 5) Compute final average MSE
        global_se      = reduced_pack[0]
        global_samples = reduced_pack[1]
        if global_samples.item() > 0:
            avg_mse = (global_se / global_samples).item()
            # print(
            #     f"\trank {rank} - gstep {gstep} - evaluate() - "
            #     f"global_samples={global_samples.item()}, val_size={self.val_size}, avg_mse={avg_mse:.6f}\n"
            # )
        else:
            avg_mse = float("nan")
            print(
                f"\trank {rank} - gstep {gstep} - evaluate() - final global_samples=0 "
                f"No samples processed. Returning NaN\n"
            )

        self.model.train()
        return avg_mse

    def _plot_loss_curves(self,
        total_processed_samples: int,
        train_losses, val_losses, clip_scores,
        gstep: int,
        total_updates: int):
        """
        train_losses: list of (samples, loss)
        val_losses:   list of (samples, loss)
        clip_scores:  list of (samples, cosine_similarity)
        gstep:        current number of update steps done so far
        total_updates: the total number of update steps you plan to run
        """
        # unpack
        steps_tr, loss_tr = zip(*train_losses) if train_losses else ([],[])
        steps_va, loss_va = zip(*val_losses)   if val_losses   else ([],[])
        steps_cs, score_cs = zip(*clip_scores) if clip_scores else ([],[])

        # coerce EVERY element to a host‐side Python scalar
        steps_tr = [
            int(s.item()) if torch.is_tensor(s) else int(s)
            for s in steps_tr
        ]
        loss_tr  = [
            float(l.item()) if torch.is_tensor(l) else float(l)
            for l in loss_tr
        ]
        steps_va = [
            int(s.item()) if torch.is_tensor(s) else int(s)
            for s in steps_va
        ]
        loss_va  = [
            float(l.item()) if torch.is_tensor(l) else float(l)
            for l in loss_va
        ]
        steps_cs = [
            int(s.item()) if torch.is_tensor(s) else int(s)
            for s in steps_cs
        ]
        score_cs  = [
            float(l.item()) if torch.is_tensor(l) else float(l)
            for l in score_cs
        ]

        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(steps_tr, loss_tr, label="train")
        if steps_va:
            ax.plot(steps_va, loss_va, label="val")

        # ── plot CLIP sim on a new right‐hand y‐axis ──
        ax2 = ax.twinx()
        ax2.plot(steps_cs, score_cs, "--", label="clip sim", color="green")
        ax2.set_ylabel("CLIP cosine sim")
        ax2.legend(loc="lower right")

        ax.set_title(
            f"Loss up to {total_processed_samples} samples\n"
            f"(update steps so far: {gstep}/{total_updates})"
        )
        ax.set_xlabel("samples processed")
        ax.set_ylabel("loss")
        ax.legend()

        # ─── mark the current gstep on the samples axis ───
        if 1 <= gstep <= len(steps_tr):
            sample_at_gstep = steps_tr[gstep-1]
            # vertical line at that sample
            ax.axvline(sample_at_gstep, color="black", linestyle="--", alpha=0.6)
        else:
            sample_at_gstep = None

        # ─── now add a twin‐x to label the gstep ───
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        if sample_at_gstep is not None:
            ax2.set_xticks([sample_at_gstep])
            ax2.set_xticklabels([f"gstep={gstep}"])
        else:
            # fallback: just show start and end
            ax2.set_xticks([ax.get_xlim()[0], ax.get_xlim()[1]])
            ax2.set_xticklabels([f"gstep=0", f"gstep={gstep}"])
        ax2.set_xlabel("update steps")

        plt.tight_layout()
        out = self.output_dir / f"loss_curve_up_to_total_processed_samples.png"
        plt.savefig(out)
        plt.close(fig)
        print(f"✔ _plot_loss_curves saved loss curve → {out}")

# EOF -------------------------------------------------------------------------
