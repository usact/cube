# cl/data/build_tiny_set_parallel.py
# TRUE PARALLEL VERSION: CPU and GPU work simultaneously
# 2025-6-14 based on commit : 81d0c4a (https://github.com/Jinxu-Ding-A3024329_bestbuy/llm-cv-pano-cubediff/blob/81d0c4a48ae77faed3b241de5fb257e310e43949/cl/data/polyhaven/build_tiny_set.py)

import multiprocessing as mp

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    # already set in this interpreter session
    pass

# --- ENSURE PYTORCH USES FILE‐SYSTEM IPC (no /tmp mkstemp) ---
import torch.multiprocessing as tmp_mp
tmp_mp.set_sharing_strategy('file_system')

# from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
# from concurrent.futures import ThreadPoolExecutor as ProcessPoolExecutor
# from concurrent.futures import as_completed
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import tempfile

# send all of Python’s temp files (and thus torch._share_fd_cpu_) to ~/tmp
scratch = os.path.expanduser("~/tmp")
os.makedirs(scratch, exist_ok=True)
os.environ.update({
    "TMPDIR": scratch,
    "TEMP":   scratch,
    "TMP":    scratch,
})
tempfile.tempdir = scratch

import time
import torch
import numpy as np
import glob
import random
import cv2
import json
from pathlib import Path
# from typing import List, Tuple, Dict, Optional
import subprocess 
import threading
import queue
import signal
import sys
from pathlib import Path

from PIL import Image
from torchvision import transforms as T
from diffusers import AutoencoderKL

# For caption embeddings
from transformers import CLIPTokenizer, CLIPTextModel


# Import your existing functions
from cl.data.polyhaven.cubemap_builder import (
    load_hdr_image_proper, 
    aces_tonemap, 
    linear_to_srgb_proper
)
from cl.model.normalization import replace_group_norms

from cl.data.load_model_util import safe_from_pretrained

from cl.data.polyhaven.report_util import generate_final_report, generate_visual_samples

# ------------------------------------------------------------------------
# Preload everything exactly once at import time
# ------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load VAE
# _VAE = safe_from_pretrained(
#     AutoencoderKL.from_pretrained,
#     "runwayml/stable-diffusion-v1-5",
#     subfolder="vae",
#     torch_dtype=torch.float16,
# )
# _REPLACED = replace_group_norms(_VAE, in_place=True)
# _VAE = _VAE.to(DEVICE).eval()

# # “Warm up” so the first real batch doesn’t pay load latency
# with torch.no_grad():
#     dummy = torch.randn(4, 3, 512, 512, device=DEVICE, dtype=torch.float16)
#     _ = _VAE.encode(dummy).latent_dist.sample()
#     del dummy
#     torch.cuda.empty_cache()

# ------------------------------------------------------------------------
def convert_exr_to_png(args):
        exr_path, png_path = args
        hdr = load_hdr_image_proper(exr_path)
        if hdr is None:
            return f"SKIP {exr_path}"
        img = linear_to_srgb_proper(hdr)
        png = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(png_path, cv2.cvtColor(png, cv2.COLOR_RGB2BGR))
        return f"DONE {exr_path}"

# ───────────────────────────────────────────────────────────────────────────────
# Validation routines (formerly in build_tiny_set_old_2025_6_13.py)
# Inlined so this file is stand-alone.
# ───────────────────────────────────────────────────────────────────────────────

def validate_single_latent_file(latent_path: Path, split: str) -> dict:
    """Validate one latent: load, type, shape."""
    rel = f"{split}/{latent_path.name}"
    try:
        tensor = torch.load(latent_path, map_location="cpu")
    except Exception as e:
        return {"file": rel, "status": "FAILED", "stage": "load", "error": str(e)}
    if not isinstance(tensor, torch.Tensor):
        return {"file": rel, "status": "FAILED", "stage": "type", "error": type(tensor).__name__}
    if tensor.ndim != 4 or tensor.shape[1] != 4:
        return {"file": rel, "status": "FAILED", "stage": "shape", "shape": list(tensor.shape)}
    return {"file": rel, "status": "OK", "stage": "all"}


def validate_dataset_integrity(root: Path) -> dict:
    """
    Walk through root/{faces,latents,embeddings,panoramas,captions},
    validate files, and summarize.
    """
    stats = {
        "missing_faces": [],
        "missing_latents": [],
        "latent_errors": [],
        "missing_embs": [],
        "missing_panos": [],
        "missing_caps": [],
        "caption_lens": [],
    }
    # gather all panorama IDs from captions
    caps_dir = root/"captions"
    ids = [p.stem for p in caps_dir.glob("*.txt")]
    # record caption lengths
    for pid in ids:
        txt = (caps_dir/f"{pid}.txt").read_text().strip()
        stats["caption_lens"].append(len(txt))
    # check each ID
    for pid in ids:
        # faces & latents
        for i in range(6):
            face_path = root/"faces"/f"{pid}_face{i}.png"
            if not face_path.exists():
                stats["missing_faces"].append(f"{pid}_face{i}")
            latent_path = root/"latents"/f"{pid}_face{i}.pt"
            if not latent_path.exists():
                stats["missing_latents"].append(f"{pid}_face{i}")
            else:
                r = validate_single_latent_file(latent_path, root.name)
                if r["status"] != "OK":
                    stats["latent_errors"].append(r)
        # embeddings
        emb_path = root/"embeddings"/f"{pid}.pt"
        if not emb_path.exists():
            stats["missing_embs"].append(pid)
        # panoramas
        pano_path = root/"panoramas"/f"{pid}.png"
        if not pano_path.exists():
            stats["missing_panos"].append(pid)
    # summarize caption lengths
    if stats["caption_lens"]:
        cl = stats["caption_lens"]
        stats["caption_len_summary"] = {
            "min": min(cl), "max": max(cl), "avg": sum(cl)/len(cl)
        }
    return stats


def test_generated_faces(root: Path, n: int = 5) -> None:
    """Sample n panoramas, load their 6 faces to catch blank or corrupt images."""
    ids = [p.stem for p in (root/"captions").glob("*.txt")]
    sample = random.sample(ids, min(n, len(ids)))
    for pid in sample:
        for i in range(6):
            img = cv2.imread(str(root/"faces"/f"{pid}_face{i}.png"))
            assert img is not None and img.size > 0, f"Face {pid}_face{i} is empty"


def save_statistics(data: dict, out_file: Path) -> None:
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)


def update_global_stats(global_stats: dict, split: str, stats: dict) -> None:
    global_stats[split] = stats

# ───────────────────────────────────────────────────────────────────────────────
# sph2cube_face_fixed + process_single_panorama_fast + ParallelGPUWorker
# (unchanged from original)
# ───────────────────────────────────────────────────────────────────────────────

def sph2cube_face_fixed(pano: np.ndarray, face: str, size: int = 512, fov: float = 95.0) -> np.ndarray:
    """Generate a single cubemap face from equirectangular panorama."""
    H, W = pano.shape[:2]
    
    # Create coordinate grids
    i, j = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    
    # Convert to normalized coordinates [-1, 1] with FOV scaling
    fov_rad = np.radians(fov)
    scale = np.tan(fov_rad / 2)
    u = (2 * j / (size - 1) - 1) * scale
    v = (2 * i / (size - 1) - 1) * scale
    
    # Define face directions
    if face == 'front':
        x, y, z = u, -v, np.ones_like(u)
    elif face == 'right':
        x, y, z = np.ones_like(u), -v, -u
    elif face == 'back':
        x, y, z = -u, -v, -np.ones_like(u)
    elif face == 'left':
        x, y, z = -np.ones_like(u), -v, u
    elif face == 'top':
        x, y, z = u, np.ones_like(u), v
    elif face == 'bottom':
        x, y, z = u, -np.ones_like(u), -v
    else:
        raise ValueError(f"Unknown face: {face}")
    
    # Convert to spherical coordinates
    norm = np.sqrt(x**2 + y**2 + z**2)
    x_norm, y_norm, z_norm = x / norm, y / norm, z / norm
    
    lon = np.arctan2(x_norm, z_norm)
    lat = np.arcsin(np.clip(y_norm, -1, 1))
    
    # Convert to pixel coordinates
    px = ((lon + np.pi) / (2 * np.pi)) * (W - 1)
    py = ((np.pi/2 - lat) / np.pi) * (H - 1)
    
    px = np.clip(px, 0, W - 1)
    py = np.clip(py, 0, H - 1)
    
    # Bilinear interpolation
    x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
    x1, y1 = np.clip(x0 + 1, 0, W - 1), np.clip(y0 + 1, 0, H - 1)
    
    wx, wy = px - x0, py - y0
    
    face_img = np.zeros((size, size, pano.shape[2]), dtype=pano.dtype)
    
    for c in range(pano.shape[2]):
        I00, I01 = pano[y0, x0, c], pano[y0, x1, c]
        I10, I11 = pano[y1, x0, c], pano[y1, x1, c]
        
        I0 = I00 * (1 - wx) + I01 * wx
        I1 = I10 * (1 - wx) + I11 * wx
        face_img[:, :, c] = I0 * (1 - wy) + I1 * wy
    
    return face_img


def process_single_panorama_fast(args):
    """Fast CPU-only processing - prepare tensors for GPU queue + writes out face PNGs and captions."""
    exr_path, split, output_dir, pid = args
    
    try:
        # Load EXR
        exr = load_hdr_image_proper(exr_path)
        if exr is None:
            return False, f"Failed to load {exr_path}", None
        
        # Handle RGBA
        if len(exr.shape) == 3 and exr.shape[2] == 4:
            exr = exr[:, :, :3]
        
        # Generate cubemap faces
        face_names = ["front", "right", "back", "left", "top", "bottom"]
        face_size_target = 512
        face_size_with_overlap = int(face_size_target * 95 / 90)
        
        faces_linear = [sph2cube_face_fixed(exr, face=name, size=face_size_with_overlap) 
                       for name in face_names]
        
        # Process faces quickly
        to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
        
        split_dir = Path(output_dir) / split
        
        face_tensors = []
        latent_paths = []
        
        for idx, face_linear in enumerate(faces_linear):
            # Tone mapping and cropping
            tone_mapped = aces_tonemap(face_linear)
            srgb = linear_to_srgb_proper(tone_mapped)
            
            current_size = srgb.shape[0]
            margin = (current_size - face_size_target) // 2
            if margin > 0:
                srgb_cropped = srgb[margin:-margin, margin:-margin]
            else:
                srgb_cropped = cv2.resize(srgb, (face_size_target, face_size_target))
            
            # Save face image immediately
            rgb8 = (srgb_cropped * 255.0).astype(np.uint8)
            face_path = split_dir / "faces" / f"{pid}_face{idx}.png"
            os.makedirs(face_path.parent, exist_ok=True)
            Image.fromarray(rgb8).save(face_path)
            
            # Prepare tensor for GPU
            pil_img = Image.fromarray(rgb8)
            face_tensor = to_tensor(pil_img)
            face_tensors.append(face_tensor)
            
            # Latent path
            latent_path = split_dir / "latents" / f"{pid}_face{idx}.pt"
            latent_paths.append(latent_path)
        
        # Save caption
        caption = pid.replace("_", " ")
        caption_path = split_dir / "captions" / f"{pid}.txt"
        os.makedirs(caption_path.parent, exist_ok=True)
        caption_path.write_text(caption)
        
        return True, "", {
            'face_tensors': face_tensors,
            'latent_paths': latent_paths,
            'pid': pid
        }
        
    except Exception as e:
        return False, str(e), None




class ParallelGPUWorker:
    """Dedicated GPU worker that processes faces as they arrive from CPU."""
    
    def __init__(self, gpu_batch_size: int = 32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_batch_size = gpu_batch_size
        self.face_queue = queue.Queue(maxsize=500)  # Buffer for face tensors
        self.running = True
        
        # Statistics
        self.faces_processed = 0
        self.batches_processed = 0
        self.start_time = time.time()
        
        print(f"🔥 GPU Worker: batch size {gpu_batch_size}, device {self.device}")
        
        # Load VAE
        # self._load_vae()
        # Load VAE
        _VAE = safe_from_pretrained(
            AutoencoderKL.from_pretrained,
            "runwayml/stable-diffusion-v1-5",
            subfolder="vae",
            torch_dtype=torch.float16,
        )
        _REPLACED = replace_group_norms(_VAE, in_place=True)
        _VAE = _VAE.to(DEVICE).eval()

        # “Warm up” so the first real batch doesn’t pay load latency
        with torch.no_grad():
            dummy = torch.randn(4, 3, 512, 512, device=DEVICE, dtype=torch.float16)
            _ = _VAE.encode(dummy).latent_dist.sample()
            del dummy
            torch.cuda.empty_cache()
        self.vae = _VAE
        
        # Start worker thread
        self.gpu_thread = threading.Thread(target=self._gpu_worker_loop, daemon=True)
        self.gpu_thread.start()
    
    def add_faces(self, face_tensors, latent_paths):
        """Add face tensors to GPU processing queue."""
        for tensor, path in zip(face_tensors, latent_paths):
            self.face_queue.put((tensor, path))
    
    def _gpu_worker_loop(self):
        """Main GPU worker loop - processes faces continuously, draining until done."""
        batch_tensors = []
        batch_paths = []
        last_batch_time = time.time()
        
        # Keep running _or_ until we’ve drained every pending face in the queue
        while self.running or not self.face_queue.empty():
            try:
                # Try to get a face tensor
                try:
                    tensor, path = self.face_queue.get(timeout=0.5)
                    batch_tensors.append(tensor)
                    batch_paths.append(path)
                except queue.Empty:
                    # No new faces, check if we should process partial batch
                    pass
                
                # Process batch when full or on timeout
                current_time = time.time()
                should_process = (
                    len(batch_tensors) >= self.gpu_batch_size or
                    (batch_tensors and (current_time - last_batch_time) > 2.0) or
                    (not self.running and batch_tensors)
                )
                
                if should_process:
                    self._process_batch(batch_tensors, batch_paths)
                    batch_tensors.clear()
                    batch_paths.clear()
                    last_batch_time = current_time
                    
            except Exception as e:
                print(f"❌ GPU worker error: {e}")
                break
        
        # Process remaining batch
        if batch_tensors:
            self._process_batch(batch_tensors, batch_paths)
        
        print("✅ GPU worker finished")
    
    def _process_batch(self, tensors, paths):
        """Process a batch of face tensors."""
        if not tensors:
            return
        
        try:
            # Stack and encode
            batch_tensor = torch.stack(tensors).to(self.device, dtype=torch.float16)
            
            with torch.no_grad():
                latents = self.vae.encode(batch_tensor).latent_dist.sample() * 0.18215
            
            # Save latents
            latents_cpu = latents.cpu()
            for latent, path in zip(latents_cpu, paths):
                os.makedirs(path.parent, exist_ok=True)
                torch.save(latent.unsqueeze(0), path)
            
            self.faces_processed += len(tensors)
            self.batches_processed += 1
            
            # Progress update
            elapsed = time.time() - self.start_time
            rate = self.faces_processed / elapsed if elapsed > 0 else 0
            
            print(f"🔥 GPU: Batch {self.batches_processed} - "
                  f"{len(tensors)} faces → {self.faces_processed} total "
                  f"({rate:.1f} faces/sec)")
            
            # Cleanup
            del batch_tensor, latents, latents_cpu
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ GPU batch error: {e}")
            # Fallback: individual processing
            for tensor, path in zip(tensors, paths):
                try:
                    tensor_gpu = tensor.unsqueeze(0).to(self.device, dtype=torch.float16)
                    with torch.no_grad():
                        latent = self.vae.encode(tensor_gpu).latent_dist.sample() * 0.18215
                    os.makedirs(path.parent, exist_ok=True)
                    torch.save(latent.cpu(), path)
                    self.faces_processed += 1
                except Exception:
                    pass
    
    def shutdown(self):
        """Shutdown GPU worker."""
        self.running = False
        self.gpu_thread.join(timeout=10)
        
        return {
            'faces_processed': self.faces_processed,
            'batches_processed': self.batches_processed,
            'processing_time': time.time() - self.start_time
        }


def build_cubediff_parallel(
    exr_dir: str,
    output_dir: str,
    split_ratio: float = 0.9,
    seed: int = 42,
    cpu_workers: int = 12,
    gpu_batch_size: int = 32
):
    """
    TRUE PARALLEL processing: CPU and GPU work simultaneously.
    """
    
    print(f"🚀 PARALLEL CubeDiff Dataset Generation")
    print(f"   CPU workers: {cpu_workers}")
    print(f"   GPU batch size: {gpu_batch_size}")
    print(f"   Strategy: CPU feeds GPU continuously")
    
    # Setup signal handler for clean shutdown
    def signal_handler(sig, frame):
        print("\n🛑 Interrupt received, shutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Prepare files
    random.seed(seed)
    exr_paths = sorted(glob.glob(os.path.join(exr_dir, "*.exr")))
    if not exr_paths:
        print("❌ No EXR files found!")
        return
    
    print(f"📁 Found {len(exr_paths)} EXR files")
    random.shuffle(exr_paths)
    
    n_train = int(len(exr_paths) * split_ratio)
    train_items = [(path, "train", output_dir, Path(path).stem) for path in exr_paths[:n_train]]
    val_items = [(path, "val", output_dir, Path(path).stem) for path in exr_paths[n_train:]]

    all_items = train_items + val_items
    print(f"📊 Processing {len(all_items)} panoramas ({len(train_items)} train, {len(val_items)} val)")
    
    # Create directories
    for split in ["train", "val"]:
        for subdir in ["faces", "latents", "captions"]:
            os.makedirs(os.path.join(output_dir, split, subdir), exist_ok=True)
    
    # Start GPU worker
    gpu_worker = ParallelGPUWorker(gpu_batch_size=gpu_batch_size)
    
    # Process panoramas with CPU workers while GPU works in parallel
    print(f"🔄 Starting parallel CPU-GPU processing...")
    start_time = time.time()
    
    successful_count = 0
    total_faces_queued = 0
    
    try:
        ctx = mp.get_context("spawn")
        # spin up CPU workers, submit every EXR→face task to gpu_worker.add_faces(…)
        with ProcessPoolExecutor(max_workers=cpu_workers, mp_context=ctx) as executor:
            # Submit all CPU tasks
            futures = {executor.submit(process_single_panorama_fast, item): item 
                      for item in all_items}
            
            for future in as_completed(futures):
                item = futures[future]
                pid = item[3]
                
                try:
                    success, error, result = future.result()
                    
                    if success:
                        successful_count += 1
                        
                        # Immediately queue face tensors for GPU processing
                        face_tensors = result['face_tensors']
                        latent_paths = result['latent_paths']
                        gpu_worker.add_faces(face_tensors, latent_paths)
                        total_faces_queued += len(face_tensors)
                        
                        if successful_count % 25 == 0:
                            elapsed = time.time() - start_time
                            cpu_rate = successful_count / elapsed
                            eta = (len(all_items) - successful_count) / cpu_rate if cpu_rate > 0 else 0
                            
                            print(f"🔄 CPU Progress: {successful_count}/{len(all_items)} "
                                  f"({successful_count/len(all_items)*100:.1f}%) - "
                                  f"Rate: {cpu_rate:.1f} panoramas/sec - "
                                  f"GPU Queue: {total_faces_queued} faces - "
                                  f"ETA: {eta/60:.1f} min")
                    else:
                        print(f"❌ Failed {pid}: {error}")
                        
                except Exception as e:
                    print(f"❌ Exception processing {pid}: {e}")
    
    except KeyboardInterrupt:
        print("\n🛑 Received interrupt, shutting down...")
    
    # Shutdown GPU worker and get final stats
    print("🏁 CPU processing complete, waiting for GPU to finish...")
    gpu_stats = gpu_worker.shutdown()
    # Wait for the GPU thread to finish before moving on to embedding or validation:
    # important to wait for the GPU thread (VAE encoding) to finish, then go for caption embedding; 
    # otherwise CPU may fail to validate some latents, which are still being processed by the GPU worker.
    gpu_worker.gpu_thread.join() 

    total_time = time.time() - start_time
    
    print(f"\n✅ Parallel processing complete!")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   CPU processed: {successful_count}/{len(all_items)} panoramas ({successful_count/len(all_items)*100:.1f}%)")
    print(f"   GPU processed: {gpu_stats['faces_processed']} faces in {gpu_stats['batches_processed']} batches")
    print(f"   CPU rate: {successful_count/total_time:.1f} panoramas/sec")
    print(f"   GPU rate: {gpu_stats['faces_processed']/gpu_stats['processing_time']:.1f} faces/sec")
    
    print(f"\n🎯 Dataset ready for CubeDiff training!")
    print(f"   ✅ successful {successful_count * 6} face images")
    print(f"   ✅ {gpu_stats['faces_processed']} latent files")
    print(f"   ✅ successful {successful_count} caption files")

    # ───────────────────────────────────────────────────────────────────────────
    # POST-BUILD: Panorama export, caption embeddings, unified validation, TAR & smoke-test
    # ───────────────────────────────────────────────────────────────────────────
    root = Path(output_dir)
    exr_root = Path(exr_dir)

    # 1) Export full panoramas (HDR→sRGB→PNG)
    print(f"🌅 Exporting panoramas to {root}/train or val/panoramas as PNGs with parallel_dump_panos ...")
    start_time_pano = time.time()
    # for split in ("train","val"):
    #     pan_out = root/split/"panoramas"
    #     pan_out.mkdir(parents=True, exist_ok=True)
    #     for cap in (root/split/"captions").glob("*.txt"):
    #         pid = cap.stem
    #         exr_f = exr_root/f"{pid}.exr"
    #         if not exr_f.exists(): continue
    #         hdr = load_hdr_image_proper(str(exr_f))
    #         if hdr is None: continue
    #         img = linear_to_srgb_proper(hdr)
    #         png = (np.clip(img,0,1)*255).astype(np.uint8)
    #         cv2.imwrite(str(pan_out/f"{pid}.png"), cv2.cvtColor(png,cv2.COLOR_RGB2BGR))


    

    def parallel_dump_panos(root: Path,
                            exr_root: Path,
                            max_workers: int = None):
        tasks = []
        for split in ("train", "val"):
            pan_out = root/split/"panoramas"
            pan_out.mkdir(parents=True, exist_ok=True)
            for cap in (root/split/"captions").glob("*.txt"):
                pid = cap.stem
                exr_file = exr_root/f"{pid}.exr"
                if not exr_file.exists():
                    continue
                png_file = pan_out/f"{pid}.png"
                tasks.append((str(exr_file), str(png_file)))

        print(f"🌅 parallel_dump_panos - {len(tasks)} panoramas need to be processed from {exr_root} to {root}/train or val/panoramas")
        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            for result in exe.map(convert_exr_to_png, tasks):
                print(f"parallel_dump_panos - convert_exr_to_png get results: {result}")

    # dump out all panoramas in parallel (speeds up by #CPUs)
    parallel_dump_panos(
        root=root,
        exr_root=exr_root,
        max_workers=16,   # or whatever you’ve got cores for
    )

    print(f"🌅 Panoramas exported to {root}/train or val/panoramas done, cost {time.time()-start_time_pano:.4f} secs")

    # 2) Caption embeddings via CLIP (batched on GPU)
    print(f"📝 Generating caption embeddings with CLIP (batched on GPU) for {root}/train and val/captions...")
    start_time_emb = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # use "runwayml/stable-diffusion-v1-5" (not "openai/clip-vit-base-patch32") because the later is not available in the local cache on aml-personal
    model_name  = "runwayml/stable-diffusion-v1-5" # "openai/clip-vit-base-patch32"
    start_time_toekn = time.time()
    tokenizer = safe_from_pretrained(
        CLIPTokenizer.from_pretrained,
        model_name,
        subfolder="tokenizer"
    )
    print(f"loaded tokenizer openai/clip-vit-base-patch32 done, cost {time.time()-start_time_toekn:.4f} secs")
    
    start_time_encode = time.time()
    text_enc = safe_from_pretrained(
        CLIPTextModel.from_pretrained,
        model_name,
        subfolder="text_encoder",
        torch_dtype=torch.float16
    ).to(device).eval()
    print(f"📝 CLIP text encoder openai/clip-vit-base-patch32 loaded on {device} done, cost {time.time()-start_time_encode:.4f} secs")

    for split in ("train","val"):
        emb_dir = root/split/"embeddings"
        emb_dir.mkdir(parents=True, exist_ok=True)
        pids = [p.stem for p in (root/split/"captions").glob("*.txt")]
        for i in range(0, len(pids), gpu_batch_size):
            batch = pids[i:i+gpu_batch_size]
            texts = [ (root/split/"captions"/f"{pid}.txt").read_text().strip() for pid in batch ]
            toks = tokenizer(texts, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt").to(device)
            with torch.no_grad():
                emb = text_enc(**toks).last_hidden_state.cpu()
            for j,pid in enumerate(batch):
                torch.save(emb[j], emb_dir/f"{pid}.pt")
    print(f"📝 Caption embeddings generated and saved to {root}/train or val/embeddings done, cost {time.time()-start_time_emb:.4f} secs")

    # 3) Unified data-quality validation
    start_time_report = time.time()
    unified = {}
    for split in ("train","val"):
        split_root = root/split
        test_generated_faces(split_root, n=3)
        stats = validate_dataset_integrity(split_root)
        update_global_stats(unified, split, stats)
    report_path = root/"validation_report.json"
    save_statistics(unified, report_path)
    print(f"🩺 Unified validation report written: {report_path}, cost {time.time()-start_time_report:0.4f} secs")

    # 4) TAR packaging & loader smoke-test
    # print(f"📦 Creating TAR archives for CubeDiff training, calling create_tar.sh with data_root as {str(root)}, workers as {str(cpu_workers)} ...")
    # start_time_tar = time.time()
    # script = Path(__file__).parent/"create_tar.sh"
    # subprocess.run([str(script), "--data_root", str(root), "--workers", str(cpu_workers)], check=True)
    # for split in ("train","val"):
    #     tar = str(root/f"cubediff_{split}.tar")
    #     print(f"📦 for {split} created {tar} and call latent_webdataset.get_dataloader")
    #     loader = latent_webdataset.get_dataloader(tar, batch_size=4, shuffle=False)
    #     batch = next(iter(loader))
    #     print(f"🔗 {split} data loader from latent_webdataset - OK -- latent shape: {batch['latent'].shape} -- encoder_hidden_state shape: {batch['encoder_hidden_state'].shape} -- attention_mask shape : {batch['attention_mask'].shape}")
    
    # print(f"📦 TAR packaging and smoke-test done, cost {time.time()-start_time_tar:.4f} secs")
    
    print("🎉 All post-build steps completed in-process!")

    # report of the final dataset
    print(f"📑 Generating final report at {output_dir}/final_report.json ...")
    report_path = Path(output_dir) / "final_report.json" 
    generate_final_report(
        output_dir   = Path(output_dir),
        exr_dir      = Path(exr_dir),
        exr_paths    = exr_paths,
        train_items  = train_items,
        val_items    = val_items,
        out_json     = report_path,
        model_name   = model_name
    )
    print(f"✅ Final report generated at {report_path}")

    generate_visual_samples(Path(output_dir))
    print(f"✅ generating visual smaples of cubemap faces done, check {output_dir}/visual")

if __name__ == "__main__":
    import cl.data.latent_webdataset  as latent_webdataset
    import argparse
    
    parser = argparse.ArgumentParser(description="Parallel CPU-GPU CubeDiff dataset generation")
    parser.add_argument("--exr_dir", required=True, help="Directory with .exr files")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--split_ratio", type=float, default=0.9, help="Train/val split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cpu_workers", type=int, default=12, help="CPU worker processes")
    parser.add_argument("--gpu_batch_size", type=int, default=32, help="GPU batch size")
    args = parser.parse_args()
    
    # Set multiprocessing method
    # try:
    #     mp.set_start_method("spawn", force=True)
    # except RuntimeError:
    #     pass
    
    build_cubediff_parallel(
        exr_dir=args.exr_dir,
        output_dir=args.output_dir,
        split_ratio=args.split_ratio,
        seed=args.seed,
        cpu_workers=args.cpu_workers,
        gpu_batch_size=args.gpu_batch_size
    )
