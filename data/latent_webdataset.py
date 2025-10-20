# latent_webdataset.py
# -------- UPDATED data loader to load the latent tensors and captions from a webdataset ---------
import math
import os
import io
from typing import List, Dict, Any
import torch
# import webdataset as wds
from torch.utils.data import DataLoader, Dataset
from diffusers import StableDiffusionPipeline, AutoencoderKL
# from transformers import CLIPTokenizer, CLIPTextModel
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import matplotlib.pyplot as plt
# from pathlib import Path
# from PIL import Image
from torch import Generator
import tarfile
# 
# ─── Load the SAME tokenizer + text_encoder as in Trainer ───
PRETRAINED_MODEL = "runwayml/stable-diffusion-v1-5"

try:
    pipe = StableDiffusionPipeline.from_pretrained(
        PRETRAINED_MODEL,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    )
    print(f"✅ Loaded {PRETRAINED_MODEL} from cache")
except (OSError, ValueError):
    pipe = StableDiffusionPipeline.from_pretrained(
        PRETRAINED_MODEL,
        torch_dtype=torch.bfloat16,
    )
    print(f"🔄 Downloaded and cached {PRETRAINED_MODEL}")

tokenizer    = pipe.tokenizer                      # CLIPTokenizer
text_encoder = pipe.text_encoder.eval().to("cpu")   # CLIPTextModel on CPU
MAX_LEN      = tokenizer.model_max_length

# Global logging counter for preprocess function
_preprocess_log_count = 0


def preprocess(sample: dict, rank: int) -> dict:
    """
    Turn one sample (containing exactly one .pt and one .txt) into:
        { "latent": Tensor[6,4,H,W],
        "encoder_hidden_states": Tensor[seq_len,hidden_dim],
        "attention_mask": Tensor[seq_len] }
    
    FIXED: Now accepts variable latent resolutions while preferring 64×64
    """
    global _preprocess_log_count
    
    # Debug: show exactly what keys arrived
    # print(f"[rank {rank}] preprocess(sample) keys = {list(sample.keys())}")

    # 1) Find & load the latent (.pt → Tensor)
    try:
        pt_key = next(k for k in sample if k.endswith(".pt"))
    except StopIteration:
        print(f"❌ rank:{rank} - No '.pt' key in sample; keys={list(sample.keys())}")
        raise

    try:
        latent = torch.load(io.BytesIO(sample[pt_key]))
    except Exception as e:
        print(f"❌ rank : {rank}, Failed to load latent from {pt_key}; keys={list(sample.keys())}; error: {e}")    
    
    # Try to load memory-mapped embeddings first, fallback to .pt format
    try:
        # Check for .npz format first (memory-mapped)
        npz_key = next((k for k in sample if k.endswith(".npz")), None)
        if npz_key:
            # Load from memory-mapped npz
            npz_bytes = io.BytesIO(sample[npz_key])
            with np.load(npz_bytes) as data:
                encoder_hidden_states = torch.from_numpy(data['encoder_hidden_states'])
                attention_mask = torch.from_numpy(data['attention_mask'])
        else:
            # Fallback to .emb/.pt format
            emb_key = next(k for k in sample if k.endswith(".emb"))
            embed_data = torch.load(io.BytesIO(sample[emb_key]))
            encoder_hidden_states = embed_data['encoder_hidden_states']
            attention_mask = embed_data['attention_mask']
    except Exception as e:
        print(f"❌ [rank {rank}] Failed to load embeddings and attention mask; keys={list(sample.keys())}; error: {e}")
        raise ValueError(f"rank {rank} - latent_webdataset.py - preprocess - Failed to load embeddings and attn mask due to error : {e}")
    
    # 4) Sanity checks: everything must be a Tensor now
    if not isinstance(latent, torch.Tensor):
        raise ValueError(f"[rank {rank}] Bad latent (not a Tensor): key={pt_key}, keys={list(sample.keys())}")
    if not isinstance(encoder_hidden_states, torch.Tensor):
        raise ValueError(f"[rank {rank}] Bad encoder_hidden_states (not a Tensor): key={txt_key}, keys={list(sample.keys())}")
    if not isinstance(attention_mask, torch.Tensor):
        raise ValueError(f"[rank {rank}] Bad attention_mask (not a Tensor): key={txt_key}, keys={list(sample.keys())}")
    
    return {
        "latent":                latent,
        "encoder_hidden_states": encoder_hidden_states,
        "attention_mask":        attention_mask,
    }

def collate_fn(batch: list[dict]) -> dict:
    """
    Stack a list of {"latent":…, "encoder_hidden_states":…, "attention_mask":…}
    into one batch of Tensors.
    
    FIXED: Now handles variable latent resolutions gracefully
    """
    # Check if all latents have the same shape
    latent_shapes = [b["latent"].shape for b in batch]
    if len(set(latent_shapes)) > 1:
        print(f"⚠️ Warning: Mixed latent shapes in batch: {set(latent_shapes)}")
        # For now, we'll still try to stack - PyTorch will error if incompatible
    
    latents = torch.stack([b["latent"]                for b in batch], dim=0)  # [B,6,4,H,W]
    enc_emb = torch.stack([b["encoder_hidden_states"] for b in batch], dim=0)  # [B,seq_len,hidden_dim]
    attn_m  = torch.stack([b["attention_mask"]        for b in batch], dim=0)  # [B,seq_len]
    return {
        "latent":                 latents,
        "encoder_hidden_states":  enc_emb,
        "attention_mask":         attn_m,
    }


class ListDataset(Dataset):
    """ Wrap a Python list of sample‐dicts into a torch.utils.data.Dataset """
    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]



def get_dataloader(
    wds_path: list[str],
    batch_size: int,
    data_size: int,
    num_workers: int = 8,
    is_eval: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """
    Build a DataLoader by:
      1) Building the raw WebDataset pipeline (shuffle if train, then map(preprocess))
      2) Materializing it into a Python list to count exactly how many valid samples remain
      3) Splitting that list contiguously across `world_size` ranks
         (so rank i takes all_valid[start_i : end_i])
      4) Wrapping the sliced list into a small Dataset and DataLoader

    This guarantees each rank gets exactly its contiguous block of the mapped samples,
    even if preprocess(...) filtered some items out.
    
    FIXED: Now handles variable latent resolutions (32×32, 64×64, etc.)

    # 2025-7-4 update to load tar files to RAM at once
    Build a DataLoader by:
      1) Loading all raw .pt/.emb/.npz files from the tar into RAM
      2) Grouping files by sample-prefix into dicts
      3) Applying preprocess(...) to each dict, collecting valid samples
      4) Splitting that list contiguously across `world_size` ranks
      5) Wrapping the sliced list into a simple Dataset + DataLoader
    """

    # 2) Build the "full" pipeline: WebDataset(urls) → shuffle (if train) → map(preprocess)
    # if not is_eval:
    #     # TRAINING: enable shardshuffle + sample‐level shuffle
    #     dataset_full = (
    #         wds.WebDataset(
    #             urls=wds_path,
    #             nodesplitter=lambda urls: urls,
    #             handler=wds.warn_and_continue,
    #             empty_check=False,
    #             shardshuffle=1000,    # shuffle which shard first
    #         )
    #         .shuffle(1000, initial=100)  # sample‐level shuffle
    #         .map(preprocess, handler="ignore")
    #     )
    # else:
    #     # VALIDATION: no shuffle, just map
    #     dataset_full = (
    #         wds.WebDataset(
    #             urls=wds_path,
    #             nodesplitter=lambda urls: urls,
    #             handler=wds.warn_and_continue,
    #             empty_check=False,
    #             shardshuffle=False,
    #         )
    #         .map(preprocess, handler="ignore")
    #     )

    # 1) Load & group raw bytes from the tar
    samples_raw: dict[str, dict[str, bytes]] = {}
    with tarfile.open(wds_path, "r") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            fname = os.path.basename(member.name)      # e.g. "abc123.pt" or "abc123.emb"
            sid   = fname.rsplit(".", 1)[0]            # e.g. "abc123"
            blob  = tar.extractfile(member).read()     # raw bytes
            samples_raw.setdefault(sid, {})[fname] = blob

    # 3) Materialize pipeline into a Python list of valid items (log along the way)
    # all_valid = []
    # print(f"[latent_webdataset] rank {rank} – is_eval={is_eval} – materializing WebDataset pipeline...")    
    # for sample in dataset_full:
    #     all_valid.append(sample)
    #     # If the dataset is huge, you could break after collecting `data_size` plus some margin,
    #     # but here we want exactly all valid samples.

    # 2) Preprocess each raw-sample dict
    all_valid = []
    print(f"[latent_webdataset] rank={rank} loading {len(samples_raw)} raw samples")
    for sid, raw in samples_raw.items():
        try:
            proc = preprocess(raw, rank)  # your existing function
            all_valid.append(proc)
        except Exception as e:
            print(f"latent_webdataset.py - ⚠️ [rank={rank}] preprocess failed for {sid}: {e}")
            continue

    actual_size = len(all_valid)
    print(
        f"[latent_webdataset] rank {rank} – is_eval={is_eval} – "
        f" length = {actual_size} (expected ~{data_size})"
    )
    
    if actual_size == 0:
        raise RuntimeError(f"rank={rank} – no valid samples after preprocessing")
    
    # ─── replicate WebDataset’s shardshuffle + sample-level shuffle ───
    if not is_eval:
        # one global shuffle of all_valid (like shardshuffle + .shuffle())
        rng = torch.Generator()
        rng.manual_seed(42)
        # Note: torch.randperm on CPU is fast even for thousands of elements
        perm = torch.randperm(actual_size, generator=rng).tolist()
        all_valid = [all_valid[i] for i in perm]
        print(f"latent_webdataset.py - [latent_webdataset] rank={rank} – global shuffle applied for training")
   # else: for eval we leave all_valid in original order
   # ────────────────────────────────────────────────────────────────

    # 4) Compute the per‐rank contiguous split of `all_valid` for DDP
    base   = actual_size // world_size
    extra  = actual_size % world_size
    if rank < extra:
        start_idx = rank * (base + 1)
        end_idx   = start_idx + (base + 1)
    else:
        start_idx = extra * (base + 1) + (rank - extra) * base
        end_idx   = start_idx + base

    # 5) Slice the list for this rank
    rank_samples = all_valid[start_idx:end_idx]
    rank_size = len(rank_samples)
    print(
        f"[latent_webdataset] rank {rank} – taking slice [{start_idx}:{end_idx}] "
        f"= {rank_size} samples from {actual_size} total"
    )
    
    # 6) Wrap the sliced list into a list-backed Dataset and DataLoader
    dataset_final = ListDataset(rank_samples)
    
    # 7) Build the DataLoader
    # create a fresh CPU generator
    gen = Generator(device='cpu')
    # (optional) seed it for reproducibility
    gen.manual_seed(42)

    dataloader = DataLoader(
        dataset_final,
        batch_size=batch_size,
        shuffle=False, # (not is_eval),  # shuffle for training, not for eval but shuffle has been applied already, no need to shuffle again
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor= 64, # 32, # 16, # 160, # 630/4=158 load all data to CPU RAM at once # 64, # 32, # 16, # 4, #2,
        persistent_workers=True, 
        generator=gen,      # ← ensure CPU‐only RNG
        drop_last=(not is_eval),  # drop last batch for training to maintain consistent batch sizes
        multiprocessing_context='fork' # On Linux, fork is actually a bit faster than forkserver for DataLoader worker IPC # forkserver still tends to be a hair faster on Linux # 'spawn' : Better for CUDA tensors and safe but higher‑latency on process startup
    )

    print(
        f"[latent_webdataset] rank {rank} – DataLoader ready: "
        f"{len(dataset_final)} samples, batch_size={batch_size}, "
        f"~{len(dataloader)} batches"
    )
    
    return dataloader


def test_dataloader(tar_path: str, batch_size: int = 2, num_batches: int = 3):
    """
    Test function to validate the dataloader works correctly with the fixed latent format.
    RESTORED: From original latent_webdataset.py with updates for new latent format.
    """
    print(f"🧪 Testing dataloader with {tar_path}")
    
    dataloader = get_dataloader(
        wds_path=[tar_path],
        batch_size=batch_size,
        data_size=100,  # Estimate
        num_workers=2,
        is_eval=True,
        rank=0,
        world_size=1
    )
    
    print(f"📊 DataLoader created with {len(dataloader)} batches")
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
            
        print(f"\nBatch {i+1}:")
        print(f"  Latent shape: {batch['latent'].shape}")
        print(f"  Encoder hidden states shape: {batch['encoder_hidden_states'].shape}")
        print(f"  Attention mask shape: {batch['attention_mask'].shape}")
        
        # Validate latent format - UPDATED for variable resolutions
        latent = batch['latent']
        if len(latent.shape) == 5 and latent.shape[1:3] == (6, 4):
            h, w = latent.shape[3], latent.shape[4]
            status = "✅ CORRECT" if h == 64 and w == 64 else f"⚠️ UNEXPECTED ({h}×{w})"
            print(f"  Latent resolution: {h}×{w} {status}")
            
            # Check value ranges
            lat_min, lat_max = latent.min().item(), latent.max().item()
            print(f"  Latent value range: [{lat_min:.3f}, {lat_max:.3f}]")
        else:
            print(f"  ❌ Invalid latent format: {latent.shape}")
    
    print(f"\n✅ DataLoader test completed")


def visualize_latent_faces(latent_tensor: torch.Tensor, output_path: str = None, title: str = "Latent Faces"):
    """
    Visualize the 6 cubemap faces from a latent tensor [6,4,H,W].
    RESTORED: From original latent_webdataset.py with updates for variable resolutions.
    
    Args:
        latent_tensor: Tensor of shape [6,4,H,W] representing 6 cubemap faces
        output_path: Optional path to save the visualization
        title: Title for the plot
    """
    if len(latent_tensor.shape) != 4 or latent_tensor.shape[0] != 6 or latent_tensor.shape[1] != 4:
        raise ValueError(f"Expected latent shape [6,4,H,W], got {latent_tensor.shape}")
    
    face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
    h, w = latent_tensor.shape[2], latent_tensor.shape[3]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i, (face_name, ax) in enumerate(zip(face_names, axes.flatten())):
        # Use the first channel for visualization
        face_data = latent_tensor[i, 0].cpu().numpy()
        
        # Normalize for better visualization
        face_norm = (face_data - face_data.min()) / (face_data.max() - face_data.min() + 1e-8)
        
        im = ax.imshow(face_norm, cmap='viridis')
        ax.set_title(f"{face_name.title()} (face {i})", fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    resolution_status = "✅ CORRECT" if h == 64 and w == 64 else f"⚠️ UNEXPECTED ({h}×{w})"
    plt.suptitle(f"{title}\nShape: {latent_tensor.shape} {resolution_status}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {output_path}")
    else:
        plt.show()
    
    plt.close()


def test_latent_visualization(tar_path: str, num_samples: int = 2, output_dir: str = "latent_visualizations"):
    """
    Test function to visualize latent faces from WebDataset samples.
    RESTORED: From original latent_webdataset.py with updates for new format.
    """
    print(f"🎨 Testing latent visualization with {tar_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load samples from WebDataset
    import webdataset as wds
    import io
    
    ds = wds.WebDataset(tar_path, handler=wds.warn_and_continue)
    
    sample_count = 0
    for sample in ds:
        if sample_count >= num_samples:
            break
        
        # Extract sample information
        keys = list(sample.keys())
        pid = sample.get("__key__", f"sample_{sample_count}")
        
        # Find .pt file
        pt_keys = [k for k in keys if k.endswith('.pt')]
        if len(pt_keys) != 1:
            print(f"❌ Sample {sample_count}: Expected 1 .pt file, found {len(pt_keys)}")
            continue
        
        try:
            # Load latent tensor
            latent_bytes = sample[pt_keys[0]]
            latent = torch.load(io.BytesIO(latent_bytes))
            
            print(f"Sample {sample_count}: {pid}")
            print(f"  Latent shape: {latent.shape}")
            
            # Create visualization
            output_path = os.path.join(output_dir, f"{pid}_latent_faces.png")
            visualize_latent_faces(latent, output_path, f"Sample {sample_count}: {pid}")
            
            sample_count += 1
            
        except Exception as e:
            print(f"❌ Failed to visualize sample {sample_count}: {e}")
            continue
    
    print(f"\n✅ Created {sample_count} latent visualizations in {output_dir}")


def debug_sample_structure(tar_path: str, num_samples: int = 5):
    """
    Debug function to examine the structure of samples in WebDataset.
    RESTORED: From original latent_webdataset.py with enhanced debugging.
    """
    print(f"🔍 Debugging sample structure in {tar_path}")
    
    import webdataset as wds
    import io
    
    ds = wds.WebDataset(tar_path, handler=wds.warn_and_continue)
    
    sample_count = 0
    latent_shapes = {}
    
    for sample in ds:
        if sample_count >= num_samples:
            break
        
        keys = list(sample.keys())
        pid = sample.get("__key__", f"sample_{sample_count}")
        
        print(f"\nSample {sample_count}: {pid}")
        print(f"  Keys: {keys}")
        
        # Analyze .pt files
        pt_keys = [k for k in keys if k.endswith('.pt')]
        txt_keys = [k for k in keys if k.endswith('.txt')]
        
        print(f"  .pt files: {len(pt_keys)}")
        print(f"  .txt files: {len(txt_keys)}")
        
        if pt_keys:
            try:
                latent_bytes = sample[pt_keys[0]]
                latent = torch.load(io.BytesIO(latent_bytes))
                shape_key = str(tuple(latent.shape))
                latent_shapes[shape_key] = latent_shapes.get(shape_key, 0) + 1
                
                print(f"  Latent shape: {latent.shape}")
                print(f"  Latent dtype: {latent.dtype}")
                print(f"  Latent range: [{latent.min().item():.3f}, {latent.max().item():.3f}]")
                
            except Exception as e:
                print(f"  ❌ Failed to load latent: {e}")
        
        if txt_keys:
            try:
                caption_bytes = sample[txt_keys[0]]
                caption = caption_bytes.decode('utf-8') if isinstance(caption_bytes, bytes) else caption_bytes
                print(f"  Caption: '{caption[:50]}{'...' if len(caption) > 50 else ''}'")
            except Exception as e:
                print(f"  ❌ Failed to load caption: {e}")
        
        sample_count += 1
    
    print(f"\n📊 Summary:")
    print(f"  Samples examined: {sample_count}")
    print(f"  Latent shapes found: {latent_shapes}")


def benchmark_dataloader(tar_path: str, batch_size: int = 4, num_batches: int = 10):
    """
    Benchmark the dataloader performance.
    RESTORED: From original latent_webdataset.py with timing improvements.
    """
    print(f"⚡ Benchmarking dataloader with {tar_path}")
    print(f"   Batch size: {batch_size}, Batches to test: {num_batches}")
    
    import time
    
    dataloader = get_dataloader(
        wds_path=[tar_path],
        batch_size=batch_size,
        data_size=1000,  # Estimate
        num_workers=4,
        is_eval=True,
        rank=0,
        world_size=1
    )
    
    print(f"📊 DataLoader created with {len(dataloader)} total batches")
    
    # Warm up
    print("🔥 Warming up...")
    try:
        batch = next(iter(dataloader))
        print(f"   Warm-up batch shape: {batch['latent'].shape}")
    except Exception as e:
        print(f"❌ Warm-up failed: {e}")
        return
    
    # Benchmark
    print("⏱️  Benchmarking...")
    start_time = time.time()
    batch_times = []
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        batch_start = time.time()
        
        # Simulate some processing
        latent = batch['latent']
        encoder_hidden_states = batch['encoder_hidden_states']
        attention_mask = batch['attention_mask']
        
        # Move to GPU if available (simulate training)
        if torch.cuda.is_available():
            latent = latent.cuda()
            encoder_hidden_states = encoder_hidden_states.cuda()
            attention_mask = attention_mask.cuda()
        
        batch_end = time.time()
        batch_time = batch_end - batch_start
        batch_times.append(batch_time)
        
        if i < 3:  # Show details for first 3 batches
            print(f"   Batch {i}: {batch_time:.3f}s, latent shape: {latent.shape}")
    
    total_time = time.time() - start_time
    avg_batch_time = np.mean(batch_times)
    
    print(f"\n📈 Benchmark Results:")
    print(f"   Total time: {total_time:.3f}s")
    print(f"   Average batch time: {avg_batch_time:.3f}s")
    print(f"   Batches per second: {1/avg_batch_time:.2f}")
    print(f"   Samples per second: {batch_size/avg_batch_time:.2f}")


# Enhanced main section with all testing functions
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test latent WebDataset loading with various functions")
    parser.add_argument("--tar", required=True, help="Path to WebDataset tar file")
    parser.add_argument("--test", choices=['dataloader', 'visualization', 'debug', 'benchmark', 'all'], 
                       default='all', help="Which test to run")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for testing")
    parser.add_argument("--num_batches", type=int, default=3, help="Number of batches to test")
    parser.add_argument("--num_samples", type=int, default=2, help="Number of samples for visualization/debug")
    parser.add_argument("--output_dir", default="test_outputs", help="Output directory for visualizations")
    args = parser.parse_args()
    
    print("="*80)
    print("🧪 LATENT WEBDATASET TESTING SUITE")
    print("="*80)
    
    if args.test in ['dataloader', 'all']:
        print("\n1️⃣ DATALOADER TEST")
        print("-" * 40)
        test_dataloader(args.tar, args.batch_size, args.num_batches)
    
    if args.test in ['debug', 'all']:
        print("\n2️⃣ SAMPLE STRUCTURE DEBUG")
        print("-" * 40)
        debug_sample_structure(args.tar, args.num_samples)
    
    if args.test in ['visualization', 'all']:
        print("\n3️⃣ LATENT VISUALIZATION TEST")
        print("-" * 40)
        viz_dir = os.path.join(args.output_dir, "latent_visualizations")
        test_latent_visualization(args.tar, args.num_samples, viz_dir)
    
    if args.test in ['benchmark', 'all']:
        print("\n4️⃣ DATALOADER BENCHMARK")
        print("-" * 40)
        benchmark_dataloader(args.tar, args.batch_size, 10)
    
    print("\n✅ All tests completed!")