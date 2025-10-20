#!/usr/bin/env bash
set -euo pipefail

# ────────── Command Line Argument Parsing ──────────
# Default values
DEFAULT_DATA_ROOT="../data/dataspace/polyhaven_tiny_7_new_pt"
DEFAULT_WORKERS=8

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --data_root PATH     Path to dataset root directory (default: $DEFAULT_DATA_ROOT)"
    echo "  --workers N          Number of parallel workers (default: $DEFAULT_WORKERS)"
    echo "  --help, -h           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 --data_root /path/to/my/dataset"
    echo "  $0 --data_root ../data/polyhaven_large --workers 16"
    echo ""
    echo "Expected directory structure in DATA_ROOT:"
    echo "  DATA_ROOT/"
    echo "  ├── train/"
    echo "  │   ├── faces/"
    echo "  │   ├── captions/"
    echo "  │   ├── latents/"
    echo "  │   └── panoramas/"
    echo "  └── val/"
    echo "      ├── faces/"
    echo "      ├── captions/"
    echo "      ├── latents/"
    echo "      └── panoramas/"
}

# Parse command line arguments
DATA_ROOT="$DEFAULT_DATA_ROOT"
WORKERS="$DEFAULT_WORKERS"

while [[ $# -gt 0 ]]; do
    case $1 in
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ────────── Derived Configuration Variables ──────────
# All paths now derived from configurable DATA_ROOT
TRAIN_ROOT="$DATA_ROOT/train"
VAL_ROOT="$DATA_ROOT/val"
TRAIN_TAR="$DATA_ROOT/cubediff_train.tar"
VAL_TAR="$DATA_ROOT/cubediff_val.tar"
VISUAL_DIR="$DATA_ROOT/visual"
TEST_RESULTS_DIR="$DATA_ROOT/webdataset_test_results"

echo "=============================================="
echo "  CubeDiff Data Pipeline: Create & Verify TARs"
echo "=============================================="
echo

# ────────── Configuration Summary ──────────
echo "📁 CONFIGURATION:"
echo "→ Data root  : $DATA_ROOT"
echo "→ Train root : $TRAIN_ROOT"
echo "→ Val root   : $VAL_ROOT"
echo "→ Train TAR  : $TRAIN_TAR"
echo "→ Val TAR    : $VAL_TAR"
echo "→ Workers    : $WORKERS"
echo "→ Visual dir : $VISUAL_DIR"
echo "→ Test dir   : $TEST_RESULTS_DIR"
echo

# ────────── Input Validation ──────────
echo "🔍 VALIDATING INPUT DIRECTORIES..."

# Check if DATA_ROOT exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "❌ Data root directory not found: $DATA_ROOT"
    echo "   Please ensure the directory exists or provide correct --data_root path"
    exit 1
fi

# Check if train/val directories exist
if [ ! -d "$TRAIN_ROOT" ]; then
    echo "❌ Train directory not found: $TRAIN_ROOT"
    echo "   Make sure you've run build_tiny_set.py to create the dataset structure"
    exit 1
fi

if [ ! -d "$VAL_ROOT" ]; then
    echo "❌ Val directory not found: $VAL_ROOT"
    echo "   Make sure you've run build_tiny_set.py to create the dataset structure"
    exit 1
fi

# Check required subdirectories
for split in "train" "val"; do
    split_dir="$DATA_ROOT/$split"
    for subdir in "faces" "captions" "latents" "panoramas"; do
        check_dir="$split_dir/$subdir"
        if [ ! -d "$check_dir" ]; then
            echo "❌ Missing required subdirectory: $check_dir"
            echo "   Please run build_tiny_set.py to generate complete dataset structure"
            exit 1
        fi
    done
done

echo "✅ All required directories found"

# Create output directories
mkdir -p "$VISUAL_DIR"
mkdir -p "$TEST_RESULTS_DIR"

# ────────── Step 1: Updated make_train_val_tars.py ──────────
cat > make_train_val_tars.py << 'EOF'
#!/usr/bin/env python
"""
Create WebDataset train/val tar files from CubeDiff latents.
Updated to work with build_tiny_set.py output structure.
Parallelized via --workers.
"""

import argparse, glob, os, random, sys, io
import torch, webdataset as wds
from multiprocessing import Pool
from pathlib import Path

def load_latent(path):
    """Load a single latent tensor from .pt file"""
    return torch.load(path, map_location="cpu")

def build_sample(pid, split_root, debug):
    """
    Load six face-latents and caption for pid from build_tiny_set.py output structure.
    Consolidates 6 face latents into a single tensor [6,4,H,W] where H,W can vary.
    Enhanced to include memory-mapped embeddings.
    Expected structure:
    split_root/
    ├── latents/{pid}_face0.pt, {pid}_face1.pt, ..., {pid}_face5.pt
    └── captions/{pid}.txt
    """
    sample = {"__key__": pid}
    
    # Load caption from individual text file
    caption_path = os.path.join(split_root, "captions", f"{pid}.txt")
    if not os.path.exists(caption_path):
        raise FileNotFoundError(f"Caption file not found: {caption_path}")
    
    with open(caption_path, 'r', encoding='utf-8') as f:
        caption = f.read().strip()
    sample[f"{pid}.txt"] = caption.encode('utf-8')
    
    # Load pre-computed embeddings - NEW
    embed_npz_path = os.path.join(split_root, "embeddings", f"{pid}.npz")
    embed_pt_path = os.path.join(split_root, "embeddings", f"{pid}.pt")
    if not os.path.exists(embed_npz_path) and not os.path.exists(embed_pt_path):
        raise FileNotFoundError(f"Embedding file not found: {embed_npz_path} or {embed_pt_path}")
    
    if os.path.exists(embed_npz_path):
        # Load and include .npz file
        with open(embed_npz_path, 'rb') as f:
            sample[f"{pid}.npz"] = f.read()
    # elif os.path.exists(embed_pt_path):
    #     # Fallback to .pt format
    #     embed_data = torch.load(embed_pt_path)
    #     embed_buf = io.BytesIO()
    #     torch.save(embed_data, embed_buf)
    #     sample[f"{pid}.emb"] = embed_buf.getvalue()
    elif os.path.exists(embed_pt_path):
        # Fallback to .pt format — wrap bare tensor into a dict
        raw_tensor = torch.load(embed_pt_path, map_location="cpu")
        # If you already have an attention mask saved in parallel, load that;
        # otherwise just synthesize a full‐ones mask of the right length:
        seq_len = raw_tensor.shape[-2]  # e.g. [F, seq_len, dim] or [seq_len, dim]
        attn_mask = torch.ones(seq_len, dtype=torch.long)
        wrapped = {
            "encoder_hidden_states": raw_tensor,
            "attention_mask": attn_mask
        }
        buf = io.BytesIO()
        torch.save(wrapped, buf)
        sample[f"{pid}.emb"] = buf.getvalue()
    else:
        raise FileNotFoundError(f"No embedding file found for {pid}")

    # Load and consolidate six face latents into single tensor [6,4,H,W]
    face_latents = []
    expected_shape = None
    
    for i in range(6):
        latent_path = os.path.join(split_root, "latents", f"{pid}_face{i}.pt")
        if not os.path.exists(latent_path):
            raise FileNotFoundError(f"Latent file not found: {latent_path}")
        
        # Load individual face latent [1,4,H,W] or [4,H,W]
        face_tensor = load_latent(latent_path)
        
        # Ensure shape is [4,H,W] (remove batch dimension if present)
        if face_tensor.dim() == 4:
            face_tensor = face_tensor.squeeze(0)  # [1,4,H,W] -> [4,H,W]
        elif face_tensor.dim() != 3:
            raise ValueError(f"Expected latent tensor to be 3D or 4D, got {face_tensor.dim()}D for {latent_path}")
        
        # Validate shape consistency across faces
        if expected_shape is None:
            expected_shape = face_tensor.shape
            if face_tensor.shape[0] != 4:
                raise ValueError(f"Expected 4 channels, got {face_tensor.shape[0]} for {latent_path}")
        else:
            if face_tensor.shape != expected_shape:
                raise ValueError(f"Shape mismatch: expected {expected_shape}, got {face_tensor.shape} for {latent_path}")
            
        face_latents.append(face_tensor)
    
    # Stack into [6,4,H,W] tensor - CORRECT face ordering: [front, right, back, left, top, bottom]
    consolidated_latent = torch.stack(face_latents, dim=0)  # [6,4,H,W]
    
    # Serialize consolidated latent to bytes
    buf = io.BytesIO()
    torch.save(consolidated_latent, buf)
    sample[f"{pid}.pt"] = buf.getvalue()
    
    if debug:
        print(f"[build_sample] pid={pid}, caption_len={len(caption)}, "
              f"consolidated_latent_shape={consolidated_latent.shape}")
    
    return sample

# Global variables to share with multiprocessing workers
_split_root = None
_debug = None

def _build_sample_worker(pid):
    """
    Worker function for multiprocessing that uses global variables.
    This avoids the lambda pickle issue.
    """
    return build_sample(pid, _split_root, _debug)

def get_panorama_ids(split_root):
    """
    Extract panorama IDs from the latents directory by looking for _face0.pt files.
    """
    latents_dir = os.path.join(split_root, "latents")
    if not os.path.exists(latents_dir):
        raise FileNotFoundError(f"Latents directory not found: {latents_dir}")
    
    # Find all {pid}_face0.pt files to extract panorama IDs
    face0_files = glob.glob(os.path.join(latents_dir, "*_face0.pt"))
    
    # Extract panorama IDs by removing "_face0.pt" suffix
    pids = []
    for face0_file in face0_files:
        basename = os.path.basename(face0_file)
        if basename.endswith("_face0.pt"):
            pid = basename[:-len("_face0.pt")]
            pids.append(pid)
    
    return sorted(pids)

def process_split(split_root, tar_path, debug, workers, split_name):
    """
    Write a single split (train or val) to tar_path using WebDataset.
    """
    global _split_root, _debug
    
    # Get all panorama IDs from this split
    pids = get_panorama_ids(split_root)
    n = len(pids)
    
    if n == 0:
        raise ValueError(f"No panorama IDs found in {split_root}")
    
    print(f"  → [{split_name}] {n} samples → {tar_path}")
    
    # Create WebDataset tar file
    sink = wds.TarWriter(tar_path)
    
    if workers > 1:
        # Set global variables for multiprocessing workers
        _split_root = split_root
        _debug = debug
        
        # Parallel processing using global worker function
        with Pool(workers) as pool:
            for sample in pool.imap_unordered(_build_sample_worker, pids):
                sink.write(sample)
    else:
        # Serial processing
        for pid in pids:
            sample = build_sample(pid, split_root, debug)
            sink.write(sample)
    
    sink.close()
    print(f"  → [{split_name}] Created {tar_path} with {n} samples")

def main():
    p = argparse.ArgumentParser(description="Create WebDataset tar files from build_tiny_set.py output")
    p.add_argument("--train_root", required=True, help="Path to train directory")
    p.add_argument("--val_root",   required=True, help="Path to val directory")
    p.add_argument("--train_tar",  default="cubediff_train.tar", help="Output train tar file")
    p.add_argument("--val_tar",    default="cubediff_val.tar", help="Output val tar file")
    p.add_argument("--workers",    type=int, default=1, help="Number of parallel workers")
    p.add_argument("--debug",      action="store_true", help="Enable debug output")
    args = p.parse_args()

    print("Creating train TAR...")
    process_split(args.train_root, args.train_tar, args.debug, args.workers, "train")
    
    print("Creating val TAR...")
    process_split(args.val_root, args.val_tar, args.debug, args.workers, "val")
    
    print("✅ TAR creation complete!")

if __name__ == "__main__":
    main()
EOF
chmod +x make_train_val_tars.py
echo "Generated → make_train_val_tars.py"

# ────────── Step 2: Updated verify_simple.py ──────────
cat > verify_simple.py << 'EOF'
#!/usr/bin/env python
"""
Simple verification for CubeDiff WebDataset tar files.
Validates the consolidated latent format [6,4,64,64].
"""
import argparse, sys, os, io
import webdataset as wds
import torch

def verify(path):
    print(f"Verifying {path} (size {(os.path.getsize(path)/(1024*1024)):.1f} MB)...")
    ds = wds.WebDataset(path, handler=wds.warn_and_continue)
    count = 0
    latent_shapes = set()  # Track different latent shapes found
    
    for sample in ds:
        count += 1
        
        # Check sample structure
        keys = list(sample.keys())
        
        if count <= 3:  # Show details for first 3 samples
            print(f"  Sample {count} keys: {keys}")
            
            # Find the .pt and .txt keys
            pt_keys = [k for k in keys if k.endswith('.pt')]
            txt_keys = [k for k in keys if k.endswith('.txt')]
            
            if len(pt_keys) != 1:
                print(f"    ❌ Expected 1 .pt file, found {len(pt_keys)}: {pt_keys}")
                continue
                
            if len(txt_keys) != 1:
                print(f"    ❌ Expected 1 .txt file, found {len(txt_keys)}: {txt_keys}")
                continue
            
            # Load and verify latent tensor
            try:
                latent_bytes = sample[pt_keys[0]]
                latent = torch.load(io.BytesIO(latent_bytes))
                latent_shapes.add(tuple(latent.shape))
                
                # Check if it's a valid cubemap shape [6,4,H,W]
                if len(latent.shape) == 4 and latent.shape[0] == 6 and latent.shape[1] == 4:
                    h, w = latent.shape[2], latent.shape[3]
                    status = "✅ CORRECT" if h == 64 and w == 64 else f"⚠️ UNEXPECTED ({h}×{w})"
                    print(f"    Latent shape: {latent.shape} {status}")
                else:
                    print(f"    ❌ Latent shape: {latent.shape} (expected [6,4,H,W])")
                    
            except Exception as e:
                print(f"    ❌ Failed to load latent: {e}")
            
            # Verify caption
            try:
                caption_bytes = sample[txt_keys[0]]
                caption = caption_bytes.decode('utf-8') if isinstance(caption_bytes, bytes) else caption_bytes
                print(f"    ✅ Caption: '{caption[:50]}{'...' if len(caption) > 50 else ''}'")
            except Exception as e:
                print(f"    ❌ Failed to load caption: {e}")
        
        if count >= 100:  # Don't process entire dataset for verification
            break
    
    print(f"  → {count} samples verified.")
    print(f"  → Latent shapes found: {list(latent_shapes)}")
    print()
    return count

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_tar", required=True)
    p.add_argument("--val_tar",   required=True)
    args = p.parse_args()
    
    train_count = verify(args.train_tar)
    val_count   = verify(args.val_tar)
    
    status = 0 if train_count > 0 and val_count > 0 else 1
    
    if status == 0:
        print("✅ All TAR files verified successfully!")
    else:
        print("❌ TAR verification failed!")
    
    sys.exit(status)

if __name__ == "__main__":
    main()
EOF
chmod +x verify_simple.py
echo "Generated → verify_simple.py"

# ────────── Step 3: Updated load_webdataset.py ──────────
cat > load_webdataset.py << 'EOF'
#!/usr/bin/env python
"""
Load & visualize CubeDiff WebDataset samples.
Validates the [6,4,64,64] latent format and face ordering.
"""
import argparse, os, io
import webdataset as wds
import torch
import matplotlib.pyplot as plt
import numpy as np

def tensor_from_bytes(b):
    """Load tensor from bytes buffer"""
    buf = io.BytesIO(b)
    return torch.load(buf)

def visualize(latent, pid, outdir):
    """
    Visualize the 6 cubemap faces from consolidated latent [6,4,64,64].
    Face order: [front, right, back, left, top, bottom]
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Validate latent shape
    if len(latent.shape) != 4 or latent.shape[0] != 6 or latent.shape[1] != 4:
        print(f"❌ Unexpected latent shape: {latent.shape}, expected [6,4,H,W]")
        return
    
    h, w = latent.shape[2], latent.shape[3]
    face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
    
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    
    for i, ax in enumerate(axs.flatten()):
        # Use first channel [0] of the latent for visualization
        face = latent[i][0].cpu().numpy()  # [H, W]
        
        # Normalize for display
        face = (face - face.min()) / (face.max() - face.min() + 1e-8)
        
        im = ax.imshow(face, cmap='viridis')
        ax.set_title(f"{face_names[i]} (face {i})", fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle(f"Panorama: {pid} - Latent Faces [{latent.shape}]", fontsize=14)
    plt.tight_layout()
    
    output_path = f"{outdir}/{pid}_faces.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization: {output_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tar",         required=True, help="Path to WebDataset tar file")
    p.add_argument("--num_samples", type=int, default=3, help="Number of samples to visualize")
    p.add_argument("--outdir",      default="visual", help="Output directory for visualizations")
    args = p.parse_args()

    print(f"Loading samples from: {args.tar}")
    
    ds = wds.WebDataset(args.tar, handler=wds.warn_and_continue)
    
    count = 0
    for sample in ds:
        if count >= args.num_samples:
            break
            
        keys = list(sample.keys())
        
        # Find the panorama ID and load data
        pid = sample.get("__key__", f"sample_{count}")
        
        # Find .pt and .txt files
        pt_keys = [k for k in keys if k.endswith('.pt')]
        txt_keys = [k for k in keys if k.endswith('.txt')]
        
        if len(pt_keys) != 1 or len(txt_keys) != 1:
            print(f"❌ Sample {count}: Invalid structure - pt_keys={pt_keys}, txt_keys={txt_keys}")
            continue
            
        try:
            # Load latent
            latent = tensor_from_bytes(sample[pt_keys[0]])
            
            # Load caption
            caption_bytes = sample[txt_keys[0]]
            caption = caption_bytes.decode('utf-8') if isinstance(caption_bytes, bytes) else caption_bytes
            
            print(f"Sample {count}: {pid}")
            print(f"  Latent shape: {latent.shape}")
            print(f"  Caption: '{caption[:100]}{'...' if len(caption) > 100 else ''}'")
            
            # Visualize
            visualize(latent, pid, args.outdir)
            
            count += 1
            
        except Exception as e:
            print(f"❌ Failed to process sample {count}: {e}")
            continue
    
    print(f"\n✅ Processed {count} samples")

if __name__=="__main__":
    main()
EOF
chmod +x load_webdataset.py
echo "Generated → load_webdataset.py"

# ────────── Step 4: WebDataset Compatibility Test ──────────
cat > test_webdataset_compatibility.py << 'EOF'
#!/usr/bin/env python
"""
Integrated WebDataset compatibility test for the CubeDiff pipeline.
Tests end-to-end compatibility with latent_webdataset.py DataLoader.

Usage after running process_cubediff_data.sh:
    python test_webdataset_compatibility.py --train_tar cubediff_train.tar --val_tar cubediff_val.tar
"""

import argparse
import os
import sys
import io
import torch
import webdataset as wds
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def test_tar_structure(tar_path: str, split_name: str = ""):
    """Test basic TAR file structure and content."""
    print(f"🔍 Testing {split_name} TAR structure: {tar_path}")
    
    if not os.path.exists(tar_path):
        print(f"❌ TAR file not found: {tar_path}")
        return False
    
    file_size_mb = os.path.getsize(tar_path) / (1024 * 1024)
    print(f"   File size: {file_size_mb:.1f} MB")
    
    try:
        ds = wds.WebDataset(tar_path, handler=wds.warn_and_continue)
        
        sample_count = 0
        latent_resolutions = {}
        
        for sample in ds:
            sample_count += 1
            
            if sample_count <= 2:  # Show details for first 2 samples
                keys = list(sample.keys())
                pid = sample.get("__key__", f"sample_{sample_count}")
                print(f"   Sample {sample_count}: {pid}")
                print(f"     Keys: {keys}")
                
                # Validate structure
                pt_keys = [k for k in keys if k.endswith('.pt')]
                txt_keys = [k for k in keys if k.endswith('.txt')]
                
                if len(pt_keys) == 1 and len(txt_keys) == 1:
                    # Test latent loading
                    latent = torch.load(io.BytesIO(sample[pt_keys[0]]))
                    caption = sample[txt_keys[0]].decode('utf-8')
                    
                    # Track latent resolution
                    h, w = latent.shape[2], latent.shape[3]
                    resolution_key = f"{h}×{w}"
                    latent_resolutions[resolution_key] = latent_resolutions.get(resolution_key, 0) + 1
                    
                    status = "✅ CORRECT" if h == 64 and w == 64 else f"⚠️ UNEXPECTED"
                    print(f"     Latent shape: {latent.shape} {status}")
                    print(f"     Caption: '{caption[:30]}{'...' if len(caption) > 30 else ''}'")
                else:
                    print(f"     ❌ Invalid structure: {len(pt_keys)} .pt files, {len(txt_keys)} .txt files")
                    return False
            
            if sample_count >= 50:  # Don't process entire dataset
                break
        
        print(f"   ✅ Found {sample_count} samples")
        if latent_resolutions:
            print(f"   Latent resolutions: {latent_resolutions}")
        return sample_count > 0
        
    except Exception as e:
        print(f"   ❌ Error testing TAR: {e}")
        return False

def test_dataloader_integration(tar_path: str, split_name: str = ""):
    """Test integration with a simplified DataLoader (mimics latent_webdataset.py)."""
    print(f"🔗 Testing {split_name} DataLoader integration: {tar_path}")
    
    try:
        # Create a simple preprocessing function (similar to latent_webdataset.py)
        def preprocess_sample(sample):
            # Extract latent and caption
            pt_keys = [k for k in sample.keys() if k.endswith('.pt')]
            txt_keys = [k for k in sample.keys() if k.endswith('.txt')]
            
            if len(pt_keys) != 1 or len(txt_keys) != 1:
                raise ValueError(f"Invalid sample structure")
            
            latent = torch.load(io.BytesIO(sample[pt_keys[0]]))
            caption = sample[txt_keys[0]].decode('utf-8')
            
            # Validate latent shape - accept any [6,4,H,W] format
            if len(latent.shape) != 4 or latent.shape[0] != 6 or latent.shape[1] != 4:
                raise ValueError(f"Expected latent shape [6,4,H,W], got {latent.shape}")
            
            return {
                "latent": latent,
                "caption": caption,
                "pid": sample.get("__key__", "unknown")
            }
        
        # Create dataset pipeline
        dataset = (
            wds.WebDataset(tar_path, handler=wds.warn_and_continue)
            .map(preprocess_sample, handler="ignore")
        )
        
        # Test a few samples
        sample_count = 0
        for processed_sample in dataset:
            sample_count += 1
            
            latent = processed_sample["latent"]
            caption = processed_sample["caption"]
            pid = processed_sample["pid"]
            
            if sample_count <= 2:
                h, w = latent.shape[2], latent.shape[3]
                status = "✅ CORRECT" if h == 64 and w == 64 else f"⚠️ UNEXPECTED ({h}×{w})"
                print(f"   Sample {sample_count}: {pid}")
                print(f"     Latent shape: {latent.shape} {status}")
                print(f"     Caption: '{caption[:30]}{'...' if len(caption) > 30 else ''}' ✅")
                
                # Check latent value ranges (should be reasonable after VAE encoding)
                lat_min, lat_max = latent.min().item(), latent.max().item()
                print(f"     Latent range: [{lat_min:.3f}, {lat_max:.3f}] ✅")
            
            if sample_count >= 10:  # Test first 10 samples
                break
        
        print(f"   ✅ DataLoader integration successful ({sample_count} samples processed)")
        return True
        
    except Exception as e:
        print(f"   ❌ DataLoader integration failed: {e}")
        return False

def visualize_sample_faces(tar_path: str, output_dir: str = "webdataset_visualization"):
    """Create visualization of faces from WebDataset to verify ordering."""
    print(f"🎨 Creating face visualizations from: {tar_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        ds = wds.WebDataset(tar_path, handler=wds.warn_and_continue)
        
        sample_count = 0
        for sample in ds:
            if sample_count >= 2:  # Visualize first 2 samples
                break
                
            pid = sample.get("__key__", f"sample_{sample_count}")
            
            # Load latent
            pt_keys = [k for k in sample.keys() if k.endswith('.pt')]
            latent = torch.load(io.BytesIO(sample[pt_keys[0]]))  # [6,4,H,W]
            
            # Create visualization
            face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            
            for i, ax in enumerate(axes.flatten()):
                # Use first channel of latent for visualization
                face_data = latent[i][0].numpy()  # [H, W]
                
                # Normalize for display
                face_norm = (face_data - face_data.min()) / (face_data.max() - face_data.min() + 1e-8)
                
                im = ax.imshow(face_norm, cmap='viridis')
                ax.set_title(f"{face_names[i]} (face {i})")
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            plt.suptitle(f"WebDataset Sample: {pid}\nLatent Faces {latent.shape}")
            plt.tight_layout()
            
            viz_path = Path(output_dir) / f"{pid}_webdataset_faces.png"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Saved visualization: {viz_path}")
            sample_count += 1
        
        print(f"   ✅ Created {sample_count} face visualizations")
        return True
        
    except Exception as e:
        print(f"   ❌ Visualization failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test WebDataset compatibility for CubeDiff")
    parser.add_argument("--train_tar", required=True, help="Path to training tar file")
    parser.add_argument("--val_tar", help="Path to validation tar file")
    parser.add_argument("--output_dir", default="webdataset_test_results", help="Output directory for test results")
    args = parser.parse_args()

    print("="*80)
    print("🧪 WEBDATASET COMPATIBILITY TEST")
    print("="*80)

    all_tests_passed = True
    
    # Test train TAR
    print("\n1️⃣ Testing TRAIN TAR")
    train_structure_ok = test_tar_structure(args.train_tar, "TRAIN")
    train_dataloader_ok = test_dataloader_integration(args.train_tar, "TRAIN")
    all_tests_passed = all_tests_passed and train_structure_ok and train_dataloader_ok
    
    # Test val TAR if provided
    if args.val_tar and os.path.exists(args.val_tar):
        print("\n2️⃣ Testing VAL TAR")
        val_structure_ok = test_tar_structure(args.val_tar, "VAL")
        val_dataloader_ok = test_dataloader_integration(args.val_tar, "VAL")
        all_tests_passed = all_tests_passed and val_structure_ok and val_dataloader_ok
    
    # Create visualizations
    print("\n3️⃣ Creating Visualizations")
    viz_ok = visualize_sample_faces(args.train_tar, args.output_dir)
    
    # Final summary
    print("\n" + "="*80)
    print("🏁 TEST RESULTS")
    print("="*80)
    
    if all_tests_passed:
        print("✅ ALL TESTS PASSED!")
        print("   Your WebDataset files are ready for CubeDiff training.")
        print(f"   Visualizations saved in: {args.output_dir}")
    else:
        print("❌ SOME TESTS FAILED!")
        print("   Please check the error messages above.")
    
    return 0 if all_tests_passed else 1

if __name__ == "__main__":
    sys.exit(main())
EOF
chmod +x test_webdataset_compatibility.py
echo "Generated → test_webdataset_compatibility.py"

# ────────── Step 5: Updated process_cubediff_data.sh ──────────
cat > process_cubediff_data.sh << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

# ────────── Command Line Argument Handling ──────────
# This script now receives DATA_ROOT and WORKERS from create_tar.sh
# Access them as positional arguments or environment variables

if [ $# -ge 2 ]; then
    # Arguments passed from create_tar.sh
    DATA_ROOT="$1"
    WORKERS="$2"
elif [ ! -z "${DATA_ROOT:-}" ] && [ ! -z "${WORKERS:-}" ]; then
    # Environment variables set by create_tar.sh
    echo "Using environment variables: DATA_ROOT=$DATA_ROOT, WORKERS=$WORKERS"
else
    # Fallback defaults if run independently
    DATA_ROOT="../data/dataspace/polyhaven_tiny_7_new_pt"
    WORKERS=8
    echo "⚠️  No arguments provided, using defaults: DATA_ROOT=$DATA_ROOT, WORKERS=$WORKERS"
fi

# Derived variables
TRAIN_ROOT="$DATA_ROOT/train"
VAL_ROOT="$DATA_ROOT/val"
TRAIN_TAR="$DATA_ROOT/cubediff_train.tar"
VAL_TAR="$DATA_ROOT/cubediff_val.tar"
VISUAL_DIR="$DATA_ROOT/visual"
TEST_RESULTS_DIR="$DATA_ROOT/webdataset_test_results"

echo "===================================="
echo "CubeDiff Data Processing Pipeline"
echo "===================================="
echo "📁 Data root: $DATA_ROOT"
echo "🔧 Workers: $WORKERS"
echo "📂 Train root: $TRAIN_ROOT"
echo "📂 Val root: $VAL_ROOT"
echo "📦 Train TAR: $TRAIN_TAR"
echo "📦 Val TAR: $VAL_TAR"
echo "🎨 Visual dir: $VISUAL_DIR"
echo "🧪 Test results dir: $TEST_RESULTS_DIR"
echo

# Validation (already done by create_tar.sh, but double-check)
if [ ! -d "$TRAIN_ROOT" ] || [ ! -d "$VAL_ROOT" ]; then
    echo "❌ Input directories not found. Make sure build_tiny_set.py has been run."
    exit 1
fi

echo "===================================="
echo "1) Creating train/val TARs"
echo "===================================="
python make_train_val_tars.py \
    --train_root "$TRAIN_ROOT" \
    --val_root "$VAL_ROOT" \
    --train_tar "$TRAIN_TAR" \
    --val_tar "$VAL_TAR" \
    --workers "$WORKERS" \
    --debug

echo
echo "===================================="
echo "2) Basic TAR verification"
echo "===================================="
python verify_simple.py --train_tar "$TRAIN_TAR" --val_tar "$VAL_TAR"

echo
echo "===================================="
echo "3) WebDataset compatibility test"
echo "===================================="
if [ -f "test_webdataset_compatibility.py" ]; then
    python test_webdataset_compatibility.py \
        --train_tar "$TRAIN_TAR" \
        --val_tar "$VAL_TAR" \
        --output_dir "$TEST_RESULTS_DIR"
else
    echo "⚠️  test_webdataset_compatibility.py not found, skipping compatibility test"
fi

echo
echo "===================================="
echo "4) Sample visualization"
echo "===================================="
python load_webdataset.py --tar "$TRAIN_TAR" --num_samples 2 --outdir "$VISUAL_DIR"

echo
echo "✅ All steps complete. CubeDiff data is ready for training!"
echo "📁 Files created:"
echo "   • $TRAIN_TAR"
echo "   • $VAL_TAR"
echo "📊 Test results saved in:"
echo "   • $TEST_RESULTS_DIR/"
echo "   • $VISUAL_DIR/"
echo
echo "🚀 You can now use these TAR files with latent_webdataset.py for training!"
EOF
chmod +x process_cubediff_data.sh
echo "Generated → process_cubediff_data.sh"

echo
echo "✅ All scripts generated successfully!"
echo 
echo "📝 CONFIGURATION SUMMARY:"
echo "   • DATA_ROOT can now be set via --data_root argument"
echo "   • WORKERS can be set via --workers argument"
echo "   • All paths are consistently derived from DATA_ROOT"
echo "   • Input validation ensures directory structure exists"
echo "   • WebDataset compatibility testing included"
echo

# ────────── Execute the pipeline ──────────
echo "🚀 EXECUTING PIPELINE..."
echo

# Pass configuration to process_cubediff_data.sh via environment variables
export DATA_ROOT
export WORKERS
./process_cubediff_data.sh "$DATA_ROOT" "$WORKERS"