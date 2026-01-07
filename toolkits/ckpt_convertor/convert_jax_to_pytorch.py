#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Convert JAX/Orbax checkpoint to PyTorch safetensors format.

Handles multi-device checkpoints by loading directly via tensorstore,
bypassing JAX device mesh requirements.

Usage:
    python toolkits/ckpt_convertor/convert_jax_to_pytorch.py \
        --jax_checkpoint model/pi05_sort_tubes_1124/params \
        --output model/pi05_sort_tubes_1124/model.safetensors

Or as a module:
    from toolkits.ckpt_convertor.convert_jax_to_pytorch import convert_jax_to_pytorch
    convert_jax_to_pytorch("model/pi05_sort_tubes_1124/params", "model/pi05_sort_tubes_1124/model.safetensors")
"""

# Set JAX to CPU-only mode BEFORE importing JAX
# This helps avoid device mesh issues
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_checkpoint_via_tensorstore(checkpoint_path: Path) -> dict:
    """
    Load checkpoint by directly reading tensorstore arrays.
    This bypasses device sharding requirements entirely.
    
    Works with checkpoints saved on multiple GPUs.
    """
    import ast
    import tensorstore as ts
    
    checkpoint_path = Path(checkpoint_path)
    metadata_path = checkpoint_path / "_METADATA"
    if not metadata_path.exists():
        raise FileNotFoundError(f"_METADATA not found at {metadata_path}")
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    tree_metadata = metadata.get("tree_metadata", {})
    
    print(f"Found {len(tree_metadata)} parameters in checkpoint metadata")
    
    # Build nested dict structure
    result = {}
    loaded_count = 0
    failed_count = 0
    
    for key_tuple_str, value_info in tree_metadata.items():
        # Parse key tuple string like "('params', 'PaliGemma', ...)"
        try:
            key_path = ast.literal_eval(key_tuple_str)
            if not isinstance(key_path, tuple):
                key_path = (key_path,)
        except:
            key_path = tuple(k.strip() for k in key_tuple_str.strip("()").replace("'", "").split(", ") if k.strip())
        
        # Get value metadata
        value_metadata = value_info.get("value_metadata", {})
        
        # Skip if not an array
        if "write_shape" not in value_metadata:
            continue
        
        # Try multiple path formats that orbax might use
        path_formats = [
            "/".join(str(k) for k in key_path),  # params/layer/weight
            ".".join(str(k) for k in key_path),  # params.layer.weight
            key_tuple_str,  # Original tuple format
        ]
        
        arr = None
        for path_format in path_formats:
            try:
                spec = {
                    "driver": "ocdbt",
                    "base": f"file://{checkpoint_path.absolute()}",
                    "path": path_format,
                }
                store = ts.open(spec).result()
                arr = np.array(store.read().result())
                break
            except Exception:
                continue
        
        if arr is None:
            # Try zarr3 driver as fallback
            for path_format in path_formats:
                try:
                    spec = {
                        "driver": "zarr3",
                        "kvstore": {
                            "driver": "ocdbt",
                            "base": f"file://{checkpoint_path.absolute()}",
                            "path": path_format,
                        }
                    }
                    store = ts.open(spec).result()
                    arr = np.array(store.read().result())
                    break
                except Exception:
                    continue
        
        if arr is not None:
            # Build nested dict
            current = result
            for k in key_path[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[key_path[-1]] = arr
            loaded_count += 1
        else:
            failed_count += 1
            if failed_count <= 5:
                print(f"Warning: Could not load {key_path}")
            elif failed_count == 6:
                print("... (suppressing further warnings)")
    
    print(f"Successfully loaded {loaded_count} parameters, {failed_count} failed")
    
    if not result:
        raise RuntimeError("Failed to load any parameters from checkpoint")
    
    return result


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten nested dictionary with dot-separated keys."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def convert_jax_array_to_torch(arr: Any) -> torch.Tensor:
    """Convert JAX array or numpy array to PyTorch tensor."""
    if hasattr(arr, "numpy"):
        # JAX array
        arr = np.asarray(arr)
    elif not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    
    # Convert to torch tensor
    return torch.from_numpy(arr.copy())


def load_orbax_checkpoint(checkpoint_path: str) -> dict:
    """Load JAX/Orbax checkpoint and return as nested dict of numpy arrays.
    
    Handles checkpoints saved on multiple devices by using tensorstore directly,
    which bypasses JAX device mesh requirements.
    """
    checkpoint_path = Path(checkpoint_path)
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # For multi-device checkpoints, go straight to tensorstore
    # This is the most reliable method that doesn't require matching devices
    print("Using tensorstore to load checkpoint (bypasses device requirements)...")
    
    try:
        return load_checkpoint_via_tensorstore(checkpoint_path)
    except Exception as e:
        print(f"Tensorstore loading failed: {e}")
        print("Trying orbax restore as fallback...")
        
        # Fallback to orbax (may work for single-device checkpoints)
        try:
            import jax
            import orbax.checkpoint as ocp
            
            print(f"Available JAX devices: {jax.devices()}")
            
            checkpointer = ocp.StandardCheckpointer()
            metadata = checkpointer.metadata(checkpoint_path)
            
            # Create restore args for numpy arrays
            restore_args = jax.tree.map(
                lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray),
                metadata,
            )
            
            restored = checkpointer.restore(
                checkpoint_path,
                args=ocp.args.StandardRestore(restore_args),
            )
            return restored
            
        except Exception as e2:
            print(f"Orbax restore also failed: {e2}")
            raise RuntimeError(
                f"Could not load checkpoint from {checkpoint_path}. "
                f"Tensorstore error: {e}, Orbax error: {e2}"
            )


def convert_params_to_pytorch_dict(params: dict) -> dict[str, torch.Tensor]:
    """Convert JAX params dict to flat PyTorch state dict."""
    # Flatten the nested params dict
    flat_params = flatten_dict(params)
    
    # Convert each array to PyTorch tensor
    pytorch_dict = {}
    for key, value in flat_params.items():
        if value is not None:
            pytorch_dict[key] = convert_jax_array_to_torch(value)
    
    return pytorch_dict


def convert_jax_to_pytorch(
    jax_checkpoint_path: str,
    output_path: str,
    verbose: bool = True,
) -> None:
    """
    Convert JAX/Orbax checkpoint to PyTorch safetensors format.
    
    Args:
        jax_checkpoint_path: Path to JAX checkpoint directory (containing params/)
        output_path: Path to output safetensors file
        verbose: Whether to print progress
    """
    import safetensors.torch
    
    # Handle paths
    jax_path = Path(jax_checkpoint_path)
    output_path = Path(output_path)
    
    # Check if we need to append /params
    if not (jax_path / "_METADATA").exists() and (jax_path / "params" / "_METADATA").exists():
        jax_path = jax_path / "params"
    
    if not jax_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {jax_path}")
    
    # Load JAX checkpoint
    if verbose:
        print(f"Loading JAX checkpoint from: {jax_path}")
    
    params = load_orbax_checkpoint(str(jax_path))
    
    if verbose:
        print(f"Checkpoint loaded. Converting to PyTorch format...")
    
    # Convert to PyTorch state dict
    pytorch_state_dict = convert_params_to_pytorch_dict(params)
    
    if verbose:
        print(f"Converted {len(pytorch_state_dict)} parameters")
        
        # Print some stats
        total_params = sum(t.numel() for t in pytorch_state_dict.values())
        print(f"Total parameters: {total_params:,}")
    
    # Save as safetensors
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(pytorch_state_dict, str(output_path))
    
    if verbose:
        print(f"Saved PyTorch checkpoint to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert JAX/Orbax checkpoint to PyTorch safetensors"
    )
    parser.add_argument(
        "--jax_checkpoint", "-i",
        type=str,
        required=True,
        help="Path to JAX checkpoint directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output path for safetensors file"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    convert_jax_to_pytorch(
        args.jax_checkpoint,
        args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

