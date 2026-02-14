#!/usr/bin/env python3
"""
Nanohead experiment: Train models with different proportions of nanoheads.

This script:
1. Estimates depth for ~50M parameter model
2. Trains 5 models with nanohead proportions: 0, 0.2, 0.4, 0.6, 0.8
3. Logs all runs to the same wandb project
4. Creates a plot of final loss vs nanohead proportion
"""

import os
import sys
import subprocess
import json
import argparse
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# Add parent directory to path to import nanochat
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt import GPT, GPTConfig
from nanochat.common import print0


def estimate_depth_for_target_params(target_params, aspect_ratio=64, vocab_size=32768,
                                       sequence_len=2048, nanohead_proportion=0.0, nanohead_dim=3):
    """
    Estimate the depth needed to achieve approximately target_params parameters.
    Uses binary search to find the closest depth.
    """
    print0(f"Estimating depth for ~{target_params/1e6:.1f}M parameters (nanohead_proportion={nanohead_proportion})...")

    def get_num_params(depth):
        """Get parameter count for a given depth."""
        n_embd = depth * aspect_ratio
        n_head = max(1, n_embd // 128)  # Default head_dim target is 128
        n_kv_head = n_head  # MHA by default

        config = GPTConfig(
            n_layer=depth,
            n_embd=n_embd,
            n_head=n_head,
            n_kv_head=n_kv_head,
            vocab_size=vocab_size,
            sequence_len=sequence_len,
            nanohead_proportion=nanohead_proportion,
            nanohead_dim=nanohead_dim,
        )

        # Create model in meta device to avoid memory allocation
        with torch.device('meta'):
            model = GPT(config)

        param_dict = model.num_scaling_params()
        return param_dict['total']

    # Binary search for the right depth
    low, high = 1, 100
    best_depth = low
    best_diff = float('inf')

    for _ in range(20):  # Limit iterations
        mid = (low + high) // 2
        num_params = get_num_params(mid)
        diff = abs(num_params - target_params)

        if diff < best_diff:
            best_diff = diff
            best_depth = mid

        if num_params < target_params:
            low = mid + 1
        else:
            high = mid - 1

    final_params = get_num_params(best_depth)
    print0(f"Selected depth={best_depth} with {final_params/1e6:.2f}M parameters")
    return best_depth, final_params


def run_training(depth, nanohead_proportion, nanohead_dim, wandb_run_name, target_param_data_ratio=10.5,
                 device_batch_size=32, nproc_per_node=1, model_tag=None, nanohead_ablation=False):
    """
    Run a single training run with the specified configuration.
    Uses Chinchilla-optimal training duration based on target_param_data_ratio.
    Returns the final validation loss.
    """
    print0(f"\n{'='*80}")
    print0(f"Starting training: proportion={nanohead_proportion}, depth={depth}, run={wandb_run_name}")
    print0(f"{'='*80}\n")

    # Build the training command.
    # Let nanochat calculate optimal iterations based on Chinchilla laws.
    # Use torchrun for multi-GPU execution when nproc_per_node > 1.
    if nproc_per_node > 1:
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={nproc_per_node}",
            "-m", "scripts.base_train",
            "--",  # stop torchrun arg parsing; remaining args go to scripts.base_train
        ]
    else:
        cmd = ["python", "-m", "scripts.base_train"]

    cmd += [
        "--depth", str(depth),
        "--nanohead-proportion", str(nanohead_proportion),
        "--nanohead-dim", str(nanohead_dim),
        "--run", wandb_run_name,
        "--target-param-data-ratio", str(target_param_data_ratio),  # Chinchilla-style compute-optimal
        "--device-batch-size", str(device_batch_size),  # Per-device batch size
        "--eval-every", "250",  # Default eval frequency
        "--core-metric-every", "-1",  # Disable CORE metric for speed
        "--sample-every", "-1",  # Disable sampling
        "--save-every", "-1",  # Don't save intermediate checkpoints
    ]
    if model_tag is not None:
        cmd += ["--model-tag", str(model_tag)]
    if nanohead_ablation:
        cmd += ["--nanohead-ablation"]

    print0(f"Launch command: {' '.join(cmd)}")

    # Run the training
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print0(f"Warning: Training failed for proportion={nanohead_proportion}")
        return None

    # Try to extract final loss from wandb or checkpoint
    # For now, we'll parse it from the wandb run after all runs complete
    print0(f"Training completed for proportion={nanohead_proportion}")
    return wandb_run_name


def collect_results_from_wandb(run_names):
    """
    Collect final validation losses from wandb runs.
    Returns dict mapping proportion to final val_bpb.
    """
    import wandb

    print0("\nCollecting results from wandb...")
    results = {}

    api = wandb.Api()

    for proportion, run_name in run_names.items():
        if run_name == "dummy":
            continue

        try:
            # Get the run from wandb
            # Assuming runs are in your default entity/project
            runs = api.runs(path=f"nanochat", filters={"display_name": run_name})

            if len(runs) > 0:
                run = runs[0]
                # Get the final val_bpb
                history = run.scan_history(keys=["val_bpb"])
                val_losses = [row["val_bpb"] for row in history if "val_bpb" in row]

                if val_losses:
                    final_loss = val_losses[-1]
                    results[proportion] = final_loss
                    print0(f"  Proportion {proportion}: final val_bpb = {final_loss:.6f}")
                else:
                    print0(f"  Warning: No val_bpb found for proportion {proportion}")
            else:
                print0(f"  Warning: Run not found for proportion {proportion}")
        except Exception as e:
            print0(f"  Error fetching results for proportion {proportion}: {e}")

    return results


def plot_results(results, output_path="nanohead_experiment_results.png"):
    """
    Create a plot of final loss vs nanohead proportion.
    """
    if not results:
        print0("No results to plot!")
        return

    proportions = sorted(results.keys())
    losses = [results[p] for p in proportions]

    plt.figure(figsize=(10, 6))
    plt.plot(proportions, losses, 'o-', linewidth=2, markersize=8, label='Validation Loss (BPB)')
    plt.xlabel('Nanohead Proportion', fontsize=12)
    plt.ylabel('Final Validation Loss (bits per byte)', fontsize=12)
    plt.title('Nanohead Experiment: Impact on Model Performance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    # Add value labels on points
    for p, l in zip(proportions, losses):
        plt.annotate(f'{l:.4f}', (p, l), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print0(f"\nPlot saved to: {output_path}")

    # Also save results to JSON
    json_path = output_path.replace('.png', '.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print0(f"Results saved to: {json_path}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Nanohead experiment runner")
    parser.add_argument("--target-params", type=float, default=50,
                        help="Target model size in millions of parameters (default: 50)")
    parser.add_argument("--nanohead-dim", type=int, default=3,
                        help="Dimension of each nanohead (default: 3)")
    parser.add_argument("--proportions", type=str, default="0.0,0.2,0.4,0.6,0.8",
                        help="Comma-separated list of nanohead proportions to test (default: 0.0,0.2,0.4,0.6,0.8)")
    parser.add_argument("--target-param-data-ratio", type=float, default=10.5,
                        help="Data-to-parameter ratio for compute-optimal training (Chinchilla=20, default: 10.5)")
    parser.add_argument("--device-batch-size", type=int, default=32,
                        help="Per-device batch size (default: 32, reduce if OOM)")
    parser.add_argument("--nproc-per-node", type=int, default=-1,
                        help="Number of GPU processes per training run "
                             "(-1 = auto-detect CUDA device count, min 1)")
    parser.add_argument("--ablation-classic-only", action="store_true",
                        help="run classic-only ablations: keep classic split implied by each proportion, disable nanoheads")
    args = parser.parse_args()

    # Configuration
    target_params = args.target_params * 1e6  # Convert from millions to actual count
    nanohead_dim = args.nanohead_dim
    proportions = [float(p.strip()) for p in args.proportions.split(',')]
    target_param_data_ratio = args.target_param_data_ratio
    device_batch_size = args.device_batch_size

    if args.nproc_per_node == -1:
        nproc_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1
        nproc_per_node = max(1, nproc_per_node)
    else:
        nproc_per_node = max(1, args.nproc_per_node)

    # Estimate depth for the baseline (no nanoheads)
    depth, actual_params = estimate_depth_for_target_params(
        target_params,
        nanohead_proportion=0.0,
        nanohead_dim=nanohead_dim
    )

    print0(f"\nExperiment configuration:")
    print0(f"  Target parameters: {target_params/1e6:.1f}M")
    print0(f"  Actual parameters (baseline): {actual_params/1e6:.2f}M")
    print0(f"  Depth: {depth}")
    print0(f"  Nanohead dimension: {nanohead_dim}")
    print0(f"  Proportions to test: {proportions}")
    print0(f"  Ablation classic-only mode: {args.ablation_classic_only}")
    print0(f"  Target param-data ratio: {target_param_data_ratio} (Chinchilla=20)")
    print0(f"  nproc_per_node (GPUs per run): {nproc_per_node}")
    print0(f"  Training will use compute-optimal iterations based on model size")
    print0(f"\n")

    # Check if we're actually going to use wandb
    wandb_project = os.environ.get("WANDB_PROJECT", "nanochat")
    use_wandb = os.environ.get("WANDB_MODE") != "disabled"

    if not use_wandb:
        print0("WARNING: wandb is disabled. Set WANDB_MODE=online to enable logging.")
        print0("Results will not be collected automatically.\n")

    # Run training for each proportion
    run_names = {}

    for proportion in proportions:
        if args.ablation_classic_only and proportion == 0.0:
            print0("Skipping p=0.0 in ablation mode (duplicate baseline).")
            continue

        # Re-estimate depth for this proportion to keep params roughly constant
        # (nanoheads have fewer params, so we might need slightly different depth)
        depth_adj, params_adj = estimate_depth_for_target_params(
            target_params,
            nanohead_proportion=proportion,
            nanohead_dim=nanohead_dim
        )

        if args.ablation_classic_only:
            run_name = f"nanoabl_p{int(proportion*100):02d}_d{depth_adj}"
            model_tag = run_name
        else:
            run_name = f"nanohead_p{int(proportion*100):02d}_d{depth_adj}"
            model_tag = None
        run_names[proportion] = run_name

        run_training(
            depth=depth_adj,
            nanohead_proportion=proportion,
            nanohead_dim=nanohead_dim,
            wandb_run_name=run_name,
            target_param_data_ratio=target_param_data_ratio,
            device_batch_size=device_batch_size,
            nproc_per_node=nproc_per_node,
            model_tag=model_tag,
            nanohead_ablation=args.ablation_classic_only,
        )

    print0("\n" + "="*80)
    print0("All training runs completed!")
    print0("="*80)

    # Collect results from wandb
    if use_wandb:
        results = collect_results_from_wandb(run_names)

        # Create plot
        if results:
            plot_results(results)
        else:
            print0("\nNo results collected. Check wandb for run data.")
    else:
        print0("\nwandb disabled - skipping result collection and plotting.")
        print0("To collect results manually, check wandb for runs:")
        for proportion, run_name in run_names.items():
            print0(f"  Proportion {proportion}: {run_name}")


if __name__ == "__main__":
    main()
