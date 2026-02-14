# Nanohead Experiment

## Overview

This experiment tests the hypothesis that mixing "nanoheads" (attention heads with very low dimensions like 2-3) with normal attention heads can improve model performance.

## Hypothesis

Nanoheads are low-rank attention heads that can efficiently learn simple patterns (e.g., "look at antonyms", "copy previous token") while using far fewer parameters than standard heads. By mixing them with normal heads, we might achieve better parameter efficiency.

## Key Features

- **No RoPE on nanoheads**: Nanoheads don't receive rotary positional embeddings. If positional information is useful, the MLP can inject it from normal heads.
- **Configurable proportion**: Control what fraction of attention QKV parameters go to nanoheads vs normal heads.
- **Automatic parameter balancing**: The implementation ensures normal heads always have even dimensions (required for RoPE) while maintaining the total embedding dimension.

## Implementation Details

### Modified Files

1. **nanochat/gpt.py**
   - Added `nanohead_proportion` and `nanohead_dim` to `GPTConfig`
   - Implemented `calculate_nanohead_split()` to compute head allocation
   - Modified `CausalSelfAttention` to handle both normal and nano heads
   - Updated `init_weights()` to initialize both head types

2. **scripts/base_train.py**
   - Added `--nanohead-proportion` and `--nanohead-dim` CLI arguments
   - Pass nanohead config to model initialization

3. **runs/nanohead_experiment.py**
   - Experiment script that trains models with proportions: 0, 0.2, 0.4, 0.6, 0.8
   - Automatically estimates depth for ~50M parameter model
   - Collects results from wandb and generates plots

## How to Run

### Quick Start

```bash
# Make sure you're in the nanochat directory and venv is activated
source .venv/bin/activate

# Run the experiment (bash script wrapper)
bash runs/run_nanohead_experiment.sh
```

### Manual Run

```bash
# Activate environment
source .venv/bin/activate

# Run the Python script directly
python runs/nanohead_experiment.py
```

### Custom Configuration

To run a single model with nanoheads:

```bash
python -m scripts.base_train \
    --depth=12 \
    --nanohead-proportion=0.3 \
    --nanohead-dim=3 \
    --run=my_nanohead_test \
    --num-iterations=500
```

## Experiment Configuration

- **Target model size**: ~50M parameters
- **Nanohead dimension**: 3
- **Proportions tested**: 0.0, 0.2, 0.4, 0.6, 0.8
- **Training iterations**: 500 (short run for quick results)

## Output

After the experiment completes, you'll have:

1. **5 wandb runs** logged to your `nanochat` project:
   - `nanohead_p00_dXX` (baseline, no nanoheads)
   - `nanohead_p20_dXX` (20% nanoheads)
   - `nanohead_p40_dXX` (40% nanoheads)
   - `nanohead_p60_dXX` (60% nanoheads)
   - `nanohead_p80_dXX` (80% nanoheads)

2. **Plot**: `nanohead_experiment_results.png`
   - Shows final validation loss (bits per byte) vs nanohead proportion
   - Includes value annotations on each point

3. **JSON results**: `nanohead_experiment_results.json`
   - Raw data for further analysis

## Technical Details

### Parameter Calculation

When `nanohead_proportion=p`:
- `nano_output_dim ≈ p × n_embd`
- `normal_output_dim = n_embd - nano_output_dim`
- Normal head dimension is chosen to be even (for RoPE compatibility)
- GQA ratio is maintained for both head types

### Attention Processing

For each layer:
1. Normal heads:
   - Compute Q, K, V projections
   - Apply RoPE to Q and K
   - Apply QK normalization
   - Run Flash Attention

2. Nano heads:
   - Compute Q, K, V projections
   - Skip RoPE (no positional encoding)
   - Apply QK normalization
   - Run Flash Attention

3. Concatenate outputs from both head types
4. Project back to `n_embd` with shared `c_proj`

## Expected Results

If the hypothesis is correct, we should see:
- Better validation loss with moderate nanohead proportions (e.g., 0.2-0.4)
- Diminishing returns or degradation at very high proportions (e.g., 0.8)

## Troubleshooting

### wandb not logging

```bash
# Login to wandb
wandb login

# Or set API key
export WANDB_API_KEY=your_key_here
```

### Out of memory

Reduce the model size or batch size:
```bash
# Edit runs/nanohead_experiment.py and change:
target_params = 25e6  # Reduce from 50M to 25M
```

Or modify training settings in the experiment script to use smaller batch sizes.

### Dimension errors

If you get errors about incompatible dimensions:
- The implementation should handle all cases automatically
- File an issue if you encounter problems with specific configurations

## Future Experiments

Ideas to explore:
- Different nanohead dimensions (2, 4, 5)
- Varying proportions per layer
- Longer training runs to see if benefits emerge over time
- Different model scales (10M, 100M, 500M)
- Analysis of what patterns nanoheads learn vs normal heads
