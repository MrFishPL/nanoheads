# Nanohead Experiment - Quick Usage Guide

## Basic Usage

```bash
# Run with default settings (50M parameters)
bash runs/run_nanohead_experiment.sh
```

## Customizing Parameters

### Change Model Size

```bash
# 100M parameters
bash runs/run_nanohead_experiment.sh --target-params 100

# 500M parameters
bash runs/run_nanohead_experiment.sh --target-params 500

# 1B parameters
bash runs/run_nanohead_experiment.sh --target-params 1000
```

### Change Nanohead Dimension

```bash
# Test with dim=2 nanoheads
bash runs/run_nanohead_experiment.sh --nanohead-dim 2

# Test with dim=4 nanoheads
bash runs/run_nanohead_experiment.sh --nanohead-dim 4
```

### Change Proportions Tested

```bash
# Test only 3 proportions
bash runs/run_nanohead_experiment.sh --proportions "0.0,0.3,0.5"

# Test more granular proportions
bash runs/run_nanohead_experiment.sh --proportions "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8"

# Test only extreme cases
bash runs/run_nanohead_experiment.sh --proportions "0.0,0.5,1.0"
```

### Change Training Duration (via Data-to-Parameter Ratio)

Training duration is automatically calculated based on model size and the data-to-parameter ratio (Chinchilla scaling laws).

```bash
# Shorter training (lower ratio)
bash runs/run_nanohead_experiment.sh --target-param-data-ratio 5

# Default (compute-optimal, slightly undertrained)
bash runs/run_nanohead_experiment.sh --target-param-data-ratio 10.5

# Chinchilla-optimal (20 tokens per parameter)
bash runs/run_nanohead_experiment.sh --target-param-data-ratio 20

# Longer training (overtrained, may give better final performance)
bash runs/run_nanohead_experiment.sh --target-param-data-ratio 30
```

**Note**: Higher ratios = more training tokens = longer training time but potentially better performance.

## Combined Examples

### Quick 100M test (shorter training)
```bash
bash runs/run_nanohead_experiment.sh \
    --target-params 100 \
    --proportions "0.0,0.2,0.4" \
    --target-param-data-ratio 5
```

### Large-scale experiment (500M with Chinchilla-optimal training)
```bash
bash runs/run_nanohead_experiment.sh \
    --target-params 500 \
    --nanohead-dim 3 \
    --proportions "0.0,0.2,0.4,0.6,0.8" \
    --target-param-data-ratio 20
```

### Test different nanohead dimensions
```bash
# First run with dim=2
bash runs/run_nanohead_experiment.sh \
    --target-params 50 \
    --nanohead-dim 2 \
    --proportions "0.0,0.3,0.6"

# Then with dim=3
bash runs/run_nanohead_experiment.sh \
    --target-params 50 \
    --nanohead-dim 3 \
    --proportions "0.0,0.3,0.6"

# Then with dim=4
bash runs/run_nanohead_experiment.sh \
    --target-params 50 \
    --nanohead-dim 4 \
    --proportions "0.0,0.3,0.6"
```

## Python Script Direct Usage

You can also call the Python script directly:

```bash
python runs/nanohead_experiment.py \
    --target-params 100 \
    --nanohead-dim 3 \
    --proportions "0.0,0.2,0.4,0.6,0.8" \
    --target-param-data-ratio 10.5
```

## Help

```bash
# Show all available options
python runs/nanohead_experiment.py --help
```

## Parameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--target-params` | 50 | Target model size in millions of parameters |
| `--nanohead-dim` | 3 | Dimension of each nanohead (2, 3, 4, etc.) |
| `--proportions` | 0.0,0.2,0.4,0.6,0.8 | Comma-separated list of proportions to test |
| `--target-param-data-ratio` | 10.5 | Data-to-parameter ratio (Chinchilla=20, higher=longer training) |

## Output

All runs will be logged to wandb and results will be saved to:
- `nanohead_experiment_results.png` - Plot of results
- `nanohead_experiment_results.json` - Raw data

## Estimated Runtime

Training duration is automatically calculated based on the data-to-parameter ratio. Times shown are for ratio=10.5 (default).

| Model Size | Ratio | Approx Tokens | Time per Model | Total (5 proportions) |
|------------|-------|---------------|----------------|----------------------|
| 50M | 10.5 | ~525M | ~5-10 min | ~25-50 min |
| 100M | 10.5 | ~1B | ~10-20 min | ~50-100 min |
| 500M | 10.5 | ~5.25B | ~60-90 min | ~5-7.5 hours |
| 500M | 20 (Chinchilla) | ~10B | ~120-180 min | ~10-15 hours |

**Formula**: Training tokens ≈ Model parameters × ratio

*Times are approximate and depend on hardware (GPU type, batch size, etc.)*
