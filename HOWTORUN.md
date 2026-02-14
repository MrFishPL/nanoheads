# How to Run the Nanohead Experiment

## Quick Start

```bash
# 1. Make sure you're in the nanochat directory
cd /workspace/nanoheads

# 2. Activate the virtual environment (if not already active)
source .venv/bin/activate

# 3. (Optional) Login to wandb for logging
wandb login

# 4. Run the experiment
bash runs/run_nanohead_experiment.sh
```

## What This Does

The experiment will:
1. **Estimate** the depth needed for a ~50M parameter model
2. **Train 5 models** with different nanohead proportions:
   - 0.0 (baseline, no nanoheads)
   - 0.2 (20% of attention params are nanoheads)
   - 0.4 (40% of attention params are nanoheads)
   - 0.6 (60% of attention params are nanoheads)
   - 0.8 (80% of attention params are nanoheads)
3. **Log** each run to wandb project "nanochat"
4. **Generate** a plot showing final validation loss vs proportion

## Expected Runtime

- **Per model**: ~5-10 minutes on a single GPU (depending on hardware)
- **Total experiment**: ~25-50 minutes for all 5 models
- Uses 500 training iterations per model (configurable)

## Output Files

After completion, you'll find:
- `nanohead_experiment_results.png` - Plot of results
- `nanohead_experiment_results.json` - Raw results data
- 5 runs in your wandb dashboard

## Checking Results

### In wandb

Visit https://wandb.ai and check your "nanochat" project. You'll see runs named:
- `nanohead_p00_dXX` (baseline)
- `nanohead_p20_dXX`
- `nanohead_p40_dXX`
- `nanohead_p60_dXX`
- `nanohead_p80_dXX`

### Local plot

```bash
# View the generated plot
open nanohead_experiment_results.png  # macOS
xdg-open nanohead_experiment_results.png  # Linux
```

### Results JSON

```bash
cat nanohead_experiment_results.json
```

## Customizing the Experiment

### Different model size

Edit `runs/nanohead_experiment.py`:
```python
target_params = 100e6  # Change from 50M to 100M
```

### Different proportions

Edit `runs/nanohead_experiment.py`:
```python
proportions = [0.0, 0.1, 0.3, 0.5, 0.7]  # Test different values
```

### Longer training

Edit `runs/nanohead_experiment.py`:
```python
num_iterations = 1000  # Increase from 500 to 1000
```

### Different nanohead dimension

Edit `runs/nanohead_experiment.py`:
```python
nanohead_dim = 2  # Test with dim=2 instead of dim=3
```

## Running a Single Model

To test a single configuration:

```bash
python -m scripts.base_train \
    --depth=12 \
    --nanohead-proportion=0.3 \
    --nanohead-dim=3 \
    --run=test_nanohead \
    --num-iterations=500 \
    --eval-every=100
```

## Troubleshooting

### wandb offline mode

If you don't want to use wandb:
```bash
export WANDB_MODE=disabled
bash runs/run_nanohead_experiment.sh
```

Note: Without wandb, results won't be automatically collected and plotted.

### Out of memory

Reduce the model size in the experiment script:
```python
target_params = 25e6  # Use smaller model
```

Or reduce batch size:
```bash
python -m scripts.base_train \
    --depth=12 \
    --nanohead-proportion=0.3 \
    --device-batch-size=16  # Reduce from 32
```

### Permission errors

Make the script executable:
```bash
chmod +x runs/run_nanohead_experiment.sh
```

## What to Look For

If the hypothesis is correct, you should see:
- **Sweet spot**: Better validation loss at moderate proportions (0.2-0.4)
- **Degradation**: Worse loss at very high proportions (0.8)
- **Efficiency**: Similar or better performance with fewer total parameters

If the hypothesis is wrong:
- Baseline (0.0) will perform best
- Loss will monotonically increase with nanohead proportion

Either way, you'll have interesting data about the impact of low-rank attention heads!

## Next Steps

After the experiment:
1. Analyze which proportion works best
2. Try longer training runs with the best proportion
3. Visualize what patterns the nanoheads learn
4. Scale up to larger models (100M, 500M params)
5. Try different nanohead dimensions (2, 4, 5)
