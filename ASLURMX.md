# ASLURMX.md

Reference guide for AutoSlurm X (`aslurmx`) — a CLI tool and Python API for generating and submitting SLURM batch scripts on HPC clusters.

**Source**: https://github.com/the16thpythonist/AutoSlurm (branch: `aslurmx`)
**Local clone**: `/tmp/AutoSlurm`

---

## Overview

AutoSlurm X automates SLURM job submission by:

1. Loading a **cluster config** (YAML) that defines partition, memory, time, GPUs, etc.
2. Rendering **Jinja2 templates** into SLURM batch scripts
3. Assigning **CUDA_VISIBLE_DEVICES** automatically for multi-GPU parallel tasks
4. Supporting **hyperparameter sweeps** (paired or grid) via special syntax
5. Implementing **chain jobs** for tasks exceeding time limits
6. Providing both a **CLI** (`aslurmx`) and a **Python API** (`ASlurmSubmitter`)

## Installation

```bash
pip install git+https://github.com/the16thpythonist/AutoSlurm.git@aslurmx
```

Requires Python 3.9+. Key dependencies: `hydra-core`, `pydantic>=2.5.3`, `rich_click>=1.8.8`, `jinja2>=3.0.0`.

On first run, `aslurmx` creates `~/.config/auto_slurm/` with default configs and templates.

---

## CLI Usage

### Basic Syntax

```bash
aslurmx [GLOBAL_OPTIONS] COMMAND [ARGS...]
```

### Global Options

| Option | Short | Description |
|--------|-------|-------------|
| `--config-name TEXT` | `-cn` | Cluster config name (e.g., `haicore_1gpu`) |
| `--overwrite-fillers TEXT` | `-o` | Override config values: `key=val,key2=val2` |
| `--same` | `-s` | Put all commands in one SLURM job |
| `--gpus-per-task INT` | `-gpt` | GPUs assigned per command |
| `--num-gpus INT` | `-ng` | Override total GPU count |
| `--max-tasks INT` | `-mt` | Max parallel tasks per job |
| `--archive-path PATH` | | Where to store generated scripts |
| `--dry-run` | `-d` | Generate scripts without submitting |
| `--exclude TEXT` | `-x` | Exclude SLURM nodes |
| `--version` | `-v` | Show version |

### Commands

#### `cmd` — Submit a shell command as a SLURM job

```bash
# Single command
aslurmx -cn haicore_1gpu cmd python train.py --epochs=100

# Multiple commands (parallel in one job, one per GPU)
aslurmx -cn horeka_4gpu \
    cmd python train1.py \
    cmd python train2.py \
    cmd python train3.py \
    cmd python train4.py

# Repeat a command N times (cmdNx syntax)
aslurmx -cn haicore_4gpu cmd5x python train.py --seed=\$RANDOM
```

#### `interactive` — Start an interactive shell job

```bash
aslurmx -cn haicore_1gpu interactive
# Then attach: srun --jobid <JOB_ID> --pty bash
```

#### `config list` — Show available cluster configs

```bash
aslurmx config list
```

#### `config edit CONFIG_NAME` — Edit a config file

```bash
aslurmx config edit haicore_1gpu
```

#### `config where` — Show config directory paths

```bash
aslurmx config where
```

---

## Hyperparameter Sweeps

### Paired List `<[...]>` — Values are zipped

```bash
aslurmx -cn horeka_4gpu cmd python train.py \
    --lr='<[0.1,0.01,0.001]>' \
    --batch='<[32,64,128]>'
# Creates 3 commands: (lr=0.1, batch=32), (lr=0.01, batch=64), (lr=0.001, batch=128)
```

All paired lists must have the same length.

### Grid Search `<{...}>` — Cartesian product

```bash
aslurmx -cn horeka_4gpu cmd python train.py \
    --lr='<{0.1,0.01,0.001}>' \
    --batch='<{32,64,128}>'
# Creates 3 × 3 = 9 commands (all combinations)
```

**Cannot mix `<[...]>` and `<{...}>` in the same command.**

### Batching

When the number of expanded commands exceeds available GPUs/tasks, they are automatically split into multiple SLURM jobs. Each job runs as many parallel tasks as the config allows (e.g., 4 commands on a 4-GPU node).

---

## GPU Assignment

When `gpus_per_task` is set (via config or `--gpus-per-task`), each command gets dedicated GPU(s):

```bash
# Config has NO_gpus=4, gpus_per_task=1
aslurmx -cn horeka_4gpu cmd python s1.py cmd python s2.py cmd python s3.py cmd python s4.py
```

Generated script assigns:
```bash
SLURM_SUBMIT_TASK_INDEX=0 CUDA_VISIBLE_DEVICES=0 python s1.py &
SLURM_SUBMIT_TASK_INDEX=1 CUDA_VISIBLE_DEVICES=1 python s2.py &
SLURM_SUBMIT_TASK_INDEX=2 CUDA_VISIBLE_DEVICES=2 python s3.py &
SLURM_SUBMIT_TASK_INDEX=3 CUDA_VISIBLE_DEVICES=3 python s4.py &
wait
```

---

## Chain Jobs (Long-Running Tasks)

For experiments that exceed SLURM time limits, chain jobs automatically re-submit.

### In your training script:

```python
from auto_slurm.helpers import start_run, write_resume_file

timer = start_run(time_limit=10)  # hours

for epoch in range(start_epoch, max_epochs):
    train_one_epoch(model)

    if timer.time_limit_reached() and epoch < max_epochs - 1:
        save_checkpoint(model, epoch)
        write_resume_file(
            f'python train.py --resume=checkpoint.pt --start_epoch={epoch+1}'
        )
        break
```

### How it works:

1. `write_resume_file()` creates `.aslurm/{SLURM_JOB_ID}_{TASK_INDEX}.resume`
2. The main SLURM script detects these files after all tasks finish
3. It auto-submits `resume_N.sh` which reads the resume commands
4. The chain continues until no resume files are created

### Environment variables available in your script:

| Variable | Description |
|----------|-------------|
| `SLURM_SUBMIT_TASK_INDEX` | Index of this task within the job (0-based) |
| `SLURM_JOB_ID` | Current SLURM job ID |
| `PREVIOUS_SLURM_ID` | Previous job's ID (set in resume jobs) |

---

## Configuration System

### Config Hierarchy (highest priority first)

1. **CLI overwrites** (`-o key=val`)
2. **Cluster config** (`~/.config/auto_slurm/configs/<name>.yaml`)
3. **Base config** (`main.yaml` via Hydra defaults)
4. **Global fillers** (`~/.config/auto_slurm/general_config.yaml`)

### Auto-detection

If `-cn` is not specified, `aslurmx` runs `hostname` and matches it against regex patterns in `general_config.yaml`:

```yaml
# general_config.yaml
hostname_config_mappings:
  "haicore.*": "haicore_1gpu"
  "horeka.*": "horeka_4gpu"
```

### Cluster Config Structure

```yaml
# ~/.config/auto_slurm/configs/haicore_1gpu.yaml
defaults:
  - main        # inherits from main.yaml
  - _self_

default_fillers:
  partition: 'normal'
  job_name: 'haicore'
  time: '72:00:00'
  cpus: '38'
  mem: '125400mb'
  gres: 'gpu:full:1'

NO_gpus: 1          # GPU mode: 1 GPU available
max_tasks: null      # null when using GPU mode
gpus_per_task: 1     # GPUs assigned per task
```

**GPU mode** (`NO_gpus` set): Tasks assigned to specific GPUs via CUDA_VISIBLE_DEVICES.
**CPU mode** (`max_tasks` set): Tasks run in parallel up to `max_tasks` limit, no GPU assignment.

### Available Pre-configured Clusters

- **HAICORE**: `haicore_1gpu`, `haicore_4gpu`, `haicore_halfgpu`
- **HoreKa**: `horeka_1gpu`, `horeka_4gpu`, `horeka_1gpu_h100`, `horeka_4gpu_h100`
- **BWUni**: `bwuni_1gpu_a100`, `bwuni_1gpu_h100`, `bwuni_4gpu_a100`, `bwuni_4gpu_h100`
- **JUWELS**: `juwels_4gpu`
- **Int-Nano**: `intnano_1gpu`, `intnano_2gpu`, `intnano_3gpu_a100`

### Common Overwrite Examples

```bash
# Change time limit and memory
aslurmx -cn haicore_1gpu -o time=10:00:00,mem=32G cmd python train.py

# Set conda environment
aslurmx -cn haicore_1gpu -o conda_env=my_env cmd python train.py

# Custom partition
aslurmx -cn horeka_4gpu -o partition=dev_gpu_4 cmd python train.py
```

### Virtual Environment Auto-Detection

If a `.venv` directory exists in the current working directory, `aslurmx` automatically activates it in the generated script (unless overridden).

---

## Python API (ASlurmSubmitter)

For programmatic job submission from Python scripts:

```python
from auto_slurm.aslurmx import ASlurmSubmitter

submitter = ASlurmSubmitter(
    config_name='haicore_4gpu',    # cluster config
    batch_size=4,                   # commands per SLURM job
    randomize=False,                # shuffle commands before batching
    gpus_per_task=1,                # GPUs per command
)

# Queue commands
for lr in [1e-3, 1e-4, 1e-5]:
    for seed in [42, 43, 44]:
        submitter.add_command(
            f'python train.py --lr={lr} --seed={seed}'
        )

# Check how many jobs will be created
print(f"Will submit {submitter.count_jobs()} SLURM jobs")

# Submit all queued commands
submitter.submit()
```

### ASlurmSubmitter Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config_name` | str | required | Cluster config to use |
| `batch_size` | int | from config | Commands per SLURM job |
| `randomize` | bool | False | Shuffle commands before batching |
| `gpus_per_task` | int | from config | GPUs per command |
| `num_gpus` | int | from config | Total available GPUs |
| `options` | dict | {} | Additional CLI options |

### ASlurmSubmitter Methods

| Method | Description |
|--------|-------------|
| `add_command(cmd: str)` | Queue a command for submission |
| `submit()` | Submit all queued commands as SLURM jobs |
| `submit_batch(commands: list)` | Submit a single batch of commands |
| `count_jobs() -> int` | Estimate number of SLURM jobs needed |

---

## Generated Script Structure

Scripts are stored in `.aslurm/{YYYY-MM-DD_HH-MM-SS}_{UUID}/`:

```
.aslurm/
└── 2024-03-15_10-30-00_abc123/
    ├── main_0.sh          # First batch SLURM script
    ├── main_1.sh          # Second batch (if needed)
    ├── resume_0.sh        # Resume script for chain jobs
    └── resume_1.sh
```

After execution, SLURM creates log files:
```
slurm-12345.out            # Main job output
slurm-12345-0.out          # Task 0 output
slurm-12345-1.out          # Task 1 output
```

---

## Template System

Templates use Jinja2 with inheritance:

- **`base.sh.j2`**: SBATCH directives, environment setup, command execution loop
- **`main.sh.j2`**: Extends base, adds `wait` and resume file detection
- **`resume.sh.j2`**: Extends base, reads commands from `.resume` files

### Custom Templates

Override templates by placing files in `~/.config/auto_slurm/templates/`:
```
~/.config/auto_slurm/templates/
├── base.sh.j2
├── main.sh.j2
└── resume.sh.j2
```

User templates take priority over packaged ones.

### Template Variables

| Variable | Description |
|----------|-------------|
| `fillers.*` | All config values (partition, time, mem, gres, etc.) |
| `commands` | List of shell commands to execute |
| `options` | CLI options dict |
| `gpus` | GPU assignment list |

---

## Key Modules Reference

| Module | Description |
|--------|-------------|
| `auto_slurm/aslurmx.py` | CLI entry point (`ASlurm` class) and Python API (`ASlurmSubmitter`) |
| `auto_slurm/config.py` | Pydantic config models (`Config`, `GeneralConfig`, `AutoSlurmConfig`) |
| `auto_slurm/helpers.py` | Utilities: `expand_commands()`, `Batched`, `RunTimer`, `write_resume_file()`, `create_slurm_jobs()` |
| `auto_slurm/templates/` | Jinja2 templates for SLURM scripts |
| `auto_slurm/configs/` | Default cluster YAML configs |

---

## Quick Reference: Common Workflows

### Submit a training run
```bash
aslurmx -cn haicore_1gpu cmd python train.py +experiment=planar
```

### Run 4 experiments in parallel on 4 GPUs
```bash
aslurmx -cn horeka_4gpu \
    cmd python train.py --seed=1 \
    cmd python train.py --seed=2 \
    cmd python train.py --seed=3 \
    cmd python train.py --seed=4
```

### Grid search over hyperparameters
```bash
aslurmx -cn horeka_4gpu cmd python train.py \
    --lr='<{0.1,0.01,0.001}>' \
    --weight_decay='<{0,1e-4}>'
```

### Dry run (inspect generated scripts)
```bash
aslurmx -cn haicore_1gpu -d cmd python train.py
```

### Override time and memory
```bash
aslurmx -cn haicore_1gpu -o time=48:00:00,mem=64G cmd python train.py
```

### Long-running chain job
```bash
# In train.py: use start_run() and write_resume_file()
aslurmx -cn haicore_1gpu cmd python train.py --max_epochs=1000
```
