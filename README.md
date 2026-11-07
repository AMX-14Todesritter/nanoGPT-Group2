# nanoGPT Hyperparameter Experiments — Group 2

This repository contains the experimental work for our CSCE 5810 group assignment. Using [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT), we train GPT models from scratch on the character-level Shakespeare dataset. Three members cover different parts of the parameter space, completing **128 experiments** to examine how model architecture and training settings affect training loss, validation loss, generalization gap, training time, parameter count, and GPU memory usage. Generated text is also saved for qualitative analysis.

## 1. Experiment Scope

- **Model architecture:** Compare context length, Transformer depth, attention heads, and embedding dimension.
- **Training settings:** Compare batch size, training iterations, and dropout.
- **Convergence and generalization:** Record training/validation loss curves, final losses, minimum validation loss, and the loss gap.
- **Resource usage:** Measure training process duration, model parameter count, and sampled GPU memory usage.
- **Text generation:** Generate text with shared sampling settings to inspect the character-level models' output.

The repository contains experiment notebooks and result artifacts. The notebooks download nanoGPT's training source and dataset at runtime.

## 2. Full Experiment Parameter Space

The experiments use a full factorial design with two values for each of seven parameters:

| Parameter | Values | Description |
| --- | --- | --- |
| `block_size` | 64, 128 | Input context length in character tokens |
| `n_layer` | 4, 6 | Number of Transformer layers |
| `n_head` | 4, 8 | Number of attention heads per layer |
| `n_embd` | 128, 256 | Embedding and hidden representation dimension |
| `batch_size` | 8, 16 | Number of sequences processed per iteration |
| `max_iters` | 1000, 2000 | Training loop iteration limit |
| `dropout` | 0.1, 0.2 | Dropout probability |

**Total configurations: `2 × 2 × 2 × 2 × 2 × 2 × 2 = 128`.** Every configuration satisfies `n_embd % n_head == 0`.

### Member Assignments

| Member | `block_size` | `n_layer` | `n_head` | `n_embd` | `batch_size` | `max_iters` | `dropout` | Completed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Member 1 | **64** | **4** | 4 / 8 | 128 / 256 | 8 / 16 | 1000 / 2000 | 0.1 / 0.2 | 32 / 32 |
| Member 2 | **64** | **6** | 4 / 8 | 128 / 256 | 8 / 16 | 1000 / 2000 | 0.1 / 0.2 | 32 / 32 |
| Member 3 | **128** | **4 / 6** | 4 / 8 | 128 / 256 | 8 / 16 | 1000 / 2000 | 0.1 / 0.2 | 64 / 64 |

Members 1 and 2 each cover `2^5 = 32` combinations of five varying parameters. Member 3 additionally varies layer count, covering `2^6 = 64` combinations. The three subsets do not overlap and together cover all 128 configurations.

Complete parameter grids:

- [Member 1 experiment grid](Member1_result/member1_experiment_grid.csv)
- [Member 2 experiment grid](Member2_result/member2_experiment_grid.csv)
- [Member 3 experiment grid](Member3_result/member3_experiment_grid.csv)

Each member has an independent `run_id` sequence. Members 1 and 2 enumerate the Cartesian product in the order `n_head → n_embd → batch_size → max_iters → dropout`, with the rightmost parameter changing fastest. Member 3 adds `n_layer` as the outermost parameter: `run_01–run_32` use 4 layers, and `run_33–run_64` use 6 layers. Identify experiments using both the member and run number, such as `Member3/run_15`.

## 3. Dataset and Shared Settings

The Shakespeare character dataset is prepared using nanoGPT's `data/shakespeare_char/prepare.py`. The first 90% of the text forms the training split, and the remaining 10% forms the validation split. The character vocabulary contains **65 tokens**. We report validation metrics; no separate test split is used.

| Setting | Value |
| --- | --- |
| Initialization | `init_from=scratch`; each configuration trains independently |
| Base training configuration | `config/train_shakespeare_char.py` |
| Device | Single CUDA GPU |
| Compilation | `compile=False` |
| Training seed | Internal seed `1337` in nanoGPT's `train.py` |
| Gradient accumulation | `gradient_accumulation_steps=1` |
| Optimizer | AdamW |
| Initial / minimum learning rate | `1e-3` / `1e-4` |
| Learning rate schedule | Warmup followed by cosine decay; `warmup_iters=100`, `lr_decay_iters=max_iters` |
| Evaluation frequency | Every 200 iterations, including step 0 |
| Evaluation batches | Average loss over 40 randomly sampled batches per split |
| Logging frequency | Every 100 iterations |
| Checkpoints | `always_save_checkpoint=True`; saved during evaluation |
| Text sampling | `seed=1337`, `num_samples=1`, `max_new_tokens=300`, `temperature=0.8`, `top_k=20` |

All three members' `provenance.json` files record the same source commit and input data checksum:

```text
nanoGPT commit: 3adf61e154c3fe3fca428ad6bc3818b27a3b8291
input.txt SHA256: 86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed
```

The recorded PyTorch versions are `2.11.0+cu130` for Member 1 and `2.14.1+cu130` for Members 2 and 3. The experiments therefore share the dataset, source revision, and main training settings, while their software environments differ.

## 4. Repository Structure and Artifacts

```text
.
├── README.md
├── Member1_nanoGPT_Colab.ipynb
├── Member2_nanoGPT_Colab.ipynb
├── Member3_nanoGPT_Colab.ipynb
├── Member1_result/                 # 32 experiments
├── Member2_result/                 # 32 experiments
└── Member3_result/                 # 64 experiments
```

Each `MemberN_result/` directory contains:

| File or directory | Contents |
| --- | --- |
| `memberN_experiment_grid.csv` | Complete parameter grid for the member |
| `provenance.json` | Source commit, dataset checksum, seed, and PyTorch version |
| `experiment_summary.csv` | Experiment statuses, configurations, and metrics |
| `ranked_experiments.csv` | Ranked experiment results |
| `hyperparameter_effects.csv` | Mean metrics grouped by individual parameter values |
| `comparison_*.png` | Comparisons of losses, loss gaps, training time, parameter counts, and GPU memory |
| `_preflight/` | Short compatibility check before formal experiments; excluded from experiment counts |
| `run_XX/config.json` | Configuration for an individual experiment |
| `run_XX/command.txt` | Actual training command |
| `run_XX/training.log` | Raw training and evaluation log |
| `run_XX/loss_history.csv` | Training and validation losses at each evaluation step |
| `run_XX/loss_curve.png` | Loss curves for an individual experiment |
| `run_XX/sample.txt` | Sampling log and generated text |

Model checkpoints are not included in this repository. The notebooks generate `checkpoint/ckpt.pt` at runtime and save it to the output directory, including Drive when enabled. The CSV `run_folder` fields and `/content/...` command paths refer to the original Colab environment. Use the corresponding `MemberN_result/run_XX/` directory to inspect archived results here.

### Metric Definitions

| Metric | Meaning |
| --- | --- |
| `final_train_loss` | Training cross-entropy loss at the final evaluation step |
| `final_val_loss` | Validation cross-entropy loss at the final evaluation step |
| `best_val_loss` | Lowest validation loss among recorded evaluation steps |
| `loss_gap` | `final_val_loss - final_train_loss` |
| `wall_seconds` | Training process duration, including evaluation and checkpoint saving, excluding subsequent text sampling |
| `parameter_count` | Total unique model parameters; tied weights are counted once |
| `gpu_used_mib_sampled_peak` | Maximum observed GPU-wide used memory in MiB, sampled approximately once per second using `nvidia-smi` |
| `training_tokens_approx` | `block_size × batch_size × (max_iters + 1)`, following the current training loop's counting convention |

Training losses in the curves are averages computed in evaluation mode, distinct from per-step training losses in the logs. `best_val_loss` is a recorded metric; because checkpoints are saved at every evaluation, the retained checkpoint may not correspond to the lowest validation loss.

## 5. Results

The three `experiment_summary.csv` files record **128 experiments with status `complete`**. The table below selects each member's configuration with the lowest `final_val_loss`:

| Member / Run | Block | Layers | Heads | Embedding | Batch | Iterations | Dropout | Final Train Loss | Final Val Loss | Parameters | Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [Member 1 / run_15](Member1_result/run_15/config.json) | 64 | 4 | 4 | 256 | 16 | 2000 | 0.1 | 1.5866 | 1.7373 | 3,181,056 | 35.98 |
| [Member 2 / run_31](Member2_result/run_31/config.json) | 64 | 6 | 8 | 256 | 16 | 2000 | 0.1 | 1.5695 | 1.7471 | 4,754,944 | 50.34 |
| [Member 3 / run_15](Member3_result/run_15/config.json) | 128 | 4 | 4 | 256 | 16 | 2000 | 0.1 | 1.4766 | **1.6450** | 3,197,440 | 28.47 |

For all three selected experiments, `best_val_loss` equals the final validation loss. **Member 3 / run_15** achieves the lowest final validation loss across the group, with a generalization gap of `0.1684`. See its [loss curve](Member3_result/run_15/loss_curve.png), [evaluation history](Member3_result/run_15/loss_history.csv), and [generated text](Member3_result/run_15/sample.txt).

Full summaries: [Member 1](Member1_result/experiment_summary.csv) · [Member 2](Member2_result/experiment_summary.csv) · [Member 3](Member3_result/experiment_summary.csv).

### Observed Hyperparameter Effects

The following values come from each member's `hyperparameter_effects.csv`. They represent **mean final validation loss**, grouped by parameter value within that member's experiments:

| Parameter Change | Member 1 | Member 2 | Member 3 |
| --- | --- | --- | --- |
| `n_head`: 4 → 8 | 2.0465 → 2.0435 | 2.0385 → 2.0312 | 2.0035 → 2.0108 |
| `n_embd`: 128 → 256 | 2.1563 → 1.9336 | 2.1359 → 1.9339 | 2.1254 → 1.8889 |
| `batch_size`: 8 → 16 | 2.1064 → 1.9835 | 2.0998 → 1.9700 | 2.0821 → 1.9321 |
| `max_iters`: 1000 → 2000 | 2.1320 → 1.9579 | 2.1205 → 1.9493 | 2.1198 → 1.8945 |
| `dropout`: 0.1 → 0.2 | 2.0083 → 2.0816 | 2.0008 → 2.0690 | 1.9637 → 2.0505 |

Within the tested ranges, larger embedding dimensions, larger batches, and more training iterations correspond to lower mean validation losses. A dropout rate of `0.1` yields lower mean validation loss than `0.2`. Increasing the attention head count produces small changes with inconsistent directions, so these results do not establish that more heads are always better. Higher dropout reduces the average training/validation loss gap while increasing validation loss, showing that the gap alone is insufficient for judging model quality.

The best configuration across the group uses 4 layers, so increasing depth did not guarantee the lowest validation loss in these experiments. These observations apply to the dataset, parameter ranges, and training budgets tested here.

### Comparison Plots

| Member | Final Losses | Loss Gap | Training Time | Parameter Count | GPU Memory |
| --- | --- | --- | --- | --- | --- |
| Member 1 | [View](Member1_result/comparison_final_losses.png) | [View](Member1_result/comparison_loss_gap.png) | [View](Member1_result/comparison_training_time.png) | [View](Member1_result/comparison_parameter_count.png) | [View](Member1_result/comparison_gpu_memory.png) |
| Member 2 | [View](Member2_result/comparison_final_losses.png) | [View](Member2_result/comparison_loss_gap.png) | [View](Member2_result/comparison_training_time.png) | [View](Member2_result/comparison_parameter_count.png) | [View](Member2_result/comparison_gpu_memory.png) |
| Member 3 | [View](Member3_result/comparison_final_losses.png) | [View](Member3_result/comparison_loss_gap.png) | [View](Member3_result/comparison_training_time.png) | [View](Member3_result/comparison_parameter_count.png) | [View](Member3_result/comparison_gpu_memory.png) |

## 6. Running and Reproducing the Experiments

Use a Google Colab runtime with a CUDA GPU, such as a T4. The notebooks use Python, PyTorch, NumPy, pandas, Matplotlib, and tiktoken. Setup cells install tiktoken, download nanoGPT, and prepare the dataset.

1. Open the [Member 1 notebook](Member1_nanoGPT_Colab.ipynb), [Member 2 notebook](Member2_nanoGPT_Colab.ipynb), or [Member 3 notebook](Member3_nanoGPT_Colab.ipynb) according to the assignment, and enable a GPU in Colab.
2. Run the environment setup, parameter settings, and helper cells in order. Keep `USE_DRIVE=True` and mount your own Google Drive to preserve results.
3. Check that the source commit and dataset SHA256 match this README. The current setup cells clone the latest upstream version without automatically pinning the commit. For strict reproduction, check out the recorded commit before preparing data and applying the compatibility patch.
4. Run the `_preflight` compatibility check before starting formal experiments. It uses only a few iterations and is excluded from the reported results.
5. Member 1 runs 32 experiments. Members 2 and 3 first run and validate the formal `run_01` pilot, then execute the remaining **31 and 63** configurations, respectively. Some Member 3 notebook messages still say “32” or “remaining 31”; use the 64-row parameter grid and actual experiment loop as the reference.
6. Run the summary, plotting, and export cells. Check experiment statuses, final evaluation steps, and output artifacts.

On reruns, the notebooks skip completed experiments whose checkpoints still exist, allowing remaining configurations to run after an interruption. Since this repository excludes checkpoints, the archived results alone do not satisfy that skip condition and may trigger retraining.

Training uses the internal seed in `train.py` without passing an additional `--seed` argument. Text sampling sets the seed through `sample.py --seed=1337`. The notebooks patch `sample.py` to use `torch.load(..., weights_only=False)` for compatibility when loading checkpoints generated by these experiments.

## 7. Interpretation and Limitations

- **Different training volumes:** `block_size`, `batch_size`, and `max_iters` all change the number of training tokens. Estimated training volumes range from 512,512 to 4,098,048 tokens, so comparisons do not use equal token or compute budgets. For example, Member 3 / run_15 processes twice as many tokens as Member 1 / run_15; their validation loss difference cannot be attributed entirely to context length.
- **Schedules change with iteration limits:** Because `lr_decay_iters=max_iters`, comparing 1000 and 2000 iterations changes both training duration and the learning rate decay schedule.
- **Environment differences:** PyTorch versions differ. GPU models and runtime conditions also need to be checked before comparing performance across members; recorded durations alone cannot establish that one architecture is faster.
- **Sampled memory measurements:** GPU memory values cover the entire device, may include other processes, and may miss brief peaks between samples.
- **Single seed and validation-based selection:** Each configuration uses one training seed, without repeated-run variance estimates or significance testing. The best model is selected using validation loss, which is not an independent test score.
- **Scope of grouped means:** `hyperparameter_effects.csv` summarizes the five shared varying parameters. Member 3's means also combine both layer counts. To analyze context length and depth, combine the original summary tables and compare configurations with matching remaining parameters while accounting for environment differences.

