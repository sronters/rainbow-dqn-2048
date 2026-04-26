# Rainbow DQN for 2048

AI agent for 2048 using a full **Rainbow DQN** stack: Dueling Networks, C51, N-step returns, Prioritized Experience Replay, and NoisyNets. Everything fits in two Python files — no browser automation, no external clients.

---
<img width="1246" height="546" alt="image" src="https://github.com/user-attachments/assets/a8651294-4644-431a-abc7-2e4c62d6d789" />


## Project Structure

```
2048/
├── colab_rainbow.py   # Training pipeline (environment + agent + loop)
├── play.py            # Inference — watch the bot play in your terminal
└── rainbow_best.pth   # Pre-trained weights (~17 MB)
```

---

## Quick Start

Install dependencies:

```bash
pip install torch numpy
```

Run with the pre-trained model:

```bash
python play.py
```

Optional flags:

```bash
python play.py --delay 0.05          # faster playback
python play.py --delay 0             # maximum speed
python play.py --model my_model.pth  # custom checkpoint
python play.py --manual              # enter a custom starting board
```

Manual board entry prompts you for 4 rows (use `0` for empty cells):

```
 Row 1: 0 2 0 2
 Row 2: 0 0 0 0
 Row 3: 0 0 0 0
 Row 4: 0 0 0 0
```

---

## Architecture

### Rainbow Components

| Component | Implementation |
|---|---|
| **C51 (Distributional RL)** | Output is a probability distribution over 51 atoms in `[v_min, v_max]`. Q-values are computed as the expected value `Σ p_i · z_i`. Loss is KL-divergence against a Bellman-projected target distribution. |
| **Dueling Networks** | The flattened CNN features split into a scalar value stream `V(s)` and an advantage stream `A(s,a)`. Combined as `Q = V + A − mean(A)` to decorrelate action scoring from state value. |
| **NoisyNets** | All linear layers replaced by `NoisyLinear`, which adds factorised Gaussian noise scaled by learned `σ` parameters. Replaces ε-greedy — exploration is state-dependent and trained end-to-end. |
| **N-step Returns** | A rolling deque accumulates 5 transitions before pushing to the replay buffer. The stored return is `R = Σ γ^i · r_i`, reducing variance compared to 1-step TD. |
| **Prioritized Experience Replay** | Transitions are stored in a `SumTree` and sampled proportional to `|TD-error|^α`. Importance-sampling weights (`IS-weights`) correct the resulting bias. `β` anneals from 0.4 → 1.0 over training. |
| **Double DQN** | Action selection uses the online network; value estimation uses the target network, decoupled to reduce overestimation. |

### Network Topology

```
Input: (B, 8, 4, 4)
  Conv2d(8→128, 2×2) → ReLU
  Conv2d(128→256, 2×2) → ReLU
  Conv2d(256→256, 1×1) → ReLU
  ResBlock(256) × 2 → Flatten → (B, 1024)
  ├─ Value:     NoisyLinear(1024→256) → ReLU → NoisyLinear(256→51)    → (B, 1, 51)
  └─ Advantage: NoisyLinear(1024→256) → ReLU → NoisyLinear(256→4×51) → (B, 4, 51)
Output: log_softmax(V + A − mean(A))  shape (B, 4, 51)
```

Residual blocks use two `Conv2d(256, 256, 3×3, padding=1)` layers with BatchNorm and a skip connection.

### State Encoding (8 channels)

The raw 4×4 integer board is converted to an `(8, 4, 4)` float tensor before entering the network:

| Ch | Content | Rationale |
|---|---|---|
| 0 | `log2(tile) / 17` | Compresses the exponential tile scale into a linear range |
| 1 | Empty cell mask | Signals available space |
| 2 | Max tile mask | Anchors the agent to the highest-value tile |
| 3 | Tiles normalized by max | Relative tile magnitudes |
| 4 | Row monotonicity (`±1`) | Encodes left→right ordering |
| 5 | Column monotonicity (`±1`) | Encodes top→bottom ordering |
| 6 | Merge opportunity mask | Highlights adjacent equal tiles |
| 7 | Corner distance gradient | Biases toward top-left corner strategy |

### Reward Shaping

Raw score delta is augmented with board-quality heuristics and squashed through `tanh` to keep rewards in `(-1, +1)`:

```
r = Δscore
  + 0.2 × empty_cells
  + 1.5 × monotonicity
  + 1.0 × smoothness          # penalises large log-differences between neighbours
  + 3.0 × (max tile in top-left corner)
  + 0.5 × merge_potential     # count of adjacent equal tile pairs

reward = tanh(r / 1000)
```

---

## Training (Google Colab)

1. Set runtime to **T4 GPU** (`Runtime → Change runtime type`).
2. Upload `colab_rainbow.py`.
3. Run:
   ```python
   !python colab_rainbow.py
   ```

Training uses **64 parallel environments** (`VectorEnv`) stepped synchronously each iteration, which maximises transition diversity and fills the replay buffer quickly. Progress is logged every 100 episodes, showing average score, loss, and the percentage of games reaching 1024/2048 tiles.

Weights are saved to `rainbow_best.pth` whenever a new best average score is reached. Periodic checkpoints (`rainbow_{N}.pth`) are saved every 50 000 steps. A three-panel training curve (score, max tile, loss) is saved to `rainbow_curves.png` at the end.

### Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `total_steps` | 500 000 | Total env steps |
| `num_envs` | 64 | Parallel environments |
| `batch_size` | 512 | Training batch size |
| `lr` | 1e-4 | Adam learning rate |
| `n_step` | 5 | N-step return length |
| `n_atoms` | 51 | C51 distribution atoms |
| `v_min / v_max` | -200 / 5000 | Return support range |
| `warmup` | 20 000 | Steps before training starts |

---

## Requirements

```
python >= 3.8
torch >= 1.12
numpy
matplotlib  # training plots only
```

---

## License

MIT
