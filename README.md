# trader_v12 — multi-agent RL trader for BTC / ETH / LTC (1-minute data)

The intense edition of the trader family, built for a **4× A10G (23 GB) /
120 GB RAM / 48-core** box. Three cooperating agents trade full days of
1-minute BTC/ETH/LTC bars, each episode starting with **$20** aiming for
**$30+** — with **two competing trade makers** and every epoch logged as PNGs,
including **one dedicated chart per trader per epoch**.

## What's new in v12

1. **Uses the whole machine**
   - `--days-per-epoch` (auto → **4** on 4-GPU/big-RAM boxes): each epoch
     draws 4 random days; both traders roll out **all of them in one
     vectorised (batched) pass** and PPO-update on 4× samples → faster
     wall-clock learning, more precise gradients.
   - All ~1,340 days are feature-precomputed at startup by a 24-worker
     multiprocessing pool (uses the 120 GB RAM, kills per-epoch latency).
   - TF32 matmul + `cudnn.benchmark` on Ampere+ GPUs.
   - Held-out evaluation is batched and runs both traders in parallel
     threads (~10× faster eval rounds).
2. **Bigger, more precise agents**
   - New **XL tier** for A10G-class GPUs: predictor/analyzer **GRU 640×3**
     (~6.2M params each, LayerNorm), traders **MLP 8192-8192-8192 (~147M params
     each)** — sized to the VRAM budget (`--vram-budget-gb`, 20 GB cap).
   - **EMA (weight-averaged) predictor/analyzer** feed the traders and the
     evaluations — steadier, more precise forecasts than raw weights.
   - Predictor/analyzer take more gradient steps per epoch (tier-scaled:
     16 steps on 24-day batches at XL).
3. **One tall PNG per epoch for BOTH traders** (all other graphs unchanged):
   `epoch_XXXXX_<date>_traders.png` draws the day's market ONCE (3 normalised
   price panels, each carrying a thin long/short exposure ribbon per trader —
   A's ribbon above B's), followed by each agent's equity curve with
   best/worst stats and per-coin portfolio-weight decisions, trader B's
   directly below trader A's.

Carried over from v10: vol-normalized 4-horizon targets + correlation loss
(anti-flat-line predictor), warmup+cosine LR for advisors, KL-adaptive LR for
traders, two competing traders A/B.

## Model sizes (XL tier, A10G)

| Model | params | fp32 size |
|---|---|---|
| Price predictor (GRU 640×3 + LN) | ~6.2M | 25 MB |
| Market analyzer (GRU 640×3 + LN) | ~6.2M | 25 MB |
| Trade maker A (MLP 8192×3) | ~147M | 587 MB |
| Trade maker B (MLP 8192×3) | ~147M | 587 MB |
| **Total** | **~306M** | **~1.2 GB** (×3 with Adam states) |

Peak per-GPU VRAM (weights + Adam + activations) is printed every epoch and
stays well inside the 20 GB budget.

## Quick start (Snowflake 4×A10G / any machine)

```bash
pip install -r requirements.txt
python train.py --epochs 2000        # everything auto-detected
```

or `bash scripts/snowflake_run.sh`.

## Outputs (all PNG, every epoch)

```
runs/
  epochs/epoch_XXXXX_YYYY-MM-DD.png            <- combined day snapshot (both traders)
  epochs/epoch_XXXXX_YYYY-MM-DD_traders.png    <- ONE tall PNG with BOTH agents
                                                  (v12.2): market drawn ONCE (3 price
                                                  panels, each with an exposure ribbon
                                                  per trader), then A's equity+weights,
                                                  then B's directly below
  training_curves.png                          <- 12 panels incl. predictor correlation
  metrics.csv                                  <- all numbers (hit_* now = fraction of
                                                  the epoch's days that reached target)
  ckpt/{predictor,analyzer,trader_a,trader_b}.pt (+ best_a, best_b)
  resources.txt, config.json
```

```bash
python eval.py --ckpt runs/ckpt --which both --days 40
```

## Key flags

```
--epochs 2000        --days-per-epoch 0(auto)  --vram-budget-gb 20
--device auto        --min-free-gb 3.0         --no-precompute
--trader-hidden auto --pred-hidden auto        --pred-layers 0
--ema-decay 0.999    --single-trader           --target-kl 0.03
--capital 20         --target 30               --max-lev 5.0
--resume runs/ckpt   --plot-every 1            --eval-batch 8
```

## Honest notes

* $20 → $30 in one day is a +50 % daily return — extreme; expect a low
  hit-rate and trust the held-out eval columns (`eval_hit_a/b`), not
  training-day equity.
* Fees (0.05 %) + slippage (0.02 %) per rebalance are simulated; churning
  dies to fees, which the agents must learn to avoid.
* Research toy, not financial advice.
