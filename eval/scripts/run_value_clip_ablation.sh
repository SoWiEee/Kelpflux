#!/usr/bin/env bash
# Duan-style target return-clip stabilizer ablation.
#
# Hypothesis: the distributional critic's collapse (§4.3 table 6: RDSAC-mean
# collapsed to 0% completion in 4/6 (trace×seed) cells; RDSAC-cvar at 1×1 also
# 0%) is driven by Z_R overestimation. Duan et al. 2021's target return-clip
# (bound the bootstrap target within ±b of the current value) should curb it.
#
# Design (matches §4.3's collapse-prone regime): 1×1, σ=1.0, fixed-α=0.05,
# train seeds 42/43/44, families philly+ali. Two conditions:
#   clip-off  : arms {sac, rdsac-mean, rdsac-cvar}   (sac = same-budget anchor)
#   clip-on   : arms {rdsac-mean, rdsac-cvar} + --value-clip B   (sac unaffected)
# Metric: completion rate (collapse count) + ΔJCT% vs score, per (arm,family).
#
#   STEPS=45000 CLIP=10 DEVICE=cuda bash eval/scripts/run_value_clip_ablation.sh
# Self-backgrounds via nohup. Kill: pkill -f run_value_clip_ablation; pkill -f sweep_stochastic
set -u
cd "$(dirname "$0")/../.." || exit 1

STEPS="${STEPS:-45000}"
CLIP="${CLIP:-10}"
DEVICE="${DEVICE:-cuda}"
PY="PYTHONPATH=. .venv-m11/bin/python"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="runs/value_clip_ablation_${STAMP}.log"
mkdir -p runs

main() {
  echo "### value-clip ablation started $(date) ; steps=${STEPS} clip=${CLIP} device=${DEVICE}" | tee -a "$LOG"
  for SEED in 42 43 44; do
    # --- clip OFF (baseline; --no-sac: SAC uses scalar critic, unaffected by
    #     clip, and §4.3's 100k SAC serves as the reference anchor) ---
    local off="runs/vclip_off_s${SEED}"
    echo "=== [$(date +%H:%M:%S)] clip-off → ${off} (train-seed ${SEED}) ===" | tee -a "$LOG"
    eval $PY eval/scripts/sweep_stochastic.py \
      --sigmas 1.0 --total-steps "$STEPS" --warmup-steps 2000 \
      --n-jobs 50 --seeds 42 43 44 45 46 --trace-families philly ali \
      --risk-modes mean cvar --no-sac --fixed-alpha --init-alpha 0.05 \
      --n-nodes 1 --gpus-per-node 1 --train-seed "$SEED" \
      --value-clip 0 --device "$DEVICE" --out-dir "$off" >>"$LOG" 2>&1
    echo "=== [$(date +%H:%M:%S)] done ${off} (exit $?) ===" | tee -a "$LOG"

    # --- clip ON (mean+cvar only; sac is scalar-critic, unaffected by clip) ---
    local on="runs/vclip_on_s${SEED}"
    echo "=== [$(date +%H:%M:%S)] clip-on(b=${CLIP}) → ${on} (train-seed ${SEED}) ===" | tee -a "$LOG"
    eval $PY eval/scripts/sweep_stochastic.py \
      --sigmas 1.0 --total-steps "$STEPS" --warmup-steps 2000 \
      --n-jobs 50 --seeds 42 43 44 45 46 --trace-families philly ali \
      --risk-modes mean cvar --no-sac --fixed-alpha --init-alpha 0.05 \
      --n-nodes 1 --gpus-per-node 1 --train-seed "$SEED" \
      --value-clip "$CLIP" --device "$DEVICE" --out-dir "$on" >>"$LOG" 2>&1
    echo "=== [$(date +%H:%M:%S)] done ${on} (exit $?) ===" | tee -a "$LOG"
  done

  echo "=== [$(date +%H:%M:%S)] aggregating ===" | tee -a "$LOG"
  eval $PY - "$STAMP" "$CLIP" >>"$LOG" 2>&1 <<'PYEOF'
import json, sys
from pathlib import Path
import numpy as np
stamp, clip = sys.argv[1], sys.argv[2]
SEEDS = [42, 43, 44]; FAMS = ["philly", "ali"]

def collect(prefix, models):
    rows = {}
    for s in SEEDS:
        f = Path(f"runs/{prefix}_s{s}/sweep.json")
        if not f.exists():
            continue
        for r in json.loads(f.read_text()):
            if r["model"] in models:
                rows.setdefault((r["family"], r["model"]), []).append(
                    (r.get("delta_pct", float("nan")),
                     r.get("completed_frac", float("nan"))))
    return rows

off = collect("vclip_off", {"rdsac-mean", "rdsac-cvar"})
on  = collect("vclip_on",  {"rdsac-mean", "rdsac-cvar"})

def fmt(vals):
    if not vals:
        return "—", "—", 0
    d = np.array([x[0] for x in vals], float); d = d[np.isfinite(d)]
    c = np.array([x[1] for x in vals], float)
    sd = np.std(d, ddof=1) if d.size > 1 else 0.0
    dstr = f"{np.mean(d):+.1f}±{sd:.1f}" if d.size else "n/a"
    # collapse = completion < 20%
    ncol = int((c < 0.20).sum())
    return dstr, f"{np.mean(c):.0%} ({ncol}/{c.size} collapsed)", d.size

out = [f"# Value-clip (Duan) ablation — {stamp}  (b={clip}, 1×1, σ=1.0, fixed-α=0.05)", "",
       "ΔJCT% vs score (mean±std across 3 train seeds; + = beats score). "
       "Completion: mean over (trace×seed); collapse = <20% done.", "",
       "| condition | arm | family | ΔJCT% | completion | n |",
       "|---|---|---|---:|---|---:|"]
for fam in FAMS:
    for m in ["rdsac-mean", "rdsac-cvar"]:
        d, c, n = fmt(off.get((fam, m)))
        out.append(f"| clip-off | {m} | {fam} | {d} | {c} | {n} |")
for fam in FAMS:
    for m in ["rdsac-mean", "rdsac-cvar"]:
        d, c, n = fmt(on.get((fam, m)))
        out.append(f"| clip-on b={clip} | {m} | {fam} | {d} | {c} | {n} |")
out += ["",
        "**Read:** clip helps iff (a) clip-on collapse count < clip-off for the "
        "distributional arms (mean/cvar), and/or (b) clip-on ΔJCT% ≥ clip-off. "
        "If neither, the return-clip does not rescue the distributional critic at "
        "this scale and RDSAC stays statistically ≤ score/SAC."]
dst = Path(f"runs/value_clip_ablation_{stamp}_TABLES.md")
dst.write_text("\n".join(out))
print(f"[agg] wrote {dst}"); print("\n".join(out))
PYEOF
  echo "### value-clip ablation finished $(date)" | tee -a "$LOG"
  echo "VCLIP_ABLATION_DONE ${STAMP}" >> "$LOG"
}

nohup bash -c "$(declare -f main); STEPS='${STEPS}' CLIP='${CLIP}' DEVICE='${DEVICE}' PY='${PY}' STAMP='${STAMP}' LOG='${LOG}' main" >/dev/null 2>&1 &
echo "value-clip ablation launched (PID $!). Log: ${LOG}"
