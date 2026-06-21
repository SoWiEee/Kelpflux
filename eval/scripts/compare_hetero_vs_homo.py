import sys, glob, numpy as np
sys.path.insert(0, ".")
from eval.scripts.sweep_stochastic import _eval_policy, _score_action
from sim.scheduler.score import ScoreScheduler
from services.rl_scheduler.dsac import DSACAgent

NODE_SPEEDS=[1.0,0.25]; SEEDS=[42,43,44]; ARMS=["rdsac-cvar","rdsac-mean"]; FAMS=["philly","ali"]
sc=ScoreScheduler()
def mk(ag): return lambda env: (lambda: ag.select_action(env._build_obs(), env.action_mask(), greedy=True))
def eval_pol(make, fam):
    avg,_=_eval_policy(make_policy=make, family=fam, sigma=1.0, interference=0.0,
        n_jobs=50, seeds=[42,43,44,45,46], n_nodes=2, gpus_per_node=1, node_speeds=NODE_SPEEDS)
    return np.array(avg)
print(f"{'fam':6} {'arm':11} {'seed':>4} {'homo Δ%':>9} {'hetero Δ%':>10}  winner")
for fam in FAMS:
    sa = eval_pol(lambda env:(lambda:_score_action(env,sc)), fam)
    for arm in ARMS:
        for s in SEEDS:
            try:
                h=DSACAgent.load(f"runs/p1p2_2x1_s{s}/ckpt_{arm}_sigma1.0.pt")
                t=DSACAgent.load(f"runs/hetero_s{s}/ckpt_{arm}_sigma1.0.pt")
            except Exception as e:
                print(f"{fam:6} {arm:11} {s:>4}  load-fail {e}"); continue
            ha=eval_pol(mk(h),fam); ta=eval_pol(mk(t),fam)
            b=np.isfinite(sa)&np.isfinite(ha)&np.isfinite(ta)
            dh=float(((sa[b]-ha[b])/sa[b]).mean()*100); dt=float(((sa[b]-ta[b])/sa[b]).mean()*100)
            w="HETERO" if dt>dh else "homo"
            print(f"{fam:6} {arm:11} {s:>4} {dh:>9.1f} {dt:>10.1f}  {w}")
