"""Real BERT GPU job for live scheduler evaluation (inference or fine-tune).

A latency-SLO "inference" job runs BERT forward passes; a best-effort "training"
job fine-tunes BERT (forward + backward + AdamW). Both use bert-base-uncased
loaded OFFLINE from the /shared HF cache, with random token batches (no dataset
needed for timing). The job's GPU footprint is governed by Slurm --gres=mps:N.

Parameterised by ``--n`` (inference batches / training steps) so the harness can
target a runtime; ``--seed`` keeps CRN across arms.

Run via the pytorch Lmod module:
  module use /shared/modulefiles && module load cuda/12.4 pytorch
  python3 /shared/scripts/bert_job.py --mode infer --n 60 --batch-size 16
"""
import argparse
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/shared/hf_cache")

import torch  # noqa: E402
from transformers import BertModel  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["infer", "train"], required=True)
    ap.add_argument("--n", type=int, default=50, help="inference batches / training steps")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="bert-base-uncased")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model = BertModel.from_pretrained(a.model).to(dev)
    vocab = model.config.vocab_size

    def batch():
        return torch.randint(0, vocab, (a.batch_size, a.seq_len), device=dev)

    load_s = time.time() - t0
    c0 = time.time()
    if a.mode == "infer":
        model.eval()
        with torch.no_grad():
            for _ in range(a.n):
                model(batch())
            if dev == "cuda":
                torch.cuda.synchronize()
    else:  # train: fine-tune (forward + backward + optimizer)
        model.train()
        head = torch.nn.Linear(model.config.hidden_size, 2).to(dev)
        opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=1e-5)
        lossfn = torch.nn.CrossEntropyLoss()
        for _ in range(a.n):
            opt.zero_grad(set_to_none=True)
            pooled = model(batch()).pooler_output
            loss = lossfn(head(pooled), torch.randint(0, 2, (a.batch_size,), device=dev))
            loss.backward()
            opt.step()
        if dev == "cuda":
            torch.cuda.synchronize()
    compute_s = time.time() - c0
    print(f"done mode={a.mode} n={a.n} bs={a.batch_size} dev={dev} "
          f"load={load_s:.1f}s compute={compute_s:.1f}s total={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
