"""Real ResNet-50 GPU job for live scheduler evaluation (training or inference).

The "medium / compute-bound" tier of the three-class AI workload (BERT inference
= small+latency, ResNet training = medium+compute, Qwen fine-tune = large+RAM).

The model is built **in-code** from ``transformers.ResNetConfig`` (the
microsoft/resnet-50 architecture) rather than loaded from a checkpoint. Two
reasons, both deliberate:

  * torchvision is not installed in /shared/py, and
  * an in-code model performs **no NFS read at all**, which removes the
    cold-page-cache confound. A cold 954MB checkpoint read from the slow node
    measured ~84s and silently blew past job time limits; for *timing* purposes
    random weights are numerically irrelevant (identical FLOPs per step), so
    building in-code buys determinism for free.

Synthetic image batches (no dataset needed for timing). ``--n`` (steps/batches)
lets the harness target a runtime; ``--seed`` keeps CRN across arms.

Run:
  /shared/py/bin/python3 /shared/scripts/resnet_job.py --mode train --n 20 --batch-size 32
"""
import argparse
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/shared/hf_cache")

import torch  # noqa: E402
from transformers import ResNetConfig, ResNetModel  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["infer", "train"], required=True)
    ap.add_argument("--n", type=int, default=20, help="inference batches / training steps")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    # ResNetConfig() defaults to the resnet-50 stage layout ([3,4,6,3] bottleneck).
    model = ResNetModel(ResNetConfig()).to(dev)
    hidden = model.config.hidden_sizes[-1]

    def batch():
        return torch.randn(a.batch_size, 3, a.image_size, a.image_size, device=dev)

    load_s = time.time() - t0
    c0 = time.time()
    if a.mode == "infer":
        model.eval()
        with torch.no_grad():
            for _ in range(a.n):
                model(batch())
            if dev == "cuda":
                torch.cuda.synchronize()
    else:  # train: forward + backward + optimizer (SGD+momentum, the ResNet default)
        model.train()
        head = torch.nn.Linear(hidden, 1000).to(dev)
        opt = torch.optim.SGD(list(model.parameters()) + list(head.parameters()),
                              lr=0.01, momentum=0.9)
        lossfn = torch.nn.CrossEntropyLoss()
        for _ in range(a.n):
            opt.zero_grad(set_to_none=True)
            pooled = model(batch()).pooler_output.flatten(1)
            loss = lossfn(head(pooled), torch.randint(0, 1000, (a.batch_size,), device=dev))
            loss.backward()
            opt.step()
        if dev == "cuda":
            torch.cuda.synchronize()
    compute_s = time.time() - c0
    peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if dev == "cuda" else 0.0
    print(f"done mode={a.mode} n={a.n} bs={a.batch_size} dev={dev} "
          f"load={load_s:.1f}s compute={compute_s:.1f}s total={time.time()-t0:.1f}s "
          f"peak_vram_mb={peak_mb:.0f}", flush=True)


if __name__ == "__main__":
    main()
