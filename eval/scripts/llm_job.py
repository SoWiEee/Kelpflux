"""Real small-LLM GPU job for live scheduler evaluation (serving / fine-tune).

A latency-SLO "inference" job runs *batched autoregressive generation* (the
modern "AI serving" shape); a best-effort "training" job fine-tunes the model
(forward + backward + AdamW). Both use Qwen2.5-0.5B-Instruct loaded OFFLINE from
/shared/models/qwen05b with random token batches (no dataset needed for timing).

Why batched generation (not single-sequence): a batch of B sequences each
generating G tokens makes the per-step matmuls large enough to be *compute-
bound*, so the job's throughput scales with the MPS thread budget
(CUDA_MPS_ACTIVE_THREAD_PERCENTAGE, set by Slurm --gres=mps:N) AND with card
speed (4070 vs 3080). A single-sequence generate would be launch/memory-bound
and blind to both levers — the whole point of this eval.

Parameterised by ``--n`` (generation rounds / training steps) so the harness can
target a runtime; ``--seed`` keeps CRN across arms. Greedy decoding with
min=max new tokens → deterministic, fixed WORK per round (calibratable).

Run via the relocatable /shared/py python (torch cu124):
  /shared/py/bin/python3 /shared/scripts/llm_job.py --mode infer --n 20
"""
import argparse
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["infer", "train"], required=True)
    ap.add_argument("--n", type=int, default=20,
                    help="generation rounds (infer) / fine-tune steps (train)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--gen-len", type=int, default=64, help="new tokens per round (infer)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="/shared/models/qwen05b")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(dev)
    vocab = model.config.vocab_size
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def batch():
        # random token ids → deterministic given seed; content irrelevant for timing
        return torch.randint(0, vocab, (a.batch_size, a.prompt_len), device=dev)

    load_s = time.time() - t0
    c0 = time.time()
    if a.mode == "infer":
        model.eval()
        with torch.no_grad():
            for _ in range(a.n):
                ids = batch()
                model.generate(
                    ids,
                    max_new_tokens=a.gen_len,
                    min_new_tokens=a.gen_len,   # force full length → fixed work
                    do_sample=False,            # greedy → deterministic
                    num_beams=1,
                    use_cache=True,
                    pad_token_id=pad_id,
                )
            if dev == "cuda":
                torch.cuda.synchronize()
    else:  # train: fine-tune (forward + backward + optimizer)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
        for _ in range(a.n):
            opt.zero_grad(set_to_none=True)
            ids = batch()
            out = model(ids, labels=ids)  # causal LM loss on random tokens (timing only)
            out.loss.backward()
            opt.step()
        if dev == "cuda":
            torch.cuda.synchronize()
    compute_s = time.time() - c0
    print(f"done mode={a.mode} n={a.n} bs={a.batch_size} gen={a.gen_len} dev={dev} "
          f"load={load_s:.1f}s compute={compute_s:.1f}s total={time.time()-t0:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
