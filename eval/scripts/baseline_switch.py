"""Switch the live Slurm scheduling policy to a named baseline arm.

The baseline live A/B compares the production `score` heuristic against
Slurm-native schedulers (FCFS / multifactor / packing). Each arm is a
different `slurm.conf` + `job_submit.lua` state. Because both files are
**subPath** ConfigMap mounts (which kubelet does NOT live-update), every
arm switch requires restarting `slurm-controller-0` so it re-materialises
the files from the (edited) ConfigMaps.

This module owns the cluster surgery: patch the ConfigMap(s) in place, roll
the controller, and wait until slurmctld answers again. `--dry-run` prints
the intended ConfigMap edits without touching the cluster.

Arm matrix (all arms set RL off via the Lua so Slurm-native ordering governs;
`score` keeps SCORE_APPLY=true, the others turn it off):

    arm          SCORE_APPLY  SchedulerType    PriorityType         SelectTypeParameters
    score        true         sched/backfill   priority/multifactor (unchanged)
    multifactor  false        sched/backfill   priority/multifactor (unchanged)
    fcfs         false        sched/builtin    priority/basic       (unchanged)
    packing      false        sched/backfill   priority/multifactor CR_Pack_Nodes (added)
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time

NS = "slurm"
CTRL = "slurm-controller-0"
CM_LUA = "slurm-config-job-submit"        # data key: job_submit.lua
CM_CONF = "slurm-config-static"           # data key: slurm.conf

# arm → (score_apply, scheduler_type, priority_type, extra_select_params)
ARMS: dict[str, dict] = {
    "score":       dict(score_apply=True,  sched="sched/backfill", prio="priority/multifactor", pack=False),
    "multifactor": dict(score_apply=False, sched="sched/backfill", prio="priority/multifactor", pack=False),
    "fcfs":        dict(score_apply=False, sched="sched/builtin",  prio="priority/basic",        pack=False),
    "packing":     dict(score_apply=False, sched="sched/backfill", prio="priority/multifactor", pack=True),
}


def _kubectl(*args: str, check: bool = True, input_: str | None = None) -> str:
    proc = subprocess.run(["kubectl", *args], text=True, capture_output=True,
                          input=input_, check=False)
    if check and proc.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _get_cm_data(cm: str, key: str) -> str:
    return _kubectl("get", "cm", "-n", NS, cm, "-o", f"jsonpath={{.data.{key.replace('.', chr(92)+'.')}}}")


def _patch_lua(lua: str, *, score_apply: bool, rl_enabled: bool) -> str:
    lua = re.sub(r"(local SCORE_APPLY\s*=\s*)(true|false)",
                 lambda m: m.group(1) + ("true" if score_apply else "false"), lua, count=1)
    lua = re.sub(r"(RL_ENABLED\s*=\s*)(true|false)",
                 lambda m: m.group(1) + ("true" if rl_enabled else "false"), lua, count=1)
    return lua


def _patch_conf(conf: str, *, sched: str, prio: str, pack: bool) -> str:
    conf = re.sub(r"(?m)^(SchedulerType\s*=\s*).*$", r"\g<1>" + sched, conf, count=1)
    conf = re.sub(r"(?m)^(PriorityType\s*=\s*).*$", r"\g<1>" + prio, conf, count=1)
    # SelectTypeParameters: ensure CR_Pack_Nodes is present (packing) or absent.
    def _fix_select(m: re.Match) -> str:
        params = [p for p in m.group(2).split(",") if p and p != "CR_Pack_Nodes"]
        if pack:
            params.append("CR_Pack_Nodes")
        return m.group(1) + ",".join(params)
    conf = re.sub(r"(?m)^(SelectTypeParameters\s*=\s*)(.*)$", _fix_select, conf, count=1)
    return conf


def _apply_cm(cm: str, key: str, new_value: str) -> None:
    # Replace just the one data key via a strategic JSON merge patch.
    import json
    patch = json.dumps({"data": {key: new_value}})
    _kubectl("patch", "cm", "-n", NS, cm, "--type", "merge", "-p", patch)


def _restart_controller(timeout_s: float = 300.0) -> None:
    print("[switch] restarting slurm-controller-0 …", flush=True)
    # Capture the current pod UID so we can tell the NEW pod from the OLD one that
    # is still gracefully Terminating (it stays Running+Ready for a while, which a
    # naive phase/ready check races against and returns too early).
    old_uid = _kubectl("get", "pod", "-n", NS, CTRL, "-o",
                       "jsonpath={.metadata.uid}", check=False).strip()
    _kubectl("delete", "pod", "-n", NS, CTRL, "--wait=false")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        uid = _kubectl("get", "pod", "-n", NS, CTRL, "-o",
                       "jsonpath={.metadata.uid}", check=False).strip()
        phase = _kubectl("get", "pod", "-n", NS, CTRL, "-o",
                         "jsonpath={.status.phase}", check=False).strip()
        ready = _kubectl("get", "pod", "-n", NS, CTRL, "-o",
                         "jsonpath={.status.containerStatuses[0].ready}", check=False).strip()
        if uid and uid != old_uid and phase == "Running" and ready == "true":
            break
        time.sleep(5)
    else:
        raise TimeoutError("controller did not become Ready in time")
    # wait for slurmctld to actually answer
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        out = _kubectl("exec", "-n", NS, CTRL, "--", "sinfo", "-h", check=False)
        if out.strip():
            print("[switch] slurmctld responsive.", flush=True)
            return
        time.sleep(5)
    raise TimeoutError("slurmctld did not answer sinfo in time")


def switch(arm: str, *, dry_run: bool = False, rl_enabled: bool = False) -> None:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {list(ARMS)}")
    cfg = ARMS[arm]
    lua = _get_cm_data(CM_LUA, "job_submit.lua")
    conf = _get_cm_data(CM_CONF, "slurm.conf")
    new_lua = _patch_lua(lua, score_apply=cfg["score_apply"], rl_enabled=rl_enabled)
    new_conf = _patch_conf(conf, sched=cfg["sched"], prio=cfg["prio"], pack=cfg["pack"])

    def _show(name, old, new):
        olds = {ln.strip() for ln in old.splitlines()}
        for ln in new.splitlines():
            if ln.strip() and ln.strip() not in olds:
                print(f"  [{name}] -> {ln.strip()}")

    print(f"[switch] arm={arm}  score_apply={cfg['score_apply']}  sched={cfg['sched']}  "
          f"prio={cfg['prio']}  pack={cfg['pack']}  rl_enabled={rl_enabled}")
    _show("lua", lua, new_lua)
    _show("conf", conf, new_conf)
    if dry_run:
        print("[switch] dry-run: no ConfigMap edits / no restart.")
        return
    lua_changed = new_lua != lua
    conf_changed = new_conf != conf
    if lua_changed:
        _apply_cm(CM_LUA, "job_submit.lua", new_lua)
    if conf_changed:
        _apply_cm(CM_CONF, "slurm.conf", new_conf)
    if lua_changed or conf_changed:
        _restart_controller()
    else:
        print("[switch] no change needed (already in this arm's state).")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Switch live Slurm scheduling to a baseline arm")
    p.add_argument("arm", choices=list(ARMS))
    p.add_argument("--rl-enabled", action="store_true",
                   help="leave RL_ENABLED=true (default false: pure Slurm-native baseline)")
    p.add_argument("--dry-run", action="store_true", help="print intended edits, no cluster change")
    args = p.parse_args(argv)
    switch(args.arm, dry_run=args.dry_run, rl_enabled=args.rl_enabled)
    return 0


if __name__ == "__main__":
    sys.exit(main())
