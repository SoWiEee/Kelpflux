# Scheduling Research

Scheduling and simulator code lives primarily in `sim/`, `services/rl_scheduler/`, and `eval/`.

- Read [scheduler.md](../scheduler.md) for the current system and algorithm description, and [eval-writeup.md](../eval-writeup.md) for experimental methodology and results.
- Treat `sim/gym_env.py`, the selected checkpoint metadata, and the live scheduler health response as the sources of truth for observation/action dimensions. A checkpoint trained for a different topology is not interchangeable.
- Do not copy historical experiment settings or performance claims from old agent notes. Verify claims against the current evaluation artifacts before changing the paper or README.
- Keep training and evaluation commands close to their scripts or in research documentation; do not add long experiment recipes to the root guide.
