# Go2 Posture Guidance Policies

This folder contains the selected Go2 posture guidance training runs copied from:

`/home/lair0/isaaclab_ws/go2_posture/logs/rsl_rl/go2_posture_direct`

Each run keeps `model_9999.pt` as the deploy checkpoint and excludes older intermediate `model_*.pt` checkpoints.

Available runs:

- `2026-05-04_23-04-13_postureON_clampON_air0.0`
- `2026-05-05_00-53-02_postureON_clampON_air0.5`
- `2026-05-05_02-41-10_postureON_clampON_air1.0`
- `2026-05-05_04-29-00_postureON_clampON_air2.0`
- `2026-05-05_06-17-00_postureON_clampOFF_air0.0`
- `2026-05-05_08-04-39_postureON_clampOFF_air0.5`
- `2026-05-05_09-53-46_postureON_clampOFF_air1.0`
- `2026-05-05_11-42-46_postureON_clampOFF_air2.0`

Select a run with:

```bash
GO2_POSTURE_RUN_NAME=2026-05-05_11-42-46_postureON_clampOFF_air2.0 \
POLICY_BACKEND=go2_posture \
python deploy/play_go2_posture_mujoco.py
```
