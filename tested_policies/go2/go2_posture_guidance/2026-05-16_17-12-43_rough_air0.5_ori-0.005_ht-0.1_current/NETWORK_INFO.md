# Go2 Posture Guidance Policy

Source run:

`/home/lair0/isaaclab_ws/go2_posture/logs/rsl_rl/go2_posture_direct/2026-05-16_17-12-43_rough_air0.5_ori-0.005_ht-0.1_current`

Committed checkpoint:

`model_4000.pt`

Required deploy files:

- `params/env.yaml`
- `params/agent.yaml`
- `model_4000.pt`
- `exported/concurrent_explicit_estimator.pth`

Network summary:

- RLvRL teacher-student policy is enabled.
- RMA is disabled.
- Concurrent state estimator mode is `explicit`.
- Actor observation dimension is 59.
- Adaptation history dimension is 240.
- RLvRL latent dimension is 8.
- Policy body input dimension is 67 = actor observation 59 + latent 8.
- Policy output dimension is 12.
- Explicit concurrent estimator input dimension is 900 = single observation 45 * history 20.
- Explicit concurrent estimator output dimension is 11 = velocity 3 + contacts 4 + foot heights 4.

Inference flow:

1. Build adaptation history observation: 48 * 5 = 240.
2. `adaptation_module` maps history 240 to latent 8.
3. Build explicit concurrent estimator history: 45 * 20 = 900.
4. `concurrent_explicit_estimator` maps history 900 to estimated state 11.
5. Build actor observation 59 = estimator current obs 45 + estimated state 11 + posture guide 3.
6. `actor_body` maps actor observation 59 + latent 8 to action 12.

The onboard deploy code must support `concurrent_state_estimator_mode: explicit` and load
`exported/concurrent_explicit_estimator.pth`.
