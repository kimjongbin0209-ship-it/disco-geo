# diagnostics diagnostics

## Scripted references (Memory Maze 9x9, policy-final-test seeds)

| policy | episodes | reward / 1000 steps |
| --- | --- | --- |
| forward:0.5 | 32 | 0.062 |
| forward_left:0.5 | 32 | 0.406 |
| uniform | 32 | 0.156 |

## Positive controls (cue recall)

| run | done | per-trial reward, last quarter | criterion | pass | entropy last |
| --- | --- | --- | --- | --- | --- |
| actor_critic_aux-control_bandit-seed0 | 300000/300000 | 0.993 | >= 0.8 | True | 0.052 |
| actor_critic_aux-control_bandit-seed1 | 300000/300000 | 0.985 | >= 0.8 | True | 0.095 |
| actor_critic_aux-control_bandit-seed2 | 300000/300000 | 0.982 | >= 0.8 | True | 0.104 |
| actor_critic_aux-control_memory-seed0 | 2000000/2000000 | 0.164 | >= 0.5 | False | 1.681 |
| actor_critic_aux-control_memory-seed1 | 2000000/2000000 | 0.162 | >= 0.5 | False | 1.710 |
| actor_critic_aux-control_memory-seed2 | 2000000/2000000 | 0.158 | >= 0.5 | False | 1.707 |
| actor_critic_aux-control_reactive-seed0 | 2000000/2000000 | 0.167 | >= 0.8 | False | 0.007 |
| actor_critic_aux-control_reactive-seed1 | 2000000/2000000 | 0.983 | >= 0.8 | True | 0.064 |
| actor_critic_aux-control_reactive-seed2 | 2000000/2000000 | 0.666 | >= 0.8 | False | 0.013 |
| disco-control_bandit-seed0 | 300000/300000 | 0.035 | >= 0.8 | False | 0.728 |
| disco-control_bandit-seed1 | 300000/300000 | 0.020 | >= 0.8 | False | 0.431 |
| disco-control_bandit-seed2 | 300000/300000 | 0.025 | >= 0.8 | False | 0.541 |
| disco-control_bandit_lr_1e-2-seed0 | 300000/300000 | 0.988 | >= 0.8 | True | 0.080 |
| disco-control_bandit_lr_1e-2-seed1 | 300000/300000 | 0.103 | >= 0.8 | False | 1.253 |
| disco-control_bandit_lr_1e-2-seed2 | 300000/300000 | 0.826 | >= 0.8 | True | 0.729 |
| disco-control_memory-seed0 | 2000000/2000000 | 0.166 | >= 0.5 | False | 1.706 |
| disco-control_memory-seed1 | 2000000/2000000 | 0.163 | >= 0.5 | False | 1.742 |
| disco-control_memory-seed2 | 2000000/2000000 | 0.161 | >= 0.5 | False | 1.646 |
| disco-control_memory_lr_1e-2-seed0 | 2000000/2000000 | 0.165 | >= 0.5 | False | 1.331 |
| disco-control_memory_lr_1e-2-seed1 | 2000000/2000000 | 0.164 | >= 0.5 | False | 1.312 |
| disco-control_memory_lr_1e-2-seed2 | 2000000/2000000 | 0.160 | >= 0.5 | False | 1.225 |
| disco-control_reactive-seed0 | 2000000/2000000 | 0.189 | >= 0.8 | False | 0.859 |
| disco-control_reactive-seed1 | 2000000/2000000 | 0.186 | >= 0.8 | False | 0.938 |
| disco-control_reactive-seed2 | 2000000/2000000 | 0.177 | >= 0.8 | False | 0.865 |
| disco-control_reactive_lr_1e-2-seed0 | 2000000/2000000 | 0.167 | >= 0.8 | False | 0.710 |
| disco-control_reactive_lr_1e-2-seed1 | 2000000/2000000 | 0.166 | >= 0.8 | False | 0.709 |
| disco-control_reactive_lr_1e-2-seed2 | 2000000/2000000 | 0.167 | >= 0.8 | False | 0.725 |

## Memory Maze

| run | done | reward/1k first 5% | reward/1k last quarter | vs uniform (95% CI) | entropy first/min/last | credit gap +- SE | rule-signal gap +- SE | target - policy entropy (first/last) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| actor_critic_aux-diag_entropy_0-seed0 | 2000000/2000000 | 0.200 | 0.296 | 1.90 [0.78, 4.62] | 1.47/0.20/0.70 | 0.0053 +- 0.0008 | 0.8484 +- 0.0024 | -/- |
| actor_critic_aux-diag_entropy_0-seed1 | 2000000/2000000 | 0.100 | 0.236 | 1.51 [0.62, 3.70] | 1.06/0.03/0.15 | 0.0057 +- 0.0009 | 0.8389 +- 0.0013 | -/- |
| actor_critic_aux-diag_entropy_0-seed2 | 2000000/2000000 | 0.160 | 0.260 | 1.66 [0.68, 4.07] | 0.97/0.04/0.28 | 0.0043 +- 0.0008 | 0.8455 +- 0.0021 | -/- |
| actor_critic_aux-diag_entropy_1e-3-seed0 | 2000000/2000000 | 0.220 | 0.334 | 2.14 [0.88, 5.21] | 1.58/1.00/1.69 | 0.0099 +- 0.0008 | 0.8433 +- 0.0008 | -/- |
| actor_critic_aux-diag_entropy_1e-3-seed1 | 2000000/2000000 | 0.190 | 0.316 | 2.02 [0.83, 4.93] | 1.63/0.84/1.64 | 0.0173 +- 0.0007 | 0.8367 +- 0.0013 | -/- |
| actor_critic_aux-diag_entropy_1e-3-seed2 | 2000000/2000000 | 0.270 | 0.304 | 1.95 [0.80, 4.74] | 1.26/0.74/1.67 | 0.0098 +- 0.0011 | 0.8373 +- 0.0014 | -/- |
| actor_critic_aux-diag_fresh-seed0 | 2000000/2000000 | 0.230 | 0.202 | 1.29 [0.53, 3.18] | 1.66/1.53/1.78 | 0.0101 +- 0.0009 | 0.8320 +- 0.0027 | -/- |
| actor_critic_aux-diag_fresh-seed1 | 2000000/2000000 | 0.160 | 0.196 | 1.26 [0.51, 3.08] | 1.77/1.70/1.78 | 0.0191 +- 0.0009 | 0.8367 +- 0.0007 | -/- |
| actor_critic_aux-diag_fresh-seed2 | 2000000/2000000 | 0.370 | 0.212 | 1.36 [0.55, 3.33] | 1.57/1.29/1.78 | 0.0126 +- 0.0013 | 0.8248 +- 0.0009 | -/- |
| disco-diag_fresh-seed0 | 2000000/2000000 | 0.260 | 0.202 | 1.29 [0.53, 3.18] | 1.54/0.16/1.76 | 0.0000 +- 0.0008 | -0.0273 +- 0.0105 | -0.004/-0.001 |
| disco-diag_fresh-seed1 | 2000000/2000000 | 0.200 | 0.200 | 1.28 [0.52, 3.14] | 1.64/1.13/1.79 | -0.0000 +- 0.0001 | 0.0013 +- 0.0014 | -0.000/0.000 |
| disco-diag_fresh-seed2 | 2000000/2000000 | 0.250 | 0.124 | 0.79 [0.32, 1.98] | 1.41/0.88/1.79 | 0.0003 +- 0.0002 | -0.0046 +- 0.0027 | -0.025/-0.000 |
| disco-diag_lr_1e-2-seed0 | 2000000/2000000 | 0.200 | 0.182 | 1.17 [0.47, 2.87] | 1.69/0.46/1.79 | -0.0002 +- 0.0001 | 0.0014 +- 0.0074 | 0.004/0.000 |
| disco-diag_lr_1e-2-seed1 | 2000000/2000000 | 0.180 | 0.168 | 1.08 [0.44, 2.65] | 1.76/1.45/1.79 | -0.0008 +- 0.0007 | 0.0013 +- 0.0006 | 0.002/0.000 |
| disco-diag_lr_1e-2-seed2 | 2000000/2000000 | 0.200 | 0.174 | 1.11 [0.45, 2.74] | 1.74/0.58/1.79 | -0.0001 +- 0.0001 | 0.0008 +- 0.0018 | 0.007/0.000 |
| disco-diag_lr_3e-3-seed0 | 2000000/2000000 | 0.160 | 0.202 | 1.29 [0.53, 3.18] | 1.71/1.02/1.79 | -0.0019 +- 0.0020 | -0.0058 +- 0.0047 | 0.003/0.000 |
| disco-diag_lr_3e-3-seed1 | 2000000/2000000 | 0.180 | 0.184 | 1.18 [0.48, 2.90] | 1.71/0.41/1.79 | 0.0001 +- 0.0001 | -0.0012 +- 0.0011 | 0.008/0.000 |
| disco-diag_lr_3e-3-seed2 | 2000000/2000000 | 0.330 | 0.148 | 0.95 [0.38, 2.34] | 1.45/0.84/1.79 | -0.0007 +- 0.0003 | -0.0110 +- 0.0026 | -0.007/0.000 |

Credit gap: mean change of log pi(a_t) from one update on steps followed by a reward within the window minus steps that are not. A learner that uses the reward should show a positive gap; a rule that drifts regardless of reward shows ~0. Diagnostic only; not a study result.
