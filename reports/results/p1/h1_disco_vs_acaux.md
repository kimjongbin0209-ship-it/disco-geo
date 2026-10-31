# h1_disco_vs_acaux (reference: actor_critic_aux)

| condition | endpoint | mean | per-seed | CI95 | paired diff vs reference | sign-flip p |
| --- | --- | --- | --- | --- | --- | --- |
| actor_critic_aux | seen_hidden_target_location_rmse | 4.3524 | 4.327, 4.393, 4.337 | [3.558, 5.081] | - | - |
| actor_critic_aux | known_wall_balanced_iou | 0.2897 | 0.286, 0.285, 0.298 | [0.259, 0.318] | - | - |
| actor_critic_aux | task_return (secondary) | 0.2917 | 0.000, 0.125, 0.750 | [0.000, 0.833] | - | - |
| actor_critic_aux | memory_reset_return_drop (secondary) | -0.0833 | -0.125, 0.000, -0.125 | [-0.583, 0.333] | - | - |
| disco | seen_hidden_target_location_rmse | 4.2744 | 4.265, 4.248, 4.310 | [3.455, 5.007] | -0.0780 (n=3) | 0.2500 |
| disco | known_wall_balanced_iou | 0.2892 | 0.286, 0.288, 0.293 | [0.260, 0.317] | -0.0006 (n=3) | 1.0000 |
| disco | task_return (secondary) | 0.0833 | 0.000, 0.125, 0.125 | [0.000, 0.333] | -0.2083 (n=3) | 1.0000 |
| disco | memory_reset_return_drop (secondary) | 0.0833 | 0.000, 0.125, 0.125 | [0.000, 0.333] | +0.1667 (n=3) | 0.2500 |
