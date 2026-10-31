# actor_critic_aux (reference: fresh)

| condition | endpoint | mean | per-seed | CI95 | paired diff vs reference | sign-flip p |
| --- | --- | --- | --- | --- | --- | --- |
| fresh | seen_hidden_target_location_rmse | 4.3524 | 4.327, 4.393, 4.337 | [3.558, 5.081] | - | - |
| fresh | known_wall_balanced_iou | 0.2897 | 0.286, 0.285, 0.298 | [0.259, 0.318] | - | - |
| fresh | task_return (secondary) | 0.2917 | 0.000, 0.125, 0.750 | [0.000, 0.833] | - | - |
| fresh | memory_reset_return_drop (secondary) | -0.0833 | -0.125, 0.000, -0.125 | [-0.583, 0.333] | - | - |
| teacher_only | seen_hidden_target_location_rmse | 4.3325 | 4.335, 4.325, 4.338 | [3.517, 5.063] | -0.0199 (n=3) | 1.0000 |
| teacher_only | known_wall_balanced_iou | 0.2891 | 0.288, 0.287, 0.292 | [0.260, 0.317] | -0.0007 (n=3) | 1.0000 |
| teacher_only | task_return (secondary) | 0.1250 | 0.125, 0.125, 0.125 | [0.000, 0.375] | -0.1667 (n=3) | 1.0000 |
| teacher_only | memory_reset_return_drop (secondary) | 0.0000 | 0.000, 0.000, 0.000 | [0.000, 0.000] | +0.0833 (n=3) | 0.5000 |
| teacher_only_teacher_acting | task_return (secondary) | 0.1250 | 0.125, 0.125, 0.125 | [0.000, 0.375] | -0.1667 (n=3) | 1.0000 |
| untrained_init | seen_hidden_target_location_rmse | 4.2434 | 4.237, 4.261, 4.232 | [3.461, 4.967] | -0.1091 (n=3) | 0.2500 |
| untrained_init | known_wall_balanced_iou | 0.2892 | 0.288, 0.289, 0.290 | [0.254, 0.320] | -0.0006 (n=3) | 1.0000 |
| untrained_init | task_return (secondary) | 0.0417 | 0.000, 0.125, 0.000 | [0.000, 0.250] | -0.2500 (n=3) | 1.0000 |
| untrained_init | memory_reset_return_drop (secondary) | 0.0000 | 0.000, 0.000, 0.000 | [-0.167, 0.167] | +0.0833 (n=3) | 0.5000 |
