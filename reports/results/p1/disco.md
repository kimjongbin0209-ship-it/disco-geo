# disco (reference: fresh)

| condition | endpoint | mean | per-seed | CI95 | paired diff vs reference | sign-flip p |
| --- | --- | --- | --- | --- | --- | --- |
| fresh | seen_hidden_target_location_rmse | 4.2744 | 4.265, 4.248, 4.310 | [3.455, 5.007] | - | - |
| fresh | known_wall_balanced_iou | 0.2892 | 0.286, 0.288, 0.293 | [0.260, 0.317] | - | - |
| fresh | task_return (secondary) | 0.0833 | 0.000, 0.125, 0.125 | [0.000, 0.333] | - | - |
| fresh | memory_reset_return_drop (secondary) | 0.0833 | 0.000, 0.125, 0.125 | [0.000, 0.333] | - | - |
| teacher_only | seen_hidden_target_location_rmse | 4.2808 | 4.282, 4.247, 4.313 | [3.464, 5.014] | +0.0064 (n=3) | 0.5000 |
| teacher_only | known_wall_balanced_iou | 0.2897 | 0.290, 0.288, 0.291 | [0.260, 0.318] | +0.0005 (n=3) | 0.7500 |
| teacher_only | task_return (secondary) | 0.1250 | 0.125, 0.125, 0.125 | [0.000, 0.375] | +0.0417 (n=3) | 1.0000 |
| teacher_only | memory_reset_return_drop (secondary) | 0.0833 | 0.000, 0.125, 0.125 | [0.000, 0.333] | +0.0000 (n=3) | 1.0000 |
| teacher_only_teacher_acting | task_return (secondary) | 0.0833 | 0.125, 0.000, 0.125 | [0.000, 0.252] | +0.0000 (n=3) | 1.0000 |
| untrained_init | seen_hidden_target_location_rmse | 4.2434 | 4.237, 4.261, 4.232 | [3.461, 4.967] | -0.0310 (n=3) | 0.5000 |
| untrained_init | known_wall_balanced_iou | 0.2892 | 0.288, 0.289, 0.290 | [0.254, 0.320] | +0.0000 (n=3) | 1.0000 |
| untrained_init | task_return (secondary) | 0.0417 | 0.000, 0.125, 0.000 | [0.000, 0.250] | -0.0417 (n=3) | 1.0000 |
| untrained_init | memory_reset_return_drop (secondary) | 0.0000 | 0.000, 0.000, 0.000 | [-0.167, 0.167] | -0.0833 (n=3) | 0.5000 |
