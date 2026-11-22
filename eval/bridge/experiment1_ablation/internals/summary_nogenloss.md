records (replans): 1800  episodes: 96

shapes / masks (constant across replans unless shown as a range)
| condition | prefix | gen | allowed und | allowed gen | action first position id |
|---|---|---|---|---|---|
| real | 1013 | 709 | 262–266 | 709 | 971–975 |
| shuffle | 1013 | 709 | 262–266 | 709 | 971–975 |
| mean | 1013 | 709 | 262–266 | 709 | 971–975 |
| removed | 304 | 0 | 262–266 | 0 | 262–266 |

attention mass of the action queries (mean over layers and denoising steps)
| condition | n | mass_und | mass_hist | mass_future | mass_act | max_gen_mass |
|---|---|---|---|---|---|---|
| real | 1800 | 0.5748 ± 0.0195 | 0.1961 ± 0.0091 | 0.0094 ± 0.0005 | 0.2196 ± 0.0158 | 0.3065 ± 0.0244 |
| shuffle | 1800 | 0.5749 ± 0.0196 | 0.1961 ± 0.0091 | 0.0094 ± 0.0005 | 0.2196 ± 0.0158 | 0.3064 ± 0.0245 |
| mean | 1800 | 0.4827 ± 0.0156 | 0.3021 ± 0.0089 | 0.0175 ± 0.0007 | 0.1978 ± 0.0152 | 0.4540 ± 0.0183 |
| removed | 1800 | 0.7604 ± 0.0130 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.2396 ± 0.0130 | 0.0000 ± 0.0000 |

how different each condition's gen block is from the real one (relative L2, mean over layers)
| condition | n | hist_k_diff | hist_v_diff | future_k_diff | future_v_diff | key_norm_ratio | value_norm_ratio |
|---|---|---|---|---|---|---|---|
| real | 1800 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| shuffle | 1800 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0740 ± 0.0107 | 0.2365 ± 0.0264 | 1.0000 ± 0.0001 | 0.9998 ± 0.0003 |
| mean | 1800 | 0.5216 ± 0.0076 | 0.8364 ± 0.0172 | 0.5726 ± 0.0134 | 0.8800 ± 0.0155 | 0.9725 ± 0.0053 | 1.0371 ± 0.0218 |
| removed | 1800 | — | — | — | — | — | — |

action branch on identical inputs
| condition | n | rel_l2_to_real | translation | rotation | gripper | nan |
|---|---|---|---|---|---|---|
| real | 1800 | 0.0000 ± 0.0000 | 0.9218 ± 0.6487 | 0.9332 ± 0.7009 | 0.1491 ± 0.8777 | 0 |
| shuffle | 1800 | 0.0033 ± 0.0088 | 0.9217 ± 0.6487 | 0.9331 ± 0.7008 | 0.1493 ± 0.8775 | 0 |
| mean | 1800 | 0.0882 ± 0.1234 | 0.8693 ± 0.6352 | 0.8739 ± 0.6743 | 0.1540 ± 0.8708 | 0 |
| removed | 1800 | 0.3673 ± 0.2639 | 0.8027 ± 0.6095 | 0.8368 ± 0.6398 | 0.1446 ± 0.8505 | 0 |

real: layer with most attention on gen (mode over replans): 1

mask test over 96 episodes, 5 gen positions each:
  perturbing an allowed gen position: min change 3.28e-02, median 8.46e-02
  same position switched off in the model's pad mask: max change 0.00e+00
