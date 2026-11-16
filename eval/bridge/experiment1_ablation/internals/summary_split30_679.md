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
| real | 1800 | 0.4874 ± 0.0161 | 0.0233 ± 0.0009 | 0.2768 ± 0.0080 | 0.2126 ± 0.0159 | 0.4801 ± 0.0195 |
| shuffle | 1800 | 0.4874 ± 0.0161 | 0.0233 ± 0.0009 | 0.2768 ± 0.0080 | 0.2125 ± 0.0159 | 0.4803 ± 0.0195 |
| mean | 1800 | 0.4896 ± 0.0165 | 0.0234 ± 0.0009 | 0.2741 ± 0.0085 | 0.2129 ± 0.0158 | 0.4778 ± 0.0199 |
| removed | 1800 | 0.7572 ± 0.0123 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.2428 ± 0.0123 | 0.0000 ± 0.0000 |

how different each condition's gen block is from the real one (relative L2, mean over layers)
| condition | n | hist_k_diff | hist_v_diff | future_k_diff | future_v_diff | key_norm_ratio | value_norm_ratio |
|---|---|---|---|---|---|---|---|
| real | 1800 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| shuffle | 1800 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0110 ± 0.0018 | 0.0272 ± 0.0041 | 1.0000 ± 0.0000 | 1.0000 ± 0.0002 |
| mean | 1800 | 0.0626 ± 0.0131 | 0.1947 ± 0.0428 | 0.1412 ± 0.0305 | 0.3837 ± 0.0587 | 0.9892 ± 0.0031 | 0.9242 ± 0.0163 |
| removed | 1800 | — | — | — | — | — | — |

action branch on identical inputs
| condition | n | rel_l2_to_real | translation | rotation | gripper | nan |
|---|---|---|---|---|---|---|
| real | 1800 | 0.0000 ± 0.0000 | 0.9162 ± 0.6786 | 0.9129 ± 0.7098 | 0.1699 ± 0.8757 | 0 |
| shuffle | 1800 | 0.0033 ± 0.0126 | 0.9162 ± 0.6787 | 0.9127 ± 0.7096 | 0.1700 ± 0.8760 | 0 |
| mean | 1800 | 0.0090 ± 0.0318 | 0.9156 ± 0.6788 | 0.9126 ± 0.7103 | 0.1697 ± 0.8759 | 0 |
| removed | 1800 | 0.4114 ± 0.3078 | 0.7683 ± 0.6134 | 0.7978 ± 0.6321 | 0.1714 ± 0.8303 | 0 |

real: layer with most attention on gen (mode over replans): 1

mask test over 96 episodes, 5 gen positions each:
  perturbing an allowed gen position: min change 2.88e-02, median 9.01e-02
  same position switched off in the model's pad mask: max change 0.00e+00
