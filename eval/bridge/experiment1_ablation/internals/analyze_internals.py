"""Summarize F1 internal-diagnostic records (records/<TASK>/episode_*.npz,
written by f1_internals_inference.py). CPU only:

    python analyze_internals.py records/

Prints, per condition (real / shuffle / mean / removed):
- shapes and masks: prefix, und and gen lengths, how many und / gen positions
  the action tokens are allowed to attend to, the action tokens' first RoPE
  position id;
- attention: mass of the action queries on und / gen-history / gen-future /
  action keys, averaged over layers, plus the largest per-layer gen mass;
- how different each condition's gen block is from the real one, separately
  for the encoded history and the generated future tokens;
- activations: gen-block key/value norms relative to the real condition;
- action branch: relative L2 to the real-condition action on identical
  inputs, translation and rotation magnitude, gripper mean, NaN count;
and the in-situ mask test (allowed vs masked perturbation of gen positions).
"""

import pathlib
import sys

import numpy as np

CONDITIONS = ("real", "shuffle", "mean", "removed")


def _table(rows, cols) -> None:
    print("| condition | n | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for name in CONDITIONS:
        r = rows[name]
        if not r:
            continue
        vals = []
        for k in cols:
            x = np.array([d[k] for d in r], dtype=float)
            vals.append(f"{int(np.nansum(x))}" if k == "nan" else
                        ("—" if np.all(np.isnan(x)) else f"{np.nanmean(x):.4f} ± {np.nanstd(x):.4f}"))
        print(f"| {name} | {len(r)} | " + " | ".join(vals) + " |")


def main(root: pathlib.Path) -> None:
    rows = {c: [] for c in CONDITIONS}
    mask_tests = []
    files = sorted(root.glob("*/episode_*.npz"))
    n_rec = 0
    for f in files:
        for rec in np.load(f, allow_pickle=True)["records"]:
            n_rec += 1
            real = rec["conditions"]["real"]
            rn = np.linalg.norm(real["action"]) + 1e-8
            for name, c in rec["conditions"].items():
                m = np.asarray(c["attn_mass_und_hist_future_act_per_layer"])  # [layers, 4]
                kd = c["gen_rel_diff_to_real"]
                kn = c["gen_key_norm_per_layer"]
                vn = c["gen_value_norm_per_layer"]
                a = c["action"]
                rows[name].append({
                    "prefix_len": c["prefix_len"], "gen_len": c["gen_len"],
                    "allowed_und": c["allowed_und"], "allowed_gen": c["allowed_gen"],
                    "act_pos0": c["act_first_position_id"],
                    "mass_und": m[:, 0].mean(), "mass_hist": m[:, 1].mean(), "mass_future": m[:, 2].mean(),
                    "mass_act": m[:, 3].mean(),
                    "max_gen_layer": int((m[:, 1] + m[:, 2]).argmax()), "max_gen_mass": (m[:, 1] + m[:, 2]).max(),
                    "hist_k_diff": kd.get("hist_k", np.nan), "hist_v_diff": kd.get("hist_v", np.nan),
                    "future_k_diff": kd.get("future_k", np.nan), "future_v_diff": kd.get("future_v", np.nan),
                    "key_norm_ratio": np.nan if kn is None else float(np.mean(kn / real["gen_key_norm_per_layer"])),
                    "value_norm_ratio": np.nan if vn is None else float(np.mean(vn / real["gen_value_norm_per_layer"])),
                    "rel_l2_to_real": c["l2_to_real"] / rn,
                    "translation": float(np.linalg.norm(a[:, :3], axis=-1).mean()),
                    "rotation": float(np.linalg.norm(a[:, 3:6], axis=-1).mean()),
                    "gripper": float(a[:, 6].mean()),
                    "nan": c["action_nan"],
                })
            if "mask_test" in rec:
                mask_tests.append(rec["mask_test"])
    print(f"records (replans): {n_rec}  episodes: {len(files)}\n")

    print("shapes / masks (constant across replans unless shown as a range)")
    print("| condition | prefix | gen | allowed und | allowed gen | action first position id |")
    print("|---|---|---|---|---|---|")
    for name in CONDITIONS:
        r = rows[name]
        if not r:
            continue
        rng = lambda k: (f"{min(d[k] for d in r)}" if min(d[k] for d in r) == max(d[k] for d in r)
                         else f"{min(d[k] for d in r)}–{max(d[k] for d in r)}")
        print(f"| {name} | {rng('prefix_len')} | {rng('gen_len')} | {rng('allowed_und')} | {rng('allowed_gen')} | {rng('act_pos0')} |")

    print("\nattention mass of the action queries (mean over layers and denoising steps)")
    cols = ["mass_und", "mass_hist", "mass_future", "mass_act", "max_gen_mass"]
    _table(rows, cols)
    print("\nhow different each condition's gen block is from the real one (relative L2, mean over layers)")
    _table(rows, ["hist_k_diff", "hist_v_diff", "future_k_diff", "future_v_diff", "key_norm_ratio", "value_norm_ratio"])
    print("\naction branch on identical inputs")
    cols = ["rel_l2_to_real", "translation", "rotation", "gripper", "nan"]
    _table(rows, cols)
    if rows["real"]:
        layers = np.bincount([d["max_gen_layer"] for d in rows["real"]])
        print(f"\nreal: layer with most attention on gen (mode over replans): {int(layers.argmax())}")

    if mask_tests:
        allowed = np.array([t["allowed_change"] for t in mask_tests])
        masked = np.array([t["masked_change"] for t in mask_tests])
        print(f"\nmask test over {len(mask_tests)} episodes, {allowed.shape[1]} gen positions each:")
        print(f"  perturbing an allowed gen position: min change {allowed.min():.2e}, median {np.median(allowed):.2e}")
        print(f"  same position switched off in the model's pad mask: max change {masked.max():.2e}")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "records"))
