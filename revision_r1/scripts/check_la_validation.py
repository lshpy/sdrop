"""Read-only exploratory LA sensitivity audit from saved prediction dumps.

Run from any directory: python3 check_la_validation.py
Requires numpy/scipy; no training. Writes audit outputs beside this script.
Tau grid and smallest-tau tie rule match the audit described in READ_ME_REVIEW_KO.
Validation group labels are used, so this is not label-free tuning.
"""
from pathlib import Path
import json
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "00_ACTIVE_SpringerML" / "paper_results"
TAUS = np.round(np.arange(0, 3.00001, 0.05), 2)


def curve(dump, split):
    y = dump[split + "_y"].astype(int)
    groups = 2 * y + dump[split + "_places"].astype(int)
    prior = float(dump["val_y"].mean())
    log_prior = np.log([1 - prior, prior])
    logits = dump[split + "_logits"]
    values = []
    for tau in TAUS:
        pred = (logits - tau * log_prior).argmax(1)
        worst = min(100 * np.mean(pred[groups == k] == y[groups == k]) for k in range(4))
        values.append((worst, 100 * np.mean(pred == y)))
    return np.array(values)


def main():
    rows = []
    for seed in range(10):
        erm_dir = "checkpoints_rev" if seed < 6 else "checkpoints_rev2b"
        erm = ROOT / erm_dir / f"waterbirds_none_rate0.1_none_ep30_seed{seed}_dump.npz"
        top = ROOT / "checkpoints_final" / f"waterbirds_sdrop_energy_rate0.1_L4_ep30_nmmean_seltopk_seed{seed}_dump.npz"
        with np.load(erm) as e, np.load(top) as a:
            ev, et = curve(e, "val"), curve(e, "test")
            av, at = curve(a, "val"), curve(a, "test")
            je, ja = int(ev[:, 0].argmax()), int(av[:, 0].argmax())
            rows.append({
                "seed": seed, "erm_path": str(erm), "top_path": str(top),
                "erm_val_tau": float(TAUS[je]), "top_val_tau": float(TAUS[ja]),
                "erm_val_tuned_test_wg": float(et[je, 0]),
                "top_val_tuned_test_wg": float(at[ja, 0]),
                "val_tuned_delta": float(at[ja, 0] - et[je, 0]),
                "oracle_delta": float(at[:, 0].max() - et[:, 0].max()),
            })
    summary = {}
    for endpoint in ("oracle_delta", "val_tuned_delta"):
        differences = np.array([row[endpoint] for row in rows])
        n = len(differences)
        summary[endpoint] = {
            "n": n, "mean_pp": float(differences.mean()),
            "wins": int((differences > 0).sum()),
            "ci95_halfwidth_pp": float(stats.t.ppf(.975, n - 1) * stats.sem(differences)),
            "paired_t_p_uncorrected": float(stats.ttest_1samp(differences, 0).pvalue),
        }
    result = {
        "protocol": "Additional exploratory audit; not preregistered. Validation-prior logit shift. Tau0..3 by.05 independently selected per model by maximum validation worst-group, smallest-tau tie rule; evaluate selected tau once on test. Uses validation group labels. P values are uncorrected. Oracle endpoint maximizes test worst-group as a diagnostic.",
        "seeds": list(range(10)), "summary": summary, "rows": rows,
    }
    (HERE / "la_validation_sensitivity.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
