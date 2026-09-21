# IsoCal-TTA: Label-Free Online Calibration for Test-Time Adaptation

Code accompanying the AAAI 2027 Student Abstract *"IsoCal-TTA: Label-Free
Online Calibration for Test-Time Adaptation"*.

IsoCal-TTA decouples test-time adaptation (updating a model's BatchNorm
affine parameters via entropy minimization, exactly as in TENT) from
calibration (a continuously refit, label-free isotonic map that tracks the
model's confidence-correctness relationship as it shifts). A label-free
instance-level correctness estimate — agreement across style- and
content-perturbed variants of each input — supplies pseudo-supervision for
this online calibration map, which is always fit only on preceding batches
and never on the batch it is used to calibrate.

## Repository structure

```
isocal-tta/
├── scripts/
│   ├── cifar10c_severity_sweep.py    # Table 1: full severity x corruption grid, 3 seeds
│   ├── cifar10c_dynamic_stream.py    # Table 2: non-stationary corruption stream, 3 seeds
│   └── auroc_signal_diagnostic.py    # Supplementary Table S4: fusion-formula diagnostic
├── figures/
│   ├── plot_severity_curve.py        # reproduces the severity-ECE figure from results/
│   └── plot_dynamic_stream.py        # reproduces the dynamic-stream ECE figure
├── results/
│   └── cgt_tta_results_cifar10.csv   # full raw results (8 modes x 5 severities x 15 corruptions x 3 seeds)
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Download [CIFAR-10-C](https://zenodo.org/record/2535967) and place the
extracted `.npy` files under `./data/CIFAR-10-C/`. CIFAR-10 itself
downloads automatically via `torchvision` on first run (used for the clean
validation split that temperature scaling and EATA's Fisher term are fit
on — never for anything else).

Each script expects a trained ResNet-18 (CIFAR variant) checkpoint at
`./checkpoints/resnet18_best_seed_42.pt`. Training that checkpoint is not
included here; any standard CIFAR-10 ResNet-18 training recipe reaching
~95% clean accuracy is sufficient, since all reported methods adapt online
from this fixed starting point.

## Reproducing the paper's results

```bash
# Table 1 (severity grid, all 15 corruptions, 3 seeds) — resumable, safe to interrupt
python scripts/cifar10c_severity_sweep.py

# Table 2 (dynamic non-stationary stream, 3 seeds)
python scripts/cifar10c_dynamic_stream.py

# Supplementary Table S4 (AUROC diagnostic behind the fusion formula)
python scripts/auroc_signal_diagnostic.py    # set FUSION_FORMULA = "correct" or "buggy"
```

`results/cgt_tta_results_cifar10.csv` contains the exact raw results
(per-mode, per-severity, per-corruption, per-seed accuracy and ECE) used to
produce every number in Table 1 and the abstract, including the
accuracy-preservation table in the supplementary material. Every summary
statistic in the paper can be recomputed directly from this file, e.g.:

```python
import pandas as pd
df = pd.read_csv("results/cgt_tta_results_cifar10.csv")
df.groupby(["mode", "severity"])[["acc", "ece"]].mean()
```

## Method summary

For each test batch:
1. Adapt the model's BatchNorm affine parameters via one entropy-minimization step (TENT).
2. Compute a label-free instance-level correctness score `s = style_invariance * content_invariance` from agreement across stochastically style- and content-perturbed variants of the input.
3. Output calibrated confidence `g(c)`, where `g` is an isotonic regression fit **only on preceding batches**.
4. Only after producing that output, add `(c, s)` to the sliding window (size 2048) and refit `g` for the next batch.

Step 4 happening strictly after step 3 is what keeps the evaluation causal —
see the Method section and Algorithm 1 in the supplementary material for
the precise statement and rationale.

## A note on an earlier, incorrect fusion formula

An early version of this codebase computed the instance-level score as
`s = (1 - content_invariance) * style_invariance`, based on the assumption
that high content invariance indicates a collapsed, style-only prediction.
An AUROC diagnostic against ground-truth correctness (evaluation-only;
never used for adaptation or calibration) showed this assumption does not
hold: both signals are independently positive predictors of correctness in
this setting, and inverting one caused a severe rank reversal (fused-signal
AUROC dropping to 0.20-0.31, worse than random). `auroc_signal_diagnostic.py`
reproduces both the broken and corrected formula for direct comparison
(`FUSION_FORMULA = "buggy"` vs. `"correct"`); every other script in this
repository uses only the corrected formula.

## Citation

```bibtex
@inproceedings{yadav2027isocaltta,
  title     = {IsoCal-TTA: Label-Free Online Calibration for Test-Time Adaptation},
  author    = {Yadav, Vineet},
  booktitle = {AAAI Student Abstract Program},
  year      = {2027}
}
```

## License

MIT — see [LICENSE](LICENSE).
