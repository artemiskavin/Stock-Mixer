# StockMixer: Reproduction and Extensions

**Kavin Karthik Yuvaraj - Shiv Nadar University Chennai**

A focused NASDAQ reproduction package containing **21 completed 100-epoch
runs**: six modified configurations across the three predefined seed pairs,
and three released-trainer activation executions. This folder contains only
their implementations, supporting numerical checks, recorded results, and
the project report. The untouched local research archive is preserved separately.

## Included configurations

| Group | Configuration | Source |
|---|---|---|
| Component removal | Without indicator mixing | `ablations/no_indicator/StockMixer` |
| Component removal | Without TriU mixing | `ablations/no_temporal/StockMixer` |
| Component removal | Without stock mixing | `ablations/no_stock/StockMixer` |
| Temporal baseline | LSTM | `baselines/LSTM` |
| Temporal baseline | GRU | `baselines/GRU` |
| Temporal + stock | LSTM + stock mixing | `baselines/LSTM_stock_mixing` |
| Author trainer | GELU / GELU / HardSwish | `authors/gelu_gelu_hardswish/StockMixer` |
| Author trainer | GELU / GELU / GELU | `authors/gelu_gelu_gelu/StockMixer` |
| Author trainer | GELU / GELU / Sigmoid | `authors/gelu_gelu_sigmoid/StockMixer` |

The first six configurations have three completed runs each. The last three
retain the released NumPy/PyTorch seed pair. The inventory is explicit in
[run_registry.json](configs/run_registry.json); it includes no other studies.

## Results and report

### Recorded forecasting results

| Configuration | Test IC | Official RIC | Gross Sharpe@5 |
|---|---:|---:|---:|
| GELU / GELU / HardSwish | 0.0408 | 0.482 | 1.62 |
| GELU / GELU / GELU | 0.0368 | 0.414 | 1.72 |
| GELU / GELU / Sigmoid | 0.0225 | 0.242 | 0.356 |
| Without indicator mixing | 0.02764 +/- 0.00510 | 0.328 +/- 0.042 | 0.812 +/- 0.981 |
| Without TriU mixing | 0.01987 +/- 0.01569 | 0.263 +/- 0.188 | 0.896 +/- 1.075 |
| Without stock mixing | 0.03517 +/- 0.00591 | 0.459 +/- 0.099 | 1.128 +/- 0.510 |
| LSTM | 0.02751 +/- 0.00531 | 0.300 +/- 0.055 | 0.395 +/- 0.544 |
| GRU | 0.02875 +/- 0.01140 | 0.300 +/- 0.117 | -0.372 +/- 0.440 |
| LSTM plus stock mixing | 0.02365 +/- 0.00801 | 0.284 +/- 0.089 | -0.054 +/- 1.285 |

Author rows retain rounded console observations. Modified rows show arithmetic
mean +/- sample standard deviation from existing minimum-validation-loss states.
Complete MSE, Precision@10 and selection records are linked below.


- [Modified individual lowest-loss results](results/modified_individual_lowest_loss.csv)
- [Modified arithmetic means and sample SD](results/modified_mean_sample_sd.csv)
- [Recorded author-code results](results/author_recorded_results.csv)
- Per-epoch logs and existing selection files: [modified](results/modified/)
- Rounded console logs and original summaries: [authors](results/authors/)
- [Final six-page IEEE report](report/DL_Lab_Project_Report_StockMixer.pdf)
- [Single-file LaTeX source](report/DL_Lab_Project_Report_StockMixer.tex)
- [Complete LaTeX package with figures](report/DL_Lab_Project_Report_StockMixer.zip)

Modified rows use their existing minimum-validation-combined-loss selection.
Author-code rows are rounded recorded observations; those trainers did not
save weights or exact recoverable selected states. See [protocol](docs/protocol.md)
for definitions, sampling overlap, precision differences, and selection limits.
The report also discusses diagnostic investigations whose experimental source
and run folders are outside this focused package.

## Project engineering contributions

The modified implementations retain exact per-epoch metrics, all-epoch
checkpoints and complete resumable state during execution. GPU preloading,
batched metric transfer, AMP, and verified compilation address execution cost.
Historical numerical checks are included in `preflight/`. These are workflow
contributions; performance differences are not attributed solely to them.

## Source versions

Publication comments, docstrings, and diagnostic messages use concise technical
wording. Model computation, numerical thresholds, optimization, and selections
are unchanged. Run metadata retains historical source hashes alongside the
publication hashes. Original numerical reports remain in `preflight/`; three
comment-only model revisions have derived compatibility reports documenting
identical executable model syntax. No numerical tests were rerun.

## Reproduction

Recorded machine: Intel Core i5-12450H, 16 GB RAM, RTX 4060 Laptop GPU with
8 GB VRAM, Windows 11. Software: Python 3.10.21, PyTorch 2.9.1+cu128,
CUDA runtime 12.8, cuDNN 9.10.2, NumPy 1.22.1, driver 617.42.
`requirements-recorded.txt` records installed Python package versions.
The modified lock helper uses Windows `msvcrt`; the environment is Windows-
oriented. A different stack/GPU is not guaranteed bitwise identical results.

1. Create a Python 3.10 virtual environment at `.venv` and install the recorded
   dependencies. With Python 3.10 available through the Windows launcher:

   ```powershell
   py -3.10 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements-recorded.txt
   ```

   CUDA execution requires a compatible NVIDIA driver.
2. Follow [data preparation](data/README.md) to obtain and verify the four
   supplied NASDAQ arrays. No datasets are bundled.
3. Print a single command, for example:

```powershell
python tools/print_run_command.py --configuration no_indicator --seed-index 0 --run-tag reproduction_01
```

To inspect the author command instead:

```powershell
python tools/print_run_command.py --configuration gelu_gelu_hardswish
```

The helper prints commands without starting training. Execute commands one at a
time. Modified trainers save new results in a distinct `artifacts/runs/` path.
The author trainers print console metrics and do not save weights or result
files; capture their terminal output separately.
Existing numerical checks describe the original environment and should not be
represented as newly verified on another machine. Full checkpoint archives
remain local; this package includes source and result records, not pretrained
weight files or Python environments.

## Interpretation

Official IC is mean daily Pearson correlation; official-code RIC is its daily
information ratio, not Spearman. Precision@10 counts nonnegative valid top-ten
returns. Gross Sharpe@5 is the released zero-cost top-five portfolio statistic.
Evaluator MSE differs from the training MSE component. Experiments use differing
execution settings; comparisons are exploratory. The test period was already
inspected and these results are not independent confirmation.

See [verification notes](VALIDATION.md), [file checksums](CHECKSUMS.sha256),
and [upstream attribution](ATTRIBUTION.md) for the original paper, source revision,
and distinction between the released implementation and project extensions.
