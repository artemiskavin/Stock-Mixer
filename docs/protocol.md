# Focused 21-run protocol

This publication contains six modified configurations with three completed
seed pairs each, plus three released-trainer activation executions.
It excludes the uniform-activation modified full-model study, released-
arrangement modified pipeline, extensions, corrected-sampler studies,
beta-zero interventions, and execution-alignment/recording comparison runs.

NASDAQ: 1,026 stocks, five features, lookback 16, horizon 1, market width 20
wherever stock mixing exists. All runs complete 100 epochs without early
stopping. Original offsets 0..755 are shuffled in place each epoch and the
first 740 used. Possible training targets are 16..771; validation targets
756..1007 and test targets 1008..1244 (237 days). The original possible
16-target training/validation overlap is retained.

Loss is the released masked MSE plus 0.1 times the unweighted pairwise
ranking component, with original denominators and masks. Adam learning
rate is 0.001. Surviving modified mixer sites use HardSwish; recurrent
gates stay native. No-TriU removes triangular mixing, not all temporal
processing. No-stock removes the stock branch AND its associated head.

Modified runs use GPU-preloaded arrays and batched metrics. Fifteen use
AMP FP16 with FP32 protected heads/returns/loss; GRU uses FP32 after a
failed FP16 compatibility check. Actual eager/compiled/compiled-head
settings are retained per run. Existing numerical preflight reports are
included; they are historical checks, not a guarantee for a different GPU.
Modified Adam requests foreach=False, fused=False; gradient clearing uses
set_to_none=False. Modified TF32 overrides remain in the copied source.

Author runs use GELU/GELU/HardSwish, GELU/GELU/GELU, and GELU/GELU/Sigmoid.
Their original trainers use ordinary eager FP32, installed backend defaults,
Adam(model.parameters(), lr=0.001), optimizer.zero_grad(), and per-window
CPU-to-device delivery. The two stock-activation variants change only the
stock activation statement in model.py.

Reported modified results reuse the EXISTING minimum-validation-combined-
loss checkpoint (historically named secondary_lowest_loss). The original
selection.json files retain their historical labels: primary is highest
validation Pearson IC within 1.10 times minimum validation loss; secondary
is earliest minimum loss. Exact IC ties use lower loss then earlier epoch.
No selections were changed for publication.

Author records retain rounded console values. Those trainers saved no
weights or complete optimizer/RNG states; rounded candidate epochs do not
establish exact recoverable selected checkpoints. The CSV and original
summary preserve available observations and selection uncertainty.

Official IC is mean daily Pearson correlation. Official-code RIC is daily
Pearson IC mean/population SD, not Spearman. Precision@10 counts nonnegative
returns among the selected valid top ten. Gross top-five Sharpe is daily
portfolio mean/population SD times 15.87, with zero costs. Evaluator MSE
differs from the training loss's MSE denominator.

Comparisons are exploratory across differing execution settings. The test
period has already been inspected. No new training, evaluation, or test-
driven checkpoint selection was performed while preparing this package.

## Publication source version

Comments, docstrings, and diagnostic messages have been standardized for this
source distribution. Numerical settings and control flow are unchanged. Model
ASTs are identical before and after comment edits. Original preflight reports
are retained; three derived `publication_preflight.json` files map comment-only
model byte-hash changes to those archived numerical results. This mapping is a
source-equivalence check, not a new numerical validation. Historical execution
hashes and result records remain preserved.
