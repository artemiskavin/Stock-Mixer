# Repository verification

The publication package was checked on 1 November 2026.

| Check | Result |
|---|---|
| Included scope | Nine configurations, 21 completed runs, 2,100 epoch records |
| Modified selections | All 18 existing minimum-validation-loss results match their logs |
| Aggregate statistics | All six arithmetic means and sample standard deviations recomputed |
| Shuffle replay | Per-seed permutation hashes agree across the six modified configurations for all 100 epochs |
| Source syntax | All Python files parse; all JSON files decode |
| Model construction | All nine parameter counts and three author activation arrangements verified on CPU |
| Initial weights | Six modified seed-0 CPU initial hashes match the historical records |
| Numerical reports | Original reports retained; current model hashes match their documented compatibility reports |
| Upstream reference | Source and dataset identities checked against the frozen GitHub revision |
| Report | Six-page PDF, complete LaTeX source, and recorded table values checked |
| Upload contents | Local links, ignored paths, credential patterns, private paths and file sizes checked |

Model construction checks performed no forward passes or optimizer updates.
This verification did not install a fresh environment or repeat GPU training.
Historical author runs did not save initial weights, so their historical initial
states cannot be verified retrospectively. Existing test observations and
checkpoint rules are unchanged.

Git attributes preserve file bytes so line-ending conversion does not invalidate
source or artifact hashes. `CHECKSUMS.sha256` covers the final package files,
excluding the checksum list itself. Upstream attribution and the release's
license-file status are recorded in `ATTRIBUTION.md`.
