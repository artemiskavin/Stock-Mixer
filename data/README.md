# NASDAQ data preparation

Data copies are excluded from Git. Obtain the four supplied NASDAQ files
from the released repository at revision `cce13598afd3ff33ae317700a85ae08db0554652`:
https://github.com/SJTU-DMTai/StockMixer/tree/cce13598afd3ff33ae317700a85ae08db0554652/dataset/NASDAQ

Required files: eod_data.pkl, mask_data.pkl, price_data.pkl, gt_data.pkl.
Their exact SHA-256 hashes are in `nasdaq_hashes.json`.

With all four files in an existing local directory, run from repository root:

```powershell
python tools/prepare_data.py --source "C:/path/to/NASDAQ"
```

This checks hashes and copies unchanged data into the nine implementation
directories expected by their original relative paths. It performs no model
imports, preprocessing, training, or evaluation. Dataset folders are ignored
by Git. A mismatching existing destination is never overwritten.
