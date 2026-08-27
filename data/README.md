# Data folder

This folder should contain `bank-full.csv` — the UCI Bank Marketing dataset
(45,211 rows, 17 columns, comma-separated, header included).

## Getting the data

**Option A — the file is already here.** If `bank-full.csv` is already in
this folder, you don't need to do anything else.

**Option B — download it yourself** (recommended if you want the exact,
original UCI file):

1. Go to the official dataset page:
   https://archive.ics.uci.edu/dataset/222/bank+marketing
2. Download the dataset zip and extract `bank-full.csv`.
3. The original UCI file uses `;` as the separator and wraps text values in
   quotes. Either:
   - place it here as-is (the code in `src/preprocessing.py` auto-detects
     and handles the `;`-separated format), **or**
   - convert it to plain comma-separated first:
     ```bash
     sed 's/;/,/g; s/"//g' bank-full.csv > bank-full.csv.clean
     mv bank-full.csv.clean bank-full.csv
     ```

**Option C — use `ucimlrepo`** (if your environment has internet access):

```python
from ucimlrepo import fetch_ucirepo
bank_marketing = fetch_ucirepo(id=222)
X = bank_marketing.data.features
y = bank_marketing.data.targets
```

The notebook (`notebooks/bank_marketing_cnn.ipynb`) tries `ucimlrepo` first
and automatically falls back to this local CSV file if `ucimlrepo` is not
available or there is no internet connection.

## Note on redistribution

This project does not redistribute the raw UCI dataset outside of this
`data/` folder — it's included here purely as a convenience for running the
notebook and scripts offline. Please refer to the UCI Machine Learning
Repository's citation and usage terms if you redistribute this project.

Citation: S. Moro, P. Cortez and P. Rita. "A Data-Driven Approach to
Predict the Success of Bank Telemarketing." Decision Support Systems,
Elsevier, 62:22-31, June 2014.
