# Data

`bank-full.csv` is the **UCI Bank Marketing** dataset (id 222), `bank-full` variant:
45,211 rows, 16 input columns + target `y`, direct-marketing phone campaigns of a
Portuguese bank, **ordered by date from May 2008 to November 2010** (no year column).
The committed copy was converted from UCI's `;`-separated, quoted format to plain CSV;
`leadintel.data.load_dataset` accepts either format and validates the schema and the
documented invariants on every load. See [docs/DATASET.md](../docs/DATASET.md).

* Source: https://archive.ics.uci.edu/dataset/222/bank+marketing
* Licence: CC BY 4.0 (redistribution with attribution is permitted)
* Dataset citation: Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset].
  UCI Machine Learning Repository. https://doi.org/10.24432/C5K306
* Paper: Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the
  success of bank telemarketing. *Decision Support Systems*, 62, 22–31.
  https://doi.org/10.1016/j.dss.2014.03.001

To re-download, extract `bank-full.csv` from the official zip into this folder; no
conversion is needed.
