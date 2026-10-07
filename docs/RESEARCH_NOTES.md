# Research notes

Sources consulted for the redesign, checked on 2026-10-07 by reading the cited page or running code. Items marked **(partly verified)** or **UNVERIFIED** could not be fully confirmed at the primary source.

## Dataset

| Source | Relevance | Influenced |
|---|---|---|
| UCI Bank Marketing, dataset 222 — https://archive.ics.uci.edu/dataset/222/bank+marketing | Variable definitions, `bank-full` ordered by date (May 2008 – Nov 2010), licence CC BY 4.0 | [DATASET.md](DATASET.md); out-of-time evaluation |
| `bank-names.txt` and `bank-additional-names.txt` in the official archive — https://archive.ics.uci.edu/static/public/222/bank+marketing.zip | Exact field descriptions. The `duration` "benchmark only" note is attribute 11 of `bank-additional-names.txt`. `day` is day of month (the UCI web table mislabels it `day_of_week`) | Exclusion of `duration` |
| Moro, Cortez & Rita (2014), *A data-driven approach to predict the success of bank telemarketing*, Decision Support Systems 62:22–31, doi:10.1016/j.dss.2014.03.001 (preprint: https://repositorium.uminho.pt/handle/1822/30994) | 150 → 22 features (incl. euribor, agent experience, call direction, not in `bank-full`); rolling-window AUCs LR 0.715, DT 0.757, SVM 0.767, NN 0.794; excluded call duration as unusable for prediction | Don't treat their AUC as a directly comparable benchmark; supports dropping `duration` |
| Dataset DOI 10.24432/C5K306; `bank-full` readme asks to cite Moro, Laureano & Cortez (2011), ESM'2011 pp. 117–121 | Citation hygiene | [DATASET.md](DATASET.md) |

## Machine learning

| Source | One-line relevance | Influenced |
|---|---|---|
| Grinsztajn, Oyallon & Varoquaux (2022), *Why do tree-based models still outperform deep learning on typical tabular data?*, NeurIPS Datasets & Benchmarks, arXiv:2207.08815 | Tree ensembles remain state of the art on medium-sized tabular data | Expectation for the CNN/MLP vs boosting comparison |
| Shwartz-Ziv & Armon (2022), *Tabular data: Deep learning is not all you need*, Information Fusion 81:84–90, doi:10.1016/j.inffus.2021.11.011 | XGBoost outperforms deep tabular models and needs less tuning | Same |
| Niculescu-Mizil & Caruana (2005), *Predicting good probabilities with supervised learning*, ICML, pp. 625–632, doi:10.1145/1102351.1102430 | Boosted trees can distort probabilities; calibration methods | Calibration was measured; none was needed (ECE 0.86 %) |
| Guo, Pleiss, Sun & Weinberger (2017), *On Calibration of Modern Neural Networks*, ICML, PMLR 70:1321–1330 | Modern neural nets are often miscalibrated | Calibration reported for every candidate |
| van den Goorbergh, van Smeden, Timmerman & Van Calster (2022), *The harm of class imbalance corrections for risk prediction models*, JAMIA 29(9):1525–1534, doi:10.1093/jamia/ocac093 | Imbalance corrections worsen calibration without improving discrimination | Unweighted training; balanced-weight experiment reproduces the effect |
| Elkan (2001), *The Foundations of Cost-Sensitive Learning*, IJCAI-01, pp. 973–978 — https://mlanthology.org/ijcai/2001/elkan2001ijcai-foundations **(partly verified: venue/pages from an index listing)** | Optimal decisions compare calibrated probability with a cost-derived threshold | Break-even = cost / value in the policy |
| Kaufman, Rosset, Perlich & Stitelman (2012), *Leakage in data mining: Formulation, detection, and avoidance*, ACM TKDD 6(4):15, doi:10.1145/2382577.2382579 | Definition of leakage; learn–predict separation | [DATA_LEAKAGE_AND_FEATURE_AVAILABILITY.md](DATA_LEAKAGE_AND_FEATURE_AVAILABILITY.md) |
| Saito & Rehmsmeier (2015), *The Precision-Recall Plot Is More Informative than the ROC Plot…*, PLoS ONE 10(3):e0118432, doi:10.1371/journal.pone.0118432 | PR curves for imbalanced problems | PR-AUC reported alongside ROC-AUC |
| Ke et al. (2017), *LightGBM*, NIPS 30, pp. 3149–3157 **(partly verified: venue/pages from search results)** | Candidate boosting library | Considered; rejected for deployment (needs `brew install libomp` on macOS) |
| Fisher, Rudin & Dominici (2019), *All Models are Wrong, but Many are Useful…*, JMLR 20(177):1–81 | Permutation-based model reliance | Background for the CNN column-permutation test (a different use: permuting *input order*, not values) |

## Explainability

| Source | Relevance | Influenced |
|---|---|---|
| Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions*, NeurIPS, arXiv:1705.07874 (proceedings page numbers **UNVERIFIED**) | SHAP: additive, game-theoretic attributions | Local explanations, odds-multiplier reading |
| Lundberg et al. (2020), *From local explanations to global understanding with explainable AI for trees*, Nature Machine Intelligence 2(1):56–67, doi:10.1038/s42256-019-0138-9 | Exact polynomial-time TreeSHAP, interaction values | Choice of TreeSHAP; interaction analysis |
| Molnar, *Interpretable Machine Learning*, 3rd ed. (2025) — https://christophm.github.io/interpretable-ml-book/ | Practical guidance and limits (correlation, causality) | Limitations section of [EXPLAINABILITY.md](EXPLAINABILITY.md) |
| SHAP `TreeExplainer` docs — https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html | Supported models | Combined with our own additivity test, which found native-categorical HistGB attributions wrong (2.68 log-odds) |

## Engineering

| Source | Relevance | Influenced |
|---|---|---|
| Streamlit docs: execution model (https://docs.streamlit.io/get-started/fundamentals/main-concepts), caching (https://docs.streamlit.io/develop/concepts/architecture/caching), fragments (https://docs.streamlit.io/develop/concepts/architecture/fragments) | Whole-script rerun on every interaction; `cache_resource` thread-safety caveats; fragments since 1.37 | Architecture decision (see ARCHITECTURE_DECISION.md) |
| scikit-learn 1.6 release notes — https://scikit-learn.org/stable/whats_new/v1.6.html | `cv="prefit"` deprecated in favour of `FrozenEstimator` (removed by 1.9.1 in our test) | Any future calibration step must use `FrozenEstimator` |
| scikit-learn `HistGradientBoostingClassifier` docs | `categorical_features="from_dtype"` default since 1.6 | We deliberately don't use native categoricals (SHAP bug above) |
| LightGBM docs (`pred_contrib`, `deterministic`) and PyPI page | `pred_contrib` returns TreeSHAP in log-odds; macOS needs libomp | Rejected for portability |
| Tailwind Play CDN docs — https://tailwindcss.com/docs/installation/play-cdn | "designed for development purposes only" | Hand-written CSS instead |
| HTMX releases — https://github.com/bigskysoftware/htmx/releases | 2.0.x is the stable npm `latest` line (4.0 tagged `next`) | Vendored htmx 2.0.10 |
| TensorFlow install page — https://www.tensorflow.org/install/pip; PyPI `tensorflow-cpu` files | `tensorflow-cpu` has no macOS wheels (only manylinux x86_64, win_amd64) | Explains why the legacy requirements failed to install on Apple Silicon; TF moved to an optional `dl` group |
| Railway docs — https://docs.railway.com/guides/dockerfiles, https://docs.railway.com/reference/pricing/free-trial | Dockerfile auto-detection, `PORT` env var, trial credits | Deployment approach (see DEPLOYMENT.md) |

## Library versions in the locked environment

From `uv pip list` in the project `.venv` (2026-10-07):

| Package | Version | Package | Version |
|---|---|---|---|
| Python | 3.12.13 | fastapi | 0.142.2 |
| scikit-learn | 1.9.1 | uvicorn | 0.54.0 |
| shap | 0.52.0 | jinja2 | 3.1.6 |
| numba (via shap) | 0.68.0 | pydantic | 2.13.5 |
| numpy | 2.5.3 | pydantic-settings | 2.15.0 |
| pandas | 3.0.6 | loguru | 0.7.3 |
| threadpoolctl | 3.7.0 | httpx | 0.28.1 |
| pytest | 9.1.1 | ruff | 0.16.10 |
| pyright | 1.1.414 | htmx (vendored JS) | 2.0.10 |

TensorFlow 2.21.0 is installed only with `uv sync --group dl` for the CNN study.
