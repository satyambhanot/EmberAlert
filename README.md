# Spark to Blaze

**Predicting large wildfires in Alberta at initial assessment.**

ENSF 611 (Machine Learning for Software Engineers), Fall 2026, Group 12.

**Client:** Alberta Wildfire duty officers.
**Question:** When Alberta Wildfire first assesses a new wildfire, can we predict whether it will become a *large* fire (final size class D or E, more than 40 ha), so heavy resources can be committed early?

We compare two linear and four non-linear classifiers against simple baselines, using only information available at the initial assessment.

## Repository layout

```
├── wildfire_project.ipynb   # main notebook: run this
├── src/
│   └── features.py          # data loading, target, leakage-safe features, shared preprocessing
├── data/
│   ├── fp-historical-wildfire-data-2006-2025.csv             # dataset (included)
│   ├── fp-historical-wildfire-data-dictionary-2006-2025.pdf  # official column definitions
│   └── README.md            # source, license, checksum
├── figures/                 # plots saved by the notebook
├── docs/                    # proposal, report, slides
└── requirements.txt
```

## How to run

1. Install Python 3.9 or newer, then install the dependencies from the repository root:
   ```bash
   pip install -r requirements.txt
   ```
2. Start Jupyter **from the repository root**, so that `from src.features import ...` and the relative `data/` and `figures/` paths resolve:
   ```bash
   jupyter notebook wildfire_project.ipynb
   ```
3. Run all cells (*Kernel → Restart & Run All*). The dataset is already in `data/`, so nothing needs to be downloaded.

The code was tested with pandas 2.0–3.1 and scikit-learn 1.3–1.9.

## Data

Government of Alberta, Forestry and Parks. *Historical wildfire data: 2006 to 2025*, published under the Open Government Licence – Alberta.
https://open.alberta.ca/opendata/a221e7a0-4f46-4be7-9c5a-e29de9a3447e

- 27,828 fires × 50 columns; 932 large fires (3.3%).
- Columns recorded after the initial assessment (suppression actions, status updates, investigated causes) are excluded to prevent data leakage. `LEAKAGE_COLUMNS` in `src/features.py` lists each one with the reason.

## Models

| Type | Models |
|---|---|
| Baselines | `DummyClassifier`; `LogisticRegression` on size at assessment only |
| Linear | `LogisticRegression`, `LinearSVC` |
| Non-linear | `RandomForestClassifier`, `GradientBoostingClassifier`, `SVC` (RBF kernel), `KNeighborsClassifier` |

## References

- Government of Alberta (2026). *Historical wildfire data: 2006 to 2025* and *Historical Wildfire Data Dictionary*. Open Government Licence – Alberta.
- Müller, A. C. & Guido, S. (2016). *Introduction to Machine Learning with Python*. O'Reilly.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR* 12, 2825–2830.
