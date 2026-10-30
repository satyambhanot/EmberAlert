# Data

| File | Description |
|---|---|
| `fp-historical-wildfire-data-2006-2025.csv` | One row per wildfire in Alberta's Forest Protection Area, 2006–2025 (27,828 rows, 50 columns, UTF-8). |
| `fp-historical-wildfire-data-dictionary-2006-2025.pdf` | Official definition of every column, including when each value is recorded. |

## Source

Government of Alberta, Forestry and Parks. *Historical wildfire data: 2006 to 2025.*
https://open.alberta.ca/opendata/a221e7a0-4f46-4be7-9c5a-e29de9a3447e

- Downloaded: 2026-10-30
- SHA-256 of the CSV: `8b6ed447ab0f958dbe845a53dc15362b8ef5c0810ad9420d2143f9f5ee010a72`

## License

Published under the [Open Government Licence – Alberta](https://open.alberta.ca/licence), which permits copying, redistribution and adaptation as long as the source is attributed. The attribution above satisfies this.

## Notes

- `SIZE_CLASS` gives the final size of each fire: A ≤ 0.1 ha, B ≤ 4 ha, C ≤ 40 ha, D ≤ 200 ha, E > 200 ha.
- Many columns are recorded *after* the initial assessment, such as suppression actions, fire status dates and investigated causes. The project drops these to prevent data leakage. `src/features.py` lists every dropped column with the reason.
