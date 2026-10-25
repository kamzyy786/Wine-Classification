# Chemical-aware Wine Transaction Anomaly Detection


This project combines a wine cultivar model with synthetic purchase records to screen for anomalous transactions. The original Milestone 1 remains a chemical classification and preprocessing study. This extension reuses that foundation for a laboratory-to-declared-label consistency check, then adds transaction screening. The transaction classifier combines that consistency signal with purchase behaviour.



## Setup and run
Requires Python 3.11 or 3.12. From this folder:

```bash
python -m venv .venv
```

Activate the environment (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on macOS/Linux), then:

```bash
python -m pip install -r requirements.txt
python process.py
```

No dataset download or API key is required. The unscaled source CSV is included in data/raw/wine.csv. To regenerate that CSV from the same scikit-learn version:

```bash
python -c "from sklearn.datasets import load_wine; load_wine(as_frame=True).frame.to_csv('data/raw/wine.csv', index=False)"
```

Custom relative paths are supported:

```bash
python process.py --raw data/raw/wine.csv --out outputs
```

## Outputs
outputs/ contains numeric_summary.csv, quality_checks.csv, summary.json, distributions.png, correlations.png, train_rows.csv, validation_rows.csv, test_rows.csv, preprocessing.json, processed_arrays.npz and processed_tensors.pt.

The PyTorch file is a dictionary of float32 feature tensors and int64 target tensors. Load with `torch.load('outputs/processed_tensors.pt', weights_only=True)`. Shapes are train 124x13, validation 27x13 and test 27x13. No classifier or test performance is included in this milestone.

The script uses all records only for descriptive reporting. Model transformations are fitted exclusively on the training partition. For future cross-validation, refit these transformations independently within every training fold, rather than reusing the saved globally fitted training arrays across folds.

## Files
- Milestone_1_Report.pdf: six-page report with student ID and repository URL fields to complete.
- report_editable.md: editable report text; charts are saved separately in outputs/.
- process.py: runnable checks, splitting, preprocessing and artifact generation.
- DATA_SOURCE.md: attribution, licence and source details.
- requirements.txt: dependencies; tested PyTorch version recorded below.

