# Chemical-aware Wine Purchase Transaction Screening

INSE 6450, Fall 2026. Kameshwar Arora. Student ID: 40325656. Milestone 2 extension.

This project combines a wine cultivar model with synthetic purchase records to screen for anomalous transactions. The original Milestone 1 remains a chemical classification and preprocessing study. This extension reuses that foundation for a laboratory-to-declared-label consistency check, then adds transaction screening. The transaction classifier combines that consistency signal with purchase behaviour.

## Run

Use Python 3.12. Install packages with `python -m pip install -r requirements.txt`.

1. `python experiment.py` regenerates data, trains models, freezes choices on validation, evaluates test performance, and writes results and checkpoints. CPU run; no API keys needed.
2. `python explain_results.py` writes explanations for every test transaction and the sample-cluster bootstrap comparison.
3. `python demo.py --row 0` loads the checkpoint and screens one synthetic test purchase.
4. Open `Demo.ipynb` for the short interactive notebook demonstration.
5. `python verify.py` checks sample separation, prediction alignment, checkpoint inference, and invalid-input handling.
6. `python efficiency_audit.py` measures a fresh serving process and estimates dense training compute.
7. `python build_report.py` rebuilds the report from saved measurements.

Run commands from this folder. Training overwrites generated outputs. Timings vary by hardware. Historical measured results are in results/metrics.json; config.json records the run configuration.

## Data and interpretation

data/wine.csv is the unchanged Milestone 1 CSV, derived from UCI Wine (CC BY 4.0); see DATA_SOURCE.md. The 178 wine samples are real historical chemical observations. All purchase records, transaction anomaly labels, merchant indicators, declared-label perturbations, and sample links are synthetic. Dataset classes are encoded 0/1/2; they are not named commercial cultivars. Prices are arbitrary simulation units, not market estimates.

Each sample creates 20 purchases. A latent scenario is drawn: normal 80%, label mismatch 8%, financial anomaly 8%, combined 4%. Label mismatch scenarios change the declared class with probability 0.92; other scenarios have incidental declaration changes with probability 0.02. Financial scenarios probabilistically perturb price and quantity and increase purchase frequency. The target is the sampled scenario, not a threshold calculated from model inputs. Some scenarios are intentionally ambiguous. See generate() for the exact rules.

The original seed-42 split (124/27/27 wine samples) is preserved. Purchase records are generated separately inside those partitions. Training uses out-of-fold chemical probabilities; held-out fold labels do not control the fold model's stopping. No true cultivar, sample ID, scenario, transaction ID, or anomaly label is an input to the risk classifier. True cultivar is used only for chemical training and synthetic-data construction. At inference the declared cultivar comes from the purchase record, and chemical measurements must come from a linked laboratory sample.

This evaluates performance on the specified simulator, not real fraud detection. Repeated transactions do not create additional independent chemical samples. Sample selection and simulation assumptions can dominate the results. No safety, authenticity, adulteration, or fraud claim follows from a disagreement.

## Outputs

- results/metrics.json: measured metrics, confusion matrices, scenario flag rates, environment, timing and memory.
- results/*_learning.csv/png: epoch-level training/validation losses and validation macro F1 at argmax.
- results/*_test_predictions.csv: auditable test scores and decisions.
- models/screening.pt: CPU checkpoint and preprocessing parameters.
- Milestone_2_Report.pdf: generated report with student ID 40325656.

Model probability scores are uncalibrated. The demo reports grouped mean-occlusion score changes, label disagreement and standardized deviations. These are local model diagnostics, not causal or SHAP attributions. A human reviews flags.

