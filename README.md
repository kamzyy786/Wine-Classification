# Chemical-aware Wine Transaction Anomaly Detection

INSE 6450: AI in Systems Engineering, Fall 2026  
Kameshwar Arora | Student ID: 40325656

This prototype screens wine purchase transactions using purchase behaviour and a linked laboratory sample. It produces an anomaly risk score, a review decision and local explanation factors. The operational goal is to detect suspicious transactions while making the cost of false alarms and missed anomalies visible.

**The chemical observations come from UCI Wine. All purchase records, purchase-to-sample links and anomaly labels are synthetic. A flag requests human review; it is not proof of fraud, product authenticity or food safety.**

## Run the transaction screener

Use Python 3.12. From the repository root:

```bash
cd milestone2
python -m pip install -r requirements.txt
python verify.py
python demo.py --row 0
```

The demo loads the included checkpoint; no retraining, API key or external service is needed. `verify.py` checks sample separation, preserved splits, checkpoint-score agreement, invalid-input rejection and metric recomputation.

For the notebook demonstration, open [Demo.ipynb](milestone2/Demo.ipynb) in VS Code or Jupyter. Saved outputs are included. For Jupyter:

```bash
python -m pip install jupyterlab ipykernel
python -m jupyter lab
```

Follow the [demonstration guide](milestone2/DEMO_GUIDE.md) for examples and presentation steps.

## System and inputs

1. Validate a purchase record and its linked laboratory sample.
2. Predict cultivar probabilities from the sample's 13 chemical measurements.
3. Compare the predicted cultivar with the purchase's declared cultivar.
4. Combine that consistency signal with unit price, quantity, recent purchase frequency and a new-merchant-category indicator.
5. Return a binary screening decision, an uncalibrated risk score and the three strongest reported local explanation factors.

The main chemical model is a small 13-16-3 PyTorch network. The selected risk model is a 6-16-2 PyTorch network, with a validation-selected threshold of 0.45. Neither sample identifiers nor true cultivar labels are inputs to the transaction classifier. Because synthetic scenarios provide labels, anomaly screening is trained as supervised binary classification.

## Held-out synthetic test results

The test partition contains 540 purchases generated from 27 held-out wine samples: 100 labelled anomalies and 440 normal purchases. Purchases linked to the same sample remain in one partition.

| Metric | Transaction-only model | Selected combined model |
| --- | ---: | ---: |
| Accuracy | 92.22% | 94.07% |
| Macro F1 | 0.846 | 0.909 |
| Anomaly precision | 98.33% | 78.33% |
| Anomaly recall | 59.00% | 94.00% |
| Detected anomalies | 59 | 94 |
| Missed anomalies | 41 | 6 |
| False alarms | 1 | 26 |

Adding chemistry catches more simulated declaration inconsistencies but increases review workload. The gain depends on the generator's scenarios. The sample-cluster bootstrap interval for the macro F1 difference includes zero; these results do not establish superiority on real transactions.

## Inspect the report and evidence

- [Milestone 2 report](milestone2/Milestone_2_Report.pdf): architecture, experiments, efficiency, deployment design, rubric audit and worked explanation examples.
- [Measured metrics](milestone2/results/metrics.json): validation/test scores, confusion matrices, training times and environment.
- [Learning curves and predictions](milestone2/results/): PNG plots, epoch histories and test predictions.
- [Transaction explanations](milestone2/results/test_explanations.json): explanations for every test purchase.
- [Serving efficiency audit](milestone2/results/efficiency_audit.json): fresh-process memory, demo timing and analytical training-compute estimates.
- [Model checkpoint](milestone2/models/screening.pt): weights and preprocessing parameters.

Explanations replace a feature group with its **overall training means** and measure the score reduction. They are local sensitivity diagnostics, not causal attributions or a normal-only reference. Negative score reductions are retained and explained.

## Reproduce the experiments

From `milestone2/`:

```bash
python experiment.py
python explain_results.py
python efficiency_audit.py
python verify.py
python build_report.py
```

These commands overwrite generated results. Seeds and hyperparameters are recorded in [config.json](milestone2/config.json); recorded dependency versions are in [environment_versions.txt](milestone2/environment_versions.txt). Timings vary by hardware. Model selection and thresholds use validation data; the reported test partition is held out from training and selection.

## Milestone 1: chemical profiling foundation

The root `process.py` prepares the original UCI Wine measurements. It performs data checks, creates the fixed sample splits, fits training-only preprocessing and saves arrays, tensors and descriptive plots. That stage contains no trained classifier. Trained chemical and risk models are provided under `milestone2/`.

To reproduce the original preprocessing from the repository root:

```bash
python -m pip install -r requirements.txt
python process.py --raw data/raw/wine.csv --out outputs
```

`outputs/` contains the original summary statistics, quality checks, distributions, correlations, preprocessing parameters and sample assignments. The 17 descriptive IQR flags are chemical review flags, not confirmed fraudulent purchases. The raw dataset and original split assignments are preserved.

The chemical profiling stage provides the foundation for the later laboratory-to-declaration consistency check. Transaction generation, risk modelling and evaluation are implemented in Milestone 2.

## Scope and further development

The local checkpoint demo and notebook are implemented. The hybrid serving architecture, hosted API, shared transaction history, caching, asynchronous review queue and rollback workflow are deployment designs. A hosted production service has not been implemented.

Recent-purchase and merchant-category indicators are generated directly by the simulator rather than reconstructed from a real event history. Real-world use requires compatible purchase-linked laboratory observations, verified outcomes, calibration and external evaluation. See [DEMO_GUIDE.md](milestone2/DEMO_GUIDE.md) for the continuation plan.

## Data attribution

The historical chemical measurements are the UCI Wine dataset, licensed under CC BY 4.0. See the root [DATA_SOURCE.md](DATA_SOURCE.md) and [Milestone 2 data notes](milestone2/DATA_SOURCE.md). Synthetic purchases are generated by `milestone2/experiment.py` and are documented in `milestone2/README.md`.
