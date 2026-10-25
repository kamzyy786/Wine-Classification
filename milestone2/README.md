# Wine Transaction Anomaly Detection

INSE 6450 | Kameshwar Arora | 40325656 |

The hybrid supervised/unsupervised design retains the supervised chemistry component and uses a PyTorch autoencoder as the primary unsupervised detector, alongside supervised transaction classifiers. All transaction events and scenario labels are synthetic. Chemical measurements are the unchanged UCI Wine observations from Milestone 1.

## Quick start

Use Python 3.12. From this folder:

```bash
python -m pip install -r requirements.txt
python verify.py
python demo.py --customer test-C0005 --event 8
```

The demo reconstructs that customer's earlier events, then scores the current event with the saved chemical and autoencoder models. The first five customer events produce `history_warmup`; subsequent events return reconstruction error, the frozen threshold, review decision and top three per-feature errors. It does not read anomaly/scenario labels to make decisions.

For an interactive demonstration open `Demo.ipynb`. To use Jupyter, install `python -m pip install jupyterlab ipykernel`, then run `python -m jupyter lab`. The notebook includes executed outputs.

## Data and partitions

- 1,000 independent synthetic customer histories, 30 raw events each: 30,000 raw events.
- Five history warm-up events per customer: 25,000 scored events.
- Training: 700 customers, 17,500 scored events; validation/test: 150 customers and 3,750 scored events each.
- Original wine split retained: 124/27/27 samples. Each customer partition uses only its own wine sample partition; no customers or wine samples cross sets.
- Sample IDs and customer IDs are linkage, never input predictors. More transactions do not increase the number of independent chemical observations beyond 178.

The generator changes raw unit price, quantity, timestamps/bursts, merchant category, latitude/longitude and declared cultivar. It then derives amount and quantity deviations, prior-hour velocity, elapsed time, merchant novelty and previous-location distance from strictly earlier customer events. History state changes after feature derivation. Two chemical consistency features complete the nine-input model.

Training uses three-fold out-of-fold chemical probabilities. Validation/test use the retained chemical checkpoint. Scaling is fitted on transaction training features only. The first five events are excluded by the same fixed warm-up rule for every customer, not by anomaly labels.

## Primary autoencoder and comparisons

The primary AE is 9-8-3-8-9 with ReLU hidden activations and linear outputs (220 parameters). It trains on all 17,500 scored training events without anomaly/scenario labels, including 553 simulated anomalies (3.16% contamination). It is not normal-only training.

Bottlenecks 2 and 3 are compared by validation reconstruction MSE; width 3 wins. The threshold is the predeclared 95th percentile of mixed validation scores: 4.424379. Model selection, stopping and thresholding do not use transaction anomaly labels. The threshold targets approximately 5% validation alert volume, not a 5% false-positive rate.

Transaction-only AE (seven inputs) is the chemistry ablation. Weighted/unweighted 9-16-2 combined MLPs, a 7-16-2 transaction-only MLP and a 9-2 linear classifier use the same dataset/splits and may use anomaly labels. The previous experiment's classifier scores are not reused as comparisons.

## New held-out results

Test set: 3,750 scored purchases, 280 synthetic anomalies and 3,470 normal purchases.

| Measure | AE + chemistry | Transaction-only AE | Weighted MLP |
| --- | ---: | ---: | ---: |
| Macro F1 | 0.673 | 0.655 | 0.884 |
| Anomaly precision | 0.474 | 0.429 | 0.944 |
| Anomaly recall | 0.329 | 0.300 | 0.668 |
| PR-AUC (average precision) | 0.441 | 0.459 | 0.907 |
| False alarms | 102 | 112 | 11 |
| Missed anomalies | 188 | 196 | 93 |

The AE is primary for the label-free fitting objective; supervised classification is the stronger operational choice on this labelled simulator. The combined weighted MLP has the highest average precision (0.907); the transaction-only MLP has the highest macro F1 (0.884235 versus 0.883808 for the combined weighted MLP). At the fixed threshold the combined AE detects most merchant/location cases and all combined cases, but only three velocity cases and no chemical-mismatch-only cases. Adding chemistry does not improve every metric. These results and threshold alternatives are reported without test-driven reselection.

## Explanations

For each input feature j, calculate `(standardized_input_j - reconstructed_j)^2`. The model score is the mean across the nine errors. Each flagged test transaction has its top three squared errors and their shares of total error saved in `results/flag_explanations.json`. They diagnose model reconstruction difficulty, not causal wrongdoing. Scores are not calibrated fraud probabilities.

## Reproduce

```bash
python experiment.py
python efficiency.py
python verify.py
python build_report.py
```

The experiment regenerates raw events, derived features, transaction models, histories, test predictions, threshold trade-offs and explanations. It reuses `models/chemistry.pt` and refits chemical OOF models for training features. Training overwrites generated outputs. Timing varies by host. Source code contains every generator rule and seed; `config.json` and `environment_versions.txt` record the executed run; config.json is a record, not the training script configuration loader.

## Files and evidence

| Path | Contents |
| --- | --- |
| `history.py` | Raw simulator and causal customer-state feature builder |
| `chemistry.py`, `models/chemistry.pt` | Retained chemical modelling helpers and trained state |
| `experiment.py`, `models.py` | Label-blind AE selection, labelled comparisons and model definitions |
| `demo.py`, `Demo.ipynb` | Working saved-model raw-event replay |
| `efficiency.py`, `verify.py` | Fresh-process benchmarks and leakage/replay checks |
| `data/events_*.csv`, `data/features_*.csv` | Raw customer histories and derived feature tables |
| `models/detectors.pt` | Selected AE, transaction-only AE and both MLPs plus scaling/thresholds |
| `results/metrics.json`, `results/efficiency.json` | Measured results and runtime evidence |
| `results/*_test_predictions.csv` | Model scores, decisions and held-out labels |
| `results/threshold_tradeoffs.csv` | Frozen validation-quantile thresholds and FP/FN consequences |
| `Milestone2.pdf` | New report with rubric coverage, failures, efficiencies and deployment design |
| `DEMO_GUIDE.md`, `PROJECT_STATE.json` | Presentation steps and continuation state |

## Implementation boundary

The local stateful CLI/notebook is working. A hosted API, shared durable customer state, laboratory integration, cache invalidation, review queues and shadow monitoring are deployment designs. Real application requires compatible laboratory-linked transactions and independently verified outcomes. The synthetic histories add behavioural structure but do not create real-world fraud evidence. Windows portability uses psutil for memory measurement; the recorded experiment was executed on Linux CPU.

## Complete supervised comparison

The transaction-only MLP and combined linear classifier were added on the same fixed histories and partitions. All AE weights, thresholds and prior predictions are preserved. Both new baselines use training labels, inverse-frequency loss weights and validation-selected thresholds. Test labels are used only after those choices are fixed.

| Model | Macro F1 | Anomaly precision | Anomaly recall | False alarms | Missed anomalies |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transaction-only MLP | 0.884 | 0.968 | 0.657 | 6 | 96 |
| Combined linear | 0.812 | 0.562 | 0.793 | 173 | 58 |
| Combined weighted MLP | 0.884 | 0.944 | 0.668 | 11 | 93 |
| Combined unweighted MLP | 0.876 | 0.888 | 0.679 | 24 | 90 |

The full reproduction command remains `python experiment.py`; it now trains all six final comparison models. `complete_comparisons.py` documents the measured supplemental run on the earlier frozen checkpoints and stops if those baseline states already exist.
