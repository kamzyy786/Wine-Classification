# Demonstration guide: history-based implementation

Kameshwar Arora | 40325656 | INSE 6450

## Preparation

From the new `milestone2` folder, install requirements and run:

```bash
python verify.py
python demo.py --customer test-C0005 --event 8
```

Verification must print PASS. The demo shows the merchant-anomaly example from the new report. It replays five warm-up events and the remaining prefix before scoring event 8. The actual output contains the score, frozen threshold, human-review flag, chemical prediction and three reconstruction-error factors.

For Jupyter: `python -m pip install jupyterlab ipykernel`, then `python -m jupyter lab`. Open `Demo.ipynb` with the same Python environment and run cells in order. Recorded outputs are already included. The demo uses saved checkpoints, so it does not need internet or retraining once dependencies are installed.

## Presentation sequence

1. Explain the two-stage system: supervised cultivar classification from chemical measurements, followed by an unsupervised transaction autoencoder. Milestone 1's original data and partitions are unchanged.
2. Show raw events with customer ID, timestamp, price, quantity, merchant, coordinates, linked sample and declaration. The generator changes these raw values; it does not directly fill in a ready-made recent-purchase count.
3. Explain past-only state: the current event is compared with earlier customer events, and state changes after the features are derived. The five-event warm-up is identical for every customer.
4. Run the merchant example. Show reconstruction error 5.181547 versus threshold 4.424379. Merchant novelty contributes about 90.1% of that record's total squared reconstruction error.
5. Show the location example `python demo.py --customer test-C0010 --event 6`. Location shift dominates its reconstruction error.
6. Show the results table and per-scenario recall. The AE has 102 false alarms and 188 missed anomalies on the new test set. The weighted supervised MLP is better on this labelled simulator. The AE is primary for the label-free training objective.
7. Show the 90/95/97.5/99 validation-percentile threshold trade-offs. The reported operating point stays at the predeclared 95th percentile; test labels do not select it.
8. Show the deployment diagram and measured CPU timing. The local replay works; a hosted API, durable shared state, laboratory acquisition and queue infrastructure are planned components.

## Questions and accurate answers

**Does the AE train only on normal records?** No. All scored training records are included, with 3.16% synthetic contamination. Anomaly labels are not passed to AE loss, stopping, capacity selection or threshold calibration. Generator labels are used after freezing for evaluation and for supervised comparisons.

**Is the chemical classifier unsupervised?** No. It remains supervised multiclass classification, consistent with Milestone 1. Its predicted consistency signals are inputs to the unlabelled transaction AE.

**Does chemistry improve every result?** No. It slightly improves macro F1 at the chosen threshold compared with transaction-only AE, but lowers ranking metrics. The primary AE misses all chemical-mismatch-only test cases at its frozen threshold. The report exposes that failure.

**Why are supervised scores higher?** The comparator receives the synthetic anomaly labels during fitting. The comparison quantifies the advantage of having labels on this generator. Its architecture is retained from the earlier idea, but its weights are retrained on the new histories; old scores are not reused.

**How are explanations made?** Rank squared standardized input-minus-reconstruction differences. Their sum divided by nine equals the anomaly score. Error shares are a decomposition of model error, not a probability of fraud or a causal attribution.

**Are there 30,000 independent chemical samples?** No. There are 30,000 synthetic events, 1,000 customer histories and the same 178 historical chemical samples. Wine samples and customers are separated between partitions.

## Continue implementation

Persist customer event state and sample links, wrap the existing `Screener.predict_event` in a Python API or interface, add idempotent event IDs and late-event replay, implement version-keyed chemical caching and human-review queues, then benchmark the full deployed request path. Monitor flag rate, drift, latency and analyst outcomes. New modelling changes require a new development/holdout protocol rather than repeated tuning on the already reported test set.

## Additional comparison evidence

The current checkpoint includes transaction-only MLP and combined linear supervised baselines in addition to the two AEs and combined MLP weighting comparison. All models use the same held-out rows. The linear model has higher anomaly recall but more false alarms than the combined MLP. Show all six models in the notebook; retain the original frozen AE operating point. Full per-scenario results are in results/metrics.json.
