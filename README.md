# Wine Transaction Anomaly Detection

INSE 6450 | Kameshwar Arora | Student ID 40325656

The current Milestone 2 study uses a supervised wine chemistry component followed by a small **unsupervised PyTorch autoencoder** for transaction screening. It generates raw customer purchase histories, derives features from past events only, and reports reconstruction-based flags with the three largest per-feature errors.

The chemistry observations are the unchanged UCI Wine dataset from Milestone 1. All customers, purchase events, coordinates and transaction anomaly labels are synthetic. An anomaly flag requests human review; it is not proof of fraud, authenticity or safety.

## Run the designed system

Use Python 3.12. From the repository root:

```bash
cd milestone2
python -m pip install -r requirements.txt
python verify.py
python demo.py --customer test-C0005 --event 8
```

- [Redesigned report](milestone2/Milestone_2_Report.pdf)
- [Executed notebook demo](milestone2/Demo.ipynb)
- [Setup, methods and results](milestone2/README.md)
- [Professor demonstration guide](milestone2/DEMO_GUIDE.md)
- [Measured metrics](milestone2/results/metrics.json)
- [Threshold trade-offs](milestone2/results/threshold_tradeoffs.csv)

## New study

The generator creates 30,000 raw events across 1,000 customers. Five initial events per customer establish history, leaving 25,000 scored events. Customer IDs and linked wine sample groups are disjoint across train/validation/test. History features are calculated before updating customer state.

The primary AE trains without anomaly/scenario labels, including the naturally contaminated simulated training records. Its 9-8-3-8-9 architecture is selected using validation reconstruction loss; a predeclared validation-score 95th percentile fixes the threshold. A transaction-only AE tests the chemical component. Weighted and unweighted supervised MLPs are retrained using exactly the same data partitions.

| New test result | AE + chemistry | Transaction-only AE | Weighted supervised MLP |
| --- | ---: | ---: | ---: |
| Test transactions / anomalies | 3,750 / 280 | Same | Same |
| Macro F1 | 0.673 | 0.655 | 0.884 |
| Anomaly precision | 0.474 | 0.429 | 0.944 |
| Anomaly recall | 0.329 | 0.300 | 0.668 |
| False alarms / missed anomalies | 102 / 188 | 112 / 196 | 11 / 93 |

The autoencoder is primary because it fits without transaction anomaly labels, not because it outperforms the labelled comparator. Scenario-specific failures, unlabelled training contamination and the threshold's review-volume trade-off are reported. The old experiment used different data, so its scores are not used as direct comparisons.

## Milestone 1 foundation

The existing root `process.py`, `data/raw/wine.csv`, `outputs/` and root dependency file retain the chemical preprocessing work. That stage prepares 178 chemical samples with the original 124/27/27 split. The retained chemical classifier adds a sample-to-declared-cultivar consistency signal in the redesigned study. More transactions do not create more independent chemical samples.

To regenerate the original preprocessing, run `python process.py` from the root after installing the root requirements. Its 17 exploratory IQR flags are chemical review flags, not verified fraud.


## Reproduce and continue

From `milestone2/`, run `python experiment.py`, `python efficiency.py`, `python verify.py`, and `python build_report.py`. Results include raw-event histories, features, all model predictions, learning curves, every flagged explanation, runtime measurements and configuration.

The local event-replay CLI and notebook are implemented. Hosted API, durable shared state, laboratory integration, caching and review-queue services remain documented deployment designs. The detailed report distinguishes measured behaviour from these planned components.

Data attribution: [original source notes](DATA_SOURCE.md) and [redesigned data notes](milestone2/DATA_SOURCE.md).
