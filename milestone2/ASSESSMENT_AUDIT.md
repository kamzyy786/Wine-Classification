# Assessment implementation audit

This audit applies to the supplied Current Repository Assessment and the current implementation, version 3.1. The document describes the earlier supervised-only study in several places; the current package already includes the later history-based AE work. The two missing supervised baseline types have now been added on the same frozen partitions.

| Assessment item | Implemented evidence | Verification |
| --- | --- | --- |
| Preserve Milestone 1 | No root Milestone 1 files are replaced. Raw wine CSV, original sample assignments and chemical weights remain unchanged. | Original CSV/chemical state byte checks; exact split checks. |
| Add unsupervised PyTorch AE | `models.py`, `experiment.py`: combined 9-8-3-8-9 AE and transaction-only AE. | Saved states reproduce scores. Transaction anomaly labels do not enter AE fitting, stopping, capacity choice or threshold. |
| Frozen label-free operating rule | Validation reconstruction-score p95, fixed before test evaluation. | Recomputed quantile agrees with saved thresholds; prior AE choices remain unchanged. |
| Retain supervised comparisons | Combined MLP, transaction-only MLP, combined linear model and unweighted MLP comparison. | All six model states, predictions and measured metrics are present; supervised thresholds use validation labels only. |
| Genuine history features | 30,000 raw events; 1,000 customer histories; actual prior-hour counts, elapsed time, novelty, amount deviation and location distance. | All 25,000 feature rows rebuilt independently from raw events. Future-event perturbation leaves earlier features unchanged. |
| Prevent entity leakage | Customers and wine samples are separated across partitions; training chemical probabilities are out of fold. | Exact 124/27/27 wine splits and disjoint customer/sample checks. |
| Explanations | Three largest per-feature squared standardized reconstruction errors for each AE flag. | Flag-ID set agrees exactly with explanation records; worked examples in PDF and notebook. |
| Metrics and comparisons | Validation/test accuracy, anomaly and macro precision/recall, F1, AUROC, average precision, FP/FN and scenario recall. | Every saved test metric recomputed for all six models. Full scenario CSV included. |
| Curves and imbalance | Train/validation curves; weighted/unweighted MLP comparison; no target weighting or label-based record removal for AE. | History CSVs and observed weighting effects reported. AE contamination remains explicit at 3.16%. |
| Efficiency | Per-model training/epoch time, parameters, dense FLOPs estimates, process memory and fresh CPU latency/throughput. | `efficiency.py` benchmarks every exported state at batches 1/32; full event replay measured separately. |
| Deployment | Offline/online, hybrid edge/cloud, streaming/batch, diagram, cache/queue, fallback/rollback, SLO and shadow-monitoring design. | Report distinguishes the working local prototype from planned service infrastructure. |
| Documentation and naming | Root README and `milestone2` guide point to `Milestone2.pdf`; no obsolete folder naming. | Relative-link and naming checks; original result sets retained without performance inflation. |

## Important interpretation

The AE trains on mixed, unlabelled transaction data with known simulation contamination; it is not trained on a normal-only subset selected using anomaly labels. All simulated anomaly labels remain available for evaluation and supervised comparisons. The supervised models outperform the AE on several measures; the AE remains primary for the label-free training objective. Neither a flag nor a synthetic scenario proves real fraud.

## External completion items

Upload the prepared revision, confirm instructor/POD access and submit the current PDF on Moodle. These account-level actions are not performed or independently verified by this package. The working CLI/notebook is supplied; a hosted production service is not claimed.
