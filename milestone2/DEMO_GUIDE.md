# Professor demonstration and continuation plan

Kameshwar Arora, student ID 40325656. INSE 6450.

## Current working prototype

The package contains a trained chemical classifier, trained transaction screener, preprocessing, saved checkpoints, an executed notebook, transaction explanations and reproducible experiment code. Transactions and their anomaly labels are synthetic; the historical chemical measurements are the UCI Wine data used in Milestone 1. The deployment architecture is a design, not an already hosted service.

## Prepare the demonstration

1. Extract the ZIP and open the milestone2 folder.
2. Use Python 3.12 and install requirements with `python -m pip install -r requirements.txt`.
3. Run `python verify.py`. It must print PASS.
4. Run `python demo.py --row 0` to confirm saved-model inference works without retraining or internet access.
5. Open Demo.ipynb with Jupyter or VS Code and select the same Python environment. Saved outputs are already present; run the four cells in order for a live demonstration. A notebook viewer can show saved outputs even without a kernel. For Jupyter, install `python -m pip install jupyterlab ipykernel` and run `python -m jupyter lab`.

## Five-minute presentation

First explain the problem: a purchase may have unusual behaviour, or its declared cultivar may disagree with a linked laboratory sample. The output is a review flag, not a legal fraud conclusion.

Show the Milestone 1 foundation: 13 chemical features, quality checks, training-only scaling, fixed sample splits and the existing recorded-versus-predicted cultivar use case. Milestone 2 uses those chemical predictions to add a consistency signal to synthetic purchase records. Transaction-only screening does not require chemistry; this particular laboratory consistency branch does.

Run the notebook's examples from each scenario. Point out the risk score, 0.45 decision threshold, predicted versus declared class and the three local explanation factors. Explain that the factors use grouped mean occlusion and are not causal explanations.

Change the declared cultivar for the same sample in the editable cell. Keep the purchase fields fixed and compare the scores. Then increase recent purchase frequency and quantity to show the transaction branch. The model is probabilistic: a changed field need not always flip a decision.

Show the comparison table: combined test macro F1 is 0.909 versus 0.846 for transaction-only screening. Anomaly recall is 0.940 versus 0.590. The combined system also flags 26 normal purchases versus 1, so the chemical signal increases review workload. Explain that the bootstrap interval includes zero and real-world superiority has not been established.

Close with the deployment diagram. Training is offline; screening happens after required inputs arrive; laboratory acquisition is outside the millisecond timing. The proposed hybrid service uses cached chemical predictions, review queues and versioned rollback. These infrastructure components remain to be implemented.

## Reproduce the report

Run, in order:

```text
python experiment.py
python explain_results.py
python efficiency_audit.py
python verify.py
python build_report.py
```

Training regenerates data and results. Timings vary by host. The source CSV, seeds and partitions remain fixed. The PDF is built from recorded outputs; add no invented observations. Font files and their licence are bundled for portable PDF generation. Linux memory measurements use `/proc`; other systems fall back to psutil, which is included in requirements.txt. The recorded results are from Linux; Windows results and timings must be measured separately. Windows portability was addressed in code but not tested on a Windows host.

## Continue toward a fuller application

1. Stabilize the input contract: 13 chemical features in recorded order, declared class 0/1/2, unit price, quantity, recent purchase count and new-category indicator. Retain sample/transaction IDs as linkage, never learned fraud predictors.
2. Add a Python interface or API around Screener.predict, with explicit missing-laboratory and invalid-input responses. Keep the checkpoint, preprocessing and threshold versioned together.
3. Add a local database for synthetic purchases, sample links, review decisions and audit versions. Compute frequency features from past events only; never use future transactions.
4. Implement version-keyed chemical caching and an asynchronous human-review queue. Add an explicitly marked transaction-only fallback, rather than inventing missing chemical values.
5. Load-test the full request path, including validation, serialization and explanations. Monitor p95 latency, flag rate, missing inputs and queue delay.
6. Before making operational fraud claims, obtain compatible purchase-linked laboratory records and independently verified outcomes. Evaluate calibration, external generalization and review costs. Do not tune on the already reported test set; collect a fresh holdout for new model development.

## Before submission

Upload the milestone2 folder as a new commit to the existing Wine-Classification repository. Keep the original Milestone 1 report and history. Share repository access with the POD, and submit the updated PDF on Moodle. This package does not perform either external action. Topic approval and marks remain the instructor's decision.
