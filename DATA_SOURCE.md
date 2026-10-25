# Data provenance and attribution
Aeberhard, S. & Forina, M. (1992). Wine [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5PC7J
Dataset page: https://archive.ics.uci.edu/dataset/109/wine
License: Creative Commons Attribution 4.0 International (CC BY 4.0).
https://creativecommons.org/licenses/by/4.0/
The bundled raw CSV is the unscaled, unfiltered copy supplied by scikit-learn load_wine.
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html
Packaging changes: named CSV columns; original cultivar IDs 1,2,3 encoded as 0,1,2.
The CSV has no added model features. Units are not inferred where metadata is incomplete.
Retrieved October 18, 2026. The SHA-256 of the exact CSV is in outputs/summary.json.

The original CSV contains chemical measurements and cultivar reference labels, not observed purchase transactions or verified fraud labels. Synthetic purchase records and their generated anomaly labels are documented separately in milestone2/README.md and produced by milestone2/experiment.py.
