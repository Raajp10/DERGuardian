# AOI Metric Definition

**Status:** implemented as an experimental, repo-specific metric.

In this repository, AOI means **Alert Overlap Index**. It is defined at the window level as:

`AOI = TP_windows / (TP_windows + FP_windows + FN_windows)`

This is the Jaccard overlap between the set of windows predicted as alerts and the set of windows labeled as attacks.
It is computable from existing prediction artifacts that contain `attack_present` and `predicted` columns.

Important constraints:

- AOI is not used to replace precision, recall, F1, ROC-AUC, or average precision.
- AOI is not claimed as a standard DER cybersecurity metric here.
- AOI does not alter the frozen canonical benchmark winner: transformer @ 60s.
- AOI rows are reported only where real prediction artifacts exist.
