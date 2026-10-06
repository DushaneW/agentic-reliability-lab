### random_run  (6630 examples, 1360 runs; headline = prefixes <= 60%)

| Predictor | AUROC early [95% CI] | AUPRC early | Brier early |
| --- | --- | --- | --- |
| constant (no features) | 0.500 [0.500, 0.500] | 0.403 | 0.250 |
| length only | 0.547 [0.533, 0.560] | 0.472 | 0.240 |
| error count only | 0.549 [0.521, 0.579] | 0.492 | 0.229 |
| logistic (all features) | 0.771 [0.756, 0.788] | 0.742 | 0.181 |
| gradient boosting (all features) | 0.791 [0.773, 0.808] | 0.776 | 0.160 |

AUROC by trajectory completion (point estimate [95% CI]):

| Predictor | 20% | 40% | 60% | 80% | 100% |
| --- | --- | --- | --- | --- | --- |
| constant (no features) | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| length only | 0.567 [0.535, 0.595] | 0.573 [0.541, 0.601] | 0.548 [0.516, 0.581] | 0.612 [0.580, 0.642] | 0.578 [0.547, 0.609] |
| error count only | 0.476 [0.446, 0.507] | 0.544 [0.515, 0.577] | 0.639 [0.610, 0.668] | 0.679 [0.647, 0.708] | 0.759 [0.729, 0.785] |
| logistic (all features) | 0.596 [0.558, 0.627] | 0.793 [0.764, 0.818] | 0.843 [0.817, 0.870] | 0.954 [0.940, 0.967] | 1.000 [1.000, 1.000] |
| gradient boosting (all features) | 0.575 [0.542, 0.603] | 0.796 [0.767, 0.821] | 0.844 [0.819, 0.869] | 0.948 [0.934, 0.962] | 1.000 [1.000, 1.000] |

### leave_task_out  (6630 examples, 1360 runs; headline = prefixes <= 60%)

| Predictor | AUROC early [95% CI] | AUPRC early | Brier early |
| --- | --- | --- | --- |
| constant (no features) | 0.500 [0.500, 0.500] | 0.403 | 0.250 |
| length only | 0.543 [0.529, 0.555] | 0.483 | 0.240 |
| error count only | 0.523 [0.493, 0.550] | 0.487 | 0.229 |
| logistic (all features) | 0.768 [0.752, 0.785] | 0.749 | 0.181 |
| gradient boosting (all features) | 0.789 [0.774, 0.807] | 0.783 | 0.159 |

AUROC by trajectory completion (point estimate [95% CI]):

| Predictor | 20% | 40% | 60% | 80% | 100% |
| --- | --- | --- | --- | --- | --- |
| constant (no features) | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| length only | 0.559 [0.526, 0.588] | 0.559 [0.529, 0.594] | 0.533 [0.498, 0.564] | 0.594 [0.557, 0.621] | 0.557 [0.522, 0.590] |
| error count only | 0.450 [0.417, 0.474] | 0.520 [0.490, 0.550] | 0.613 [0.577, 0.644] | 0.660 [0.626, 0.688] | 0.746 [0.716, 0.773] |
| logistic (all features) | 0.590 [0.559, 0.624] | 0.788 [0.759, 0.817] | 0.841 [0.816, 0.867] | 0.945 [0.929, 0.960] | 1.000 [1.000, 1.000] |
| gradient boosting (all features) | 0.567 [0.536, 0.598] | 0.794 [0.763, 0.822] | 0.839 [0.813, 0.866] | 0.940 [0.924, 0.956] | 1.000 [1.000, 1.000] |

AUROC inside each held-out group (prefixes <= 60%; groups with a single outcome class are omitted; 0.500 = chance):

| Predictor | coding-001 | coding-002 | coding-003 | coding-004 | coding-005 | coding-006 | coding-007 | coding-008 | coding-009 | coding-010 | smoke-001 | smoke-002 | tool-use-001 | tool-use-002 | tool-use-003 | tool-use-004 | tool-use-005 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constant (no features) | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| length only | 0.563 | 0.555 | 0.587 | 0.547 | 0.566 | 0.550 | 0.566 | 0.557 | 0.546 | 0.570 | 0.551 | 0.550 | 0.565 | 0.569 | 0.544 | 0.567 | 0.528 |
| error count only | 0.553 | 0.550 | 0.582 | 0.563 | 0.582 | 0.582 | 0.569 | 0.570 | 0.572 | 0.532 | 0.564 | 0.595 | 0.551 | 0.587 | 0.541 | 0.547 | 0.633 |
| logistic (all features) | 0.764 | 0.789 | 0.744 | 0.766 | 0.779 | 0.737 | 0.762 | 0.788 | 0.765 | 0.752 | 0.794 | 0.802 | 0.773 | 0.757 | 0.742 | 0.781 | 0.816 |
| gradient boosting (all features) | 0.800 | 0.813 | 0.768 | 0.795 | 0.793 | 0.737 | 0.786 | 0.824 | 0.803 | 0.786 | 0.813 | 0.828 | 0.785 | 0.783 | 0.771 | 0.808 | 0.846 |

### leave_profile_out  (6630 examples, 1360 runs; headline = prefixes <= 60%)

| Predictor | AUROC early [95% CI] | AUPRC early | Brier early |
| --- | --- | --- | --- |
| constant (no features) | 0.500 [0.500, 0.500] | 0.403 | 0.250 |
| length only | 0.194 [0.177, 0.212] | 0.275 | 0.272 |
| error count only | 0.278 [0.255, 0.300] | 0.398 | 0.252 |
| logistic (all features) | 0.785 [0.766, 0.805] | 0.790 | 0.184 |
| gradient boosting (all features) | 0.714 [0.695, 0.736] | 0.726 | 0.176 |

AUROC by trajectory completion (point estimate [95% CI]):

| Predictor | 20% | 40% | 60% | 80% | 100% |
| --- | --- | --- | --- | --- | --- |
| constant (no features) | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] |
| length only | 0.139 [0.121, 0.160] | 0.154 [0.135, 0.174] | 0.159 [0.138, 0.185] | 0.216 [0.191, 0.245] | 0.203 [0.179, 0.228] |
| error count only | 0.137 [0.117, 0.158] | 0.283 [0.253, 0.315] | 0.433 [0.396, 0.472] | 0.492 [0.461, 0.528] | 0.617 [0.577, 0.650] |
| logistic (all features) | 0.864 [0.841, 0.884] | 0.791 [0.761, 0.822] | 0.779 [0.749, 0.810] | 0.944 [0.927, 0.961] | 0.987 [0.979, 0.993] |
| gradient boosting (all features) | 0.309 [0.273, 0.345] | 0.690 [0.654, 0.724] | 0.763 [0.731, 0.799] | 0.926 [0.908, 0.945] | 1.000 [1.000, 1.000] |

AUROC inside each held-out group (prefixes <= 60%; groups with a single outcome class are omitted; 0.500 = chance):

| Predictor | early_stop | mixed_heavy | mixed_severe | repeat | wrong_tool | wrong_tool_ignore_error |
| --- | --- | --- | --- | --- | --- | --- |
| constant (no features) | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| length only | 0.758 | 0.500 | 0.602 | 0.500 | 0.500 | 0.500 |
| error count only | 0.500 | 0.533 | 0.559 | 0.500 | 0.531 | 0.559 |
| logistic (all features) | 0.708 | 0.738 | 0.801 | 0.775 | 0.588 | 0.617 |
| gradient boosting (all features) | 0.907 | 0.774 | 0.832 | 0.775 | 0.624 | 0.645 |
