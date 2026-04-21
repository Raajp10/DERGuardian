# Zero-Day Family Difficulty Summary

This ranking uses mean per-scenario F1 and detection rate across completed heldout synthetic evaluations. Lower values indicate harder synthetic transfer families within this bounded benchmark.

## Family Ranking

| attack_family        |   scenario_count |   mean_recall |   mean_f1 |   detection_rate |
|:---------------------|-----------------:|--------------:|----------:|-----------------:|
| false_data_injection |              209 |        0.5804 |    0.3642 |           0.8333 |
| unauthorized_command |              133 |        0.5971 |    0.3723 |           0.8789 |
| degradation          |               95 |        0.5953 |    0.377  |           0.8487 |
| command_suppression  |               95 |        0.711  |    0.4776 |           0.8947 |
| command_delay        |               76 |        0.7721 |    0.5062 |           0.9474 |
| coordinated_campaign |               57 |        0.7211 |    0.5293 |           0.9123 |

## Hardest Families In This Pass

| attack_family        |   scenario_count |   mean_recall |   mean_f1 |   detection_rate |
|:---------------------|-----------------:|--------------:|----------:|-----------------:|
| false_data_injection |              209 |        0.5804 |    0.3642 |           0.8333 |
| unauthorized_command |              133 |        0.5971 |    0.3723 |           0.8789 |
| degradation          |               95 |        0.5953 |    0.377  |           0.8487 |

## Easiest Families In This Pass

| attack_family        |   scenario_count |   mean_recall |   mean_f1 |   detection_rate |
|:---------------------|-----------------:|--------------:|----------:|-----------------:|
| coordinated_campaign |               57 |        0.7211 |    0.5293 |           0.9123 |
| command_delay        |               76 |        0.7721 |    0.5062 |           0.9474 |
| command_suppression  |               95 |        0.711  |    0.4776 |           0.8947 |
