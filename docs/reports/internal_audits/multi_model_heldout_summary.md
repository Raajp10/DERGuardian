# Multi-model Heldout Summary

## Aggregate Replay Means

```text
        model_name window_label  precision   recall       f1  mean_latency_seconds
              lstm         300s   0.946255 0.718485 0.809885             65.666667
       transformer          60s   0.825609 0.562208 0.661286              3.700000
threshold_baseline          10s   0.414570 0.272966 0.251209            114.466667
```

## Required Answers

- Does transformer remain best under heldout shift? no. Best mean heldout replay row: lstm (300s) with mean F1=0.8099.
- Does a different frozen model generalize better? yes. transformer(60s) mean F1=0.6613; best heldout replay row=lstm(300s) with mean F1=0.8099.
- Is lower-latency windowing more robust than 60s under generator shift? no. The 10s threshold baseline had mean F1=0.2512 and mean latency=114.47s, while transformer(60s) had mean F1=0.6613 and mean latency=3.70s.
- Does one model trade F1 for better latency? yes. lstm(300s) had the strongest mean heldout replay F1=0.8099 but much slower mean latency=65.67s, while transformer(60s) kept mean latency=3.70s with lower mean F1=0.6613. The 10s threshold baseline was neither the strongest nor the fastest in this replay context.
