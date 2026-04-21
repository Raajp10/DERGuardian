# Deployment Environment Manifest

## Host machine

- Platform: Windows-10-10.0.19045-SP0
- Python: 3.11.9
- CPU: Intel(R) Core(TM) i7-8850H CPU @ 2.60GHz
- Physical cores: 6
- Logical cores: 12
- System memory (GB): 15.9

## Library versions

- torch: 2.10.0+cpu
- transformers: 4.57.6
- scikit_learn: 1.7.2
- pandas: 3.0.2
- numpy: 2.4.4
- psutil: 7.2.2
- tsfm_public: 0.3.5

## Benchmark scope

- This benchmark used offline replay inputs already present in the workspace.
- No edge hardware was used.
- TTM is not included in this deployment benchmark because it currently exists as an extension checkpoint rather than a frozen Phase 1 ready-package runtime contract.

## Benchmarked frozen packages

- threshold_baseline @ 10s: package=`<repo>/outputs\window_size_study\10s\ready_packages\threshold_baseline`, size_mb=`0.246`
- transformer @ 60s: package=`<repo>/outputs\window_size_study\60s\ready_packages\transformer`, size_mb=`0.333`
- lstm @ 300s: package=`<repo>/outputs\window_size_study\300s\ready_packages\lstm`, size_mb=`0.122`
