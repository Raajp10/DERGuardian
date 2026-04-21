# XAI Final Validation Report

The heldout XAI layer is now audited at case level. The safe interpretation remains conservative: grounded explanation layer, post-alert operator-facing support, and evidence-grounded family attribution.

## Generator summary

```text
generator_source  case_count  family_accuracy  family_top3_accuracy  asset_accuracy  evidence_grounding_rate  action_relevance  partial_alignment  unsupported_claim_rate
         chatgpt          43         0.744186              0.837209        0.139535                 0.199889          0.872093           0.808887                0.069767
          claude          36         0.861111              0.972222        0.222222                 0.187698          0.736111           0.871667                0.055556
          gemini          29         0.896552              0.931034        0.258621                 0.359195          0.741379           0.896552                0.000000
            grok          47         0.659574              0.829787        0.425532                 0.301418          0.680851           0.724379                0.063830
         overall         155         0.774194              0.883871        0.267742                 0.257650          0.758065           0.814245                0.051613
```

## Interpretation

- Overall family accuracy: 0.774
- Overall asset accuracy: 0.268
- Overall evidence grounding rate: 0.258
- Overall action relevance: 0.758
- Unsupported claim rate: 0.052

Family attribution is materially stronger than asset attribution. Grounding is partial rather than exhaustive, so the correct claim remains a grounded explanation layer for operator support rather than human-like root-cause analysis.
