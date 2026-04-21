# XAI Error Taxonomy

The taxonomy below is computed from the heldout v4 case packets, not from free-form narration.

## Overall counts

```text
           primary_error_category  count
     correct family / wrong asset     77
correct family / partial evidence     30
 wrong family / grounded evidence     26
                    fully aligned     13
                unsupported claim      8
                    mixed partial      1
```

## By generator

```text
generator_source            primary_error_category  count
         chatgpt correct family / partial evidence      6
         chatgpt      correct family / wrong asset     26
         chatgpt                 unsupported claim      3
         chatgpt  wrong family / grounded evidence      8
          claude correct family / partial evidence      9
          claude      correct family / wrong asset     22
          claude                     mixed partial      1
          claude                 unsupported claim      2
          claude  wrong family / grounded evidence      2
          gemini correct family / partial evidence      6
          gemini      correct family / wrong asset     16
          gemini                     fully aligned      4
          gemini  wrong family / grounded evidence      3
            grok correct family / partial evidence      9
            grok      correct family / wrong asset     13
            grok                     fully aligned      9
            grok                 unsupported claim      3
            grok  wrong family / grounded evidence     13
```

## Reading guide

- `correct family / wrong asset`: family attribution landed, but localization missed the affected asset.
- `correct family / partial evidence`: family is right, but cited grounded signals only partially cover the true observable set.
- `wrong family / grounded evidence`: evidence overlaps the true signals, but the family call still drifts.
- `unsupported claim`: unsupported support-map references or a wrong family with no grounding overlap.
- `mixed partial`: partial cases that do not fit the tighter headline bins above.
- `fully aligned`: correct family, correct asset set, and strong signal grounding.
