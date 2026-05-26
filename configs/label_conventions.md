# Label Conventions

The scripts accept configurable labels. The defaults used in the examples are:

## Organ Mask

For KiTS-style organ masks:

| Label | Structure |
|---:|---|
| 1 | Kidney |
| 2 | Tumor |
| 3 | Cyst |

For KIPA-style vessel masks, adjust labels according to the model that produced the mask.

## Vessel Mask

| Label | Structure |
|---:|---|
| 3 | Renal vein |

## Fat Mask

For AATTCT-style fat masks:

| Label | Structure |
|---:|---|
| 1 | SAT |
| 2 | VAT |

Always verify label mapping before running cohort-level extraction.
