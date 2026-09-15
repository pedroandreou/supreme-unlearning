# Reference results and community contributions

Explore the [published results](https://pedroandreou.github.io/supreme-unlearning-page/results/)
or [download all paper tables](../docs/results/). The reference tables below
contain existing paper values for Pins Face Recognition. They are not a universal
ranking of methods.

## Published reference results

Values are mean ± standard deviation across ten seeds (260–269), using one
NVIDIA L40S GPU and the matched seed protocol (J = K = 1). Full-class values
average five forget identities within each seed; random-sample values use a
0.1% forget set. Raw accuracies are percentages. MIA is a membership-inference
score. Compare each method with the retrained reference on the relevant metric;
lower raw forget accuracy is not a universal success criterion.

### ResNet18: full-class

| Method | Forget accuracy (%) | Retain accuracy (%) | MIA score |
|---|---:|---:|---:|
| Retrain (reference) | 0.00 ± 0.00 | 97.41 ± 0.12 | 0.05 ± 0.01 |
| FT | 29.22 ± 5.21 | 99.94 ± 0.07 | 0.00 ± 0.00 |
| BadT | 0.26 ± 0.17 | 95.07 ± 0.30 | 0.00 ± 0.00 |
| UNSIR | 89.44 ± 1.98 | 99.93 ± 0.01 | 0.04 ± 0.03 |
| RL | 0.00 ± 0.00 | 100.00 ± 0.00 | 0.08 ± 0.02 |
| SSD | 1.97 ± 6.22 | 88.28 ± 7.41 | 0.06 ± 0.07 |
| LFSSD | 0.00 ± 0.00 | 93.76 ± 1.54 | 0.02 ± 0.01 |

### ResNet18: random-sample (forget 0.1%)

| Method | Forget accuracy (%) | Retain accuracy (%) | MIA score |
|---|---:|---:|---:|
| Retrain (reference) | 88.33 ± 8.47 | 97.69 ± 0.22 | 0.06 ± 0.18 |
| FT | 91.11 ± 20.65 | 95.09 ± 15.30 | 0.00 ± 0.00 |
| BadT | 53.33 ± 24.60 | 65.23 ± 29.78 | 0.04 ± 0.08 |
| RL | 39.44 ± 37.45 | 93.10 ± 20.16 | 0.12 ± 0.13 |
| SSD | 13.33 ± 21.63 | 17.73 ± 23.72 | 0.00 ± 0.00 |
| LFSSD | 20.00 ± 15.09 | 39.09 ± 19.16 | 0.00 ± 0.00 |

### ViT: full-class

| Method | Forget accuracy (%) | Retain accuracy (%) | MIA score |
|---|---:|---:|---:|
| Retrain (reference) | 0.00 ± 0.00 | 99.78 ± 0.03 | 0.00 ± 0.00 |
| FT | 0.03 ± 0.06 | 77.12 ± 23.47 | 0.04 ± 0.08 |
| BadT | 17.90 ± 5.18 | 98.90 ± 0.11 | 0.00 ± 0.00 |
| UNSIR | 35.52 ± 4.92 | 99.47 ± 0.13 | 0.00 ± 0.00 |
| RL | 0.00 ± 0.00 | 99.98 ± 0.02 | 0.02 ± 0.00 |
| SSD | 0.00 ± 0.00 | 97.38 ± 1.00 | 0.00 ± 0.00 |
| LFSSD | 0.00 ± 0.00 | 95.90 ± 2.09 | 0.00 ± 0.00 |

### ViT: random-sample (forget 0.1%)

| Method | Forget accuracy (%) | Retain accuracy (%) | MIA score |
|---|---:|---:|---:|
| Retrain (reference) | 90.00 ± 5.74 | 98.58 ± 0.13 | 0.00 ± 0.00 |
| FT | 98.33 ± 3.75 | 100.00 ± 0.00 | 0.00 ± 0.00 |
| BadT | 81.67 ± 7.43 | 98.79 ± 0.62 | 0.19 ± 0.05 |
| RL | 13.33 ± 8.36 | 99.91 ± 0.01 | 0.01 ± 0.04 |
| SSD | 35.00 ± 40.83 | 44.49 ± 45.61 | 0.00 ± 0.00 |
| LFSSD | 0.56 ± 1.76 | 7.89 ± 6.59 | 0.00 ± 0.00 |

Source: [SUPREME paper](https://arxiv.org/abs/2606.00380), appendix raw-values
table (`tab:pins_raw_values`). [CSV](../docs/results/pins_raw.csv) and
[measurement definitions](../docs/results/README.md).
UNSIR is excluded from random-sample unlearning by design. The paper's
difference tables, including their standard deviations, are available separately.

## Contribute a method or result

Implement and register your component using [the extension guide](../docs/extending.md).
Start from [the method template](methods/template/) and submit a pull request with:

- The method and original paper, plus the SUPREME commit and dependency environment.
- Dataset, model, scenario, forget target and exact split configuration.
- Training, unlearning and evaluation seeds and their pairing.
- The original checkpoint and retrained reference provenance, where applicable.
- An executable configuration, metric definitions and results at each seed.
- Hardware and timing scope for any efficiency claim.

Keep submissions for different experimental settings in separate tables. Published
reference values stay tied to their source; new submissions should identify their
own provenance. [Discuss an experiment](https://github.com/pedroandreou/supreme-unlearning/discussions)
if you need help selecting a configuration.
