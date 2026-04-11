---
dataset_info:
- config_name: algebra
  features:
  - name: problem
    dtype: string
  - name: level
    dtype: string
  - name: type
    dtype: string
  - name: solution
    dtype: string
  splits:
  - name: train
    num_bytes: 955021
    num_examples: 1744
  - name: test
    num_bytes: 648291
    num_examples: 1187
  download_size: 854357
  dataset_size: 1603312
- config_name: counting_and_probability
  features:
  - name: problem
    dtype: string
  - name: level
    dtype: string
  - name: type
    dtype: string
  - name: solution
    dtype: string
  splits:
  - name: train
    num_bytes: 667385
    num_examples: 771
  - name: test
    num_bytes: 353803
    num_examples: 474
  download_size: 501973
  dataset_size: 1021188
- config_name: default
  features:
  - name: problem
    dtype: string
  - name: level
    dtype: string
  - name: type
    dtype: string
  - name: solution
    dtype: string
  splits:
  - name: train
    num_bytes: 816245
    num_examples: 746
  - name: test
    num_bytes: 552893
    num_examples: 546
  download_size: 591622
  dataset_size: 1369138
- config_name: geometry
  features:
  - name: problem
    dtype: string
  - name: level
    dtype: string
  - name: type
    dtype: string
  - name: solution
    dtype: string
  splits:
  - name: train
    num_bytes: 1077241
    num_examples: 870
  - name: test
    num_bytes: 523126
    num_examples: 479
  download_size: 807701
  dataset_size: 1600367
- config_name: intermediate_algebra
  features:
  - name: problem
    dtype: string
  - name: level
    dtype: string
  - name: type
    dtype: string
  - name: solution
    dtype: string
  splits:
  - name: train
    num_bytes: 1157476
    num_examples: 1295
  - name: test
    num_bytes: 795070
    num_examples: 903
  download_size: 965232
  dataset_size: 1952546
- config_name: number_theory
  features:
  - name: problem
    dtype: string
  - name: level
    dtype: string
  - name: type
    dtype: string
  - name: solution
    dtype: string
  splits:
  - name: train
    num_bytes: 595793
    num_examples: 869
  - name: test
    num_bytes: 349455
    num_examples: 540
  download_size: 486821
  dataset_size: 945248
- config_name: prealgebra
  features:
  - name: problem
    dtype: string
  - name: level
    dtype: string
  - name: type
    dtype: string
  - name: solution
    dtype: string
  splits:
  - name: train
    num_bytes: 715611
    num_examples: 1205
  - name: test
    num_bytes: 510195
    num_examples: 871
  download_size: 647529
  dataset_size: 1225806
- config_name: precalculus
  features:
  - name: problem
    dtype: string
  - name: level
    dtype: string
  - name: type
    dtype: string
  - name: solution
    dtype: string
  splits:
  - name: train
    num_bytes: 816245
    num_examples: 746
  - name: test
    num_bytes: 552893
    num_examples: 546
  download_size: 591622
  dataset_size: 1369138
configs:
- config_name: algebra
  data_files:
  - split: train
    path: algebra/train-*
  - split: test
    path: algebra/test-*
- config_name: counting_and_probability
  data_files:
  - split: train
    path: counting_and_probability/train-*
  - split: test
    path: counting_and_probability/test-*
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
- config_name: geometry
  data_files:
  - split: train
    path: geometry/train-*
  - split: test
    path: geometry/test-*
- config_name: intermediate_algebra
  data_files:
  - split: train
    path: intermediate_algebra/train-*
  - split: test
    path: intermediate_algebra/test-*
- config_name: number_theory
  data_files:
  - split: train
    path: number_theory/train-*
  - split: test
    path: number_theory/test-*
- config_name: prealgebra
  data_files:
  - split: train
    path: prealgebra/train-*
  - split: test
    path: prealgebra/test-*
- config_name: precalculus
  data_files:
  - split: train
    path: precalculus/train-*
  - split: test
    path: precalculus/test-*
---
