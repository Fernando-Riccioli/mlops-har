# Model card: har-classificatore

**English** · [Italiano](MODEL_CARD.it.md)

**Task.** Recognise 6 human activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS,
SITTING, STANDING, LAYING) from 2.56 s windows of accelerometer and gyroscope data from a
waist-mounted smartphone (50 Hz), described by the 561 features computed by the dataset
authors.

**Data.** UCI HAR, *Human Activity Recognition Using Smartphones* (Anguita et al., 2013),
CC BY 4.0 licence. 30 volunteers aged 19-48, a single phone model (Samsung Galaxy S II).
Official per-subject split: 21 subjects in train (7352 windows), 9 in test (2947 windows).
The exact data version is the `dataset_sha256` hash, stored as a tag on every model version.

**Evaluation.**
- Promotion metric: macro-F1 on the test set (the 9 unseen subjects). Macro, so that every
  class weighs the same despite different frequencies.
- Stability check: macro-F1 in *per-subject* cross-validation (5-fold GroupKFold on train),
  so no subject is ever in both training and validation.
- Production gate: test macro-F1 ≥ 0.90 and not worse than the version in production.

**Known limitations.**
- One phone and one position (waist): quality on other devices, or in a pocket or hand, is
  not measured.
- 30 young adult subjects: generalisation to older people or to people with movement
  disorders is not measured.
- The most frequent error is between static postures. For v1 (logistic regression), on the
  2947 test windows, 58 SITTING windows are predicted as STANDING and 14 the other way round:
  72 of the 132 errors. The second group is stairs: 23 WALKING_UPSTAIRS predicted as WALKING
  and 18 WALKING_DOWNSTAIRS predicted as WALKING_UPSTAIRS.
- The features are the dataset's precomputed ones: in production the exact same extraction
  chain would have to be reproduced.

**Intended use.** A learning exercise in MLOps (tracking, registry, data versioning).
Not for clinical use or for decisions about people.
