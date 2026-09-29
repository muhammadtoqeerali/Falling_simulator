# Phase-2 CNN Simulation Augmentation Research Snapshot

## Campaign status

Final campaign completed successfully.

Completion timestamp from campaign log:

**2026-08-23 04:53:42 — PHASE 2 COMPLETE**

All six experimental conditions completed five folds each:

- EXP01_REAL_ONLY
- EXP02_SIM_ONLY
- EXP03_MIX20
- EXP04_MIX50
- EXP05_MIX70
- EXP06_MIX100

Total final training/evaluation folds: **30**

Each fold was verified to contain:

- `best_model.pth`
- `metrics.csv`
- `confusion_matrix.csv`
- `training_history.csv`

---

# Final CNN Results

| Experiment | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| EXP01 Real only | 0.7904 ± 0.0126 | 0.8929 ± 0.0073 | 0.7183 ± 0.0257 | 0.7959 ± 0.0150 |
| EXP02 Simulation only | 0.9453 ± 0.0018 | 0.9497 ± 0.0015 | 0.9867 ± 0.0016 | 0.9678 ± 0.0011 |
| EXP03 Real + 20% Simulation | 0.8298 ± 0.0187 | 0.8678 ± 0.0344 | 0.8290 ± 0.0130 | 0.8475 ± 0.0134 |
| EXP04 Real + 50% Simulation | 0.8194 ± 0.0055 | 0.8565 ± 0.0124 | 0.8207 ± 0.0143 | 0.8380 ± 0.0052 |
| EXP05 Real + 70% Simulation | 0.8152 ± 0.0085 | 0.8498 ± 0.0321 | 0.8232 ± 0.0346 | 0.8353 ± 0.0076 |
| EXP06 Real + 100% Simulation | 0.8063 ± 0.0140 | 0.8382 ± 0.0281 | 0.8189 ± 0.0121 | 0.8281 ± 0.0104 |

## Augmentation gain versus real-only baseline

### Real + 20% simulation

- Accuracy: **+0.039420**
- Precision: **-0.025128**
- Recall: **+0.110694**
- F1: **+0.051597**

This is the strongest mixed-domain configuration.

### Real + 50% simulation

- Accuracy: +0.028940
- Precision: -0.036426
- Recall: +0.102321
- F1: +0.042118

### Real + 70% simulation

- Accuracy: +0.024809
- Precision: -0.043054
- Recall: +0.104831
- F1: +0.039380

### Real + 100% simulation

- Accuracy: +0.015885
- Precision: -0.054644
- Recall: +0.100594
- F1: +0.032231

---

# Main research observation

Moderate simulated-data augmentation improved CNN performance on held-out
real data.

The strongest setting was **20% simulated augmentation**.

Relative to the real-only model, MIX20 produced approximately:

- **+3.94 percentage points accuracy**
- **+11.07 percentage points recall**
- **+5.16 percentage points F1**
- **-2.51 percentage points precision**

The major gain is therefore increased fall sensitivity / recall.

Increasing the simulated fraction beyond 20% progressively reduced accuracy,
precision and F1 relative to MIX20, although all mixed configurations retained
substantially higher recall than the real-only baseline.

This provides a useful synthetic-data dose-response result:
more simulated data is not automatically better.

---

# EXP02 Simulation-Only Interpretation

EXP02_SIM_ONLY achieved:

- Accuracy: 94.53%
- Precision: 94.97%
- Recall: 98.67%
- F1: 96.78%

This experiment demonstrates strong learnability within the simulated domain.

**Important:** EXP02 must not be presented as a +15.49-point real-world
accuracy improvement over EXP01 because EXP02 is trained/evaluated within
the simulation domain, whereas EXP01 and the mixed experiments address
performance on real data.

---

# Dataset information

## Extended35 simulation

- High-rate truth files: **572**
- CNN windows: **1,377,447**
- Window shape: **30 × 9**

Relevant simulated high-rate columns:

- accel_true_x
- accel_true_y
- accel_true_z
- gyro_true_x
- gyro_true_y
- gyro_true_z
- accel_true_mag
- gyro_true_mag
- sensor_pos_x/y/z
- sensor_vel_x/y/z
- pelvis_height

## Real datasets

KFall files: **5,075**

UniVRFall files: **1,234**

Combined real CNN windows:

**4,918,911 × 30 × 9**

---

# Input unit harmonization

A major methodological correction was performed before the final campaign.

## Simulated acceleration

Extended35 acceleration is already expressed in physical SI units:

**m/s²**

No acceleration scaling is required.

## Simulated gyroscope

MuJoCo generator angular velocity is expressed in:

**rad/s**

Conversion used:

`gyro_deg_s = gyro_rad_s × 57.295779513`

## Real acceleration

Real KFall / UniVR acceleration values are raw sensor counts.

LIS3DH sensor resolution used:

**1 mg/count**

Conversion:

`Acc_raw × 0.00981 -> m/s²`

## Real gyroscope

Real gyro values are raw counts.

LSM6DS3 sensitivity used:

**0.07 deg/s per count**

Conversion:

`Gyr_raw × 0.07 -> deg/s`

## Orientation channels

Three orientation/Euler channels are currently set to zero for all datasets.

This is consistent because the Protechto CNN normalizer subsequently retains
the six acceleration + gyroscope channels.

---

# Sensor specification represented by simulation

Generator configuration records:

- MCU: STM32F722RET6
- Accelerometer: LIS3DH
- Range: ±16 g
- Resolution: 1 mg
- Gyroscope: LSM6DS3
- Range: ±2000 deg/s
- Resolution: 0.07 deg/s
- Sampling frequency: 100 Hz
- Mount location: lower back / L1-L2 proxy
- MuJoCo proxy body: torso
- Proxy fraction torso-to-pelvis: 0.40

---

# CNN normalization after physical-unit correction

The active `IMUNormalizer.py` uses physical-unit compatible normalization.

Acceleration:

`acc / [39.24, 39.24, 39.24]`

Gyroscope:

`gyro / [1800, 1800, 1800]`

The previous normalizer intended for raw/mdps-style units was backed up as:

`IMUNormalizer_backup_before_physical_units.py`

---

# Training settings

Final campaign:

- CNN input channels before model split: 9
- Actual accel + gyro features used by CNN: 6
- Window length: 30 samples
- Cross-validation folds: 5
- Maximum epochs: 100
- Early-stopping patience: 20
- Batch size: 64
- Random seed: 42

---

# Important methodological cautions for paper writing

These points must be audited before making final publication claims.

## 1. EXP02 simulated grouping

The current simulation-only GroupKFold implementation assigned simulated
groups at window level rather than guaranteed source-trial/profile level.

Consequently, windows originating from the same simulated trajectory may
potentially appear in different EXP02 folds.

Therefore:

**Do not use EXP02 as evidence of subject-independent or trial-independent
simulation generalization until it is rerun with source-trial/profile groups.**

This issue does not automatically invalidate the mixed-domain comparison,
where the research question concerns simulation augmentation of real-data
training and evaluation on held-out real groups.

## 2. Window labels

The current windowing procedure should be audited against the original
Protechto preprocessing to determine whether all windows from a fall-labelled
trial inherit the trial-level fall label.

If so, quiet pre/post-event windows may also be labelled as fall.

Do not silently change this until comparability with the canonical Protechto
pipeline is established.

## 3. Validation splitting

The internal training/validation split should be compared directly with
Protechto's canonical subject-level validation design.

The outer real-data GroupKFold evaluation is important, but internal
validation methodology should also be documented precisely.

## 4. Simulation label composition

The archived file:

`methodology/extended35_label_counts.json`

records the actual cached Extended35 label distribution and should be checked
before describing simulation-only class balance.

---

# Recommended paper-level statement

A defensible current result statement is:

"Moderate augmentation with MuJoCo-derived synthetic IMU samples improved
CNN fall-detection performance on held-out real data. The strongest mixed
configuration used 20% simulated augmentation, increasing recall by
approximately 11.1 percentage points and F1 by approximately 5.2 percentage
points relative to real-only training. Larger simulated proportions retained
higher recall than the real-only baseline but showed progressively lower
precision and F1, indicating a non-monotonic relationship between synthetic
data quantity and real-domain performance."

---

# Next analysis stage

The next major analysis should compare:

1. CNN segment-level performance
2. Threshold / event-level fall detection
3. Real versus simulated event characteristics
4. Event-level false positives / false negatives
5. Cross-dataset performance where methodologically appropriate

Do not overwrite the archived campaign while performing those analyses.
