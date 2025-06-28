# Computing-All-Embedded-Candidate-Keys
# Embedded Candidate Key Discovery and Performance Analysis

This repository implements and compares two algorithms for candidate key discovery in relational schemas:

- The **Classical Candidate Key (CK)** algorithm based on the Lucchesi–Osborn enumeration strategy.
- The **Embedded Candidate Key (ECK)** algorithm, which extends key discovery to support partial schemas, missing values, and embedded constraints (eFDs and eUCs).

---

## Features

- Implementation of classical candidate key discovery (CK) via closure and pruning.
- Embedded candidate key discovery (ECK) via local environments and constraint projection.
- Random synthetic data generation for controlled benchmarking.
- Runtime performance evaluation:
  - Execution time
  - Number of operations (closure + key checks)
  - Key counts
  - Theoretical and revised complexity estimation
- Automated plotting and visualization:
  - Time complexity comparison
  - Operation count comparison
  - Time vs. Operations plots

---

##  How to Run

Install required dependencies:
```bash
pip install matplotlib numpy
