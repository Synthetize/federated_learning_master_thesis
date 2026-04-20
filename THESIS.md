# MSc Thesis Registration: Resilience and Critical Failure Points of Federated Learning

**Student:** Leonardo Salpini
**Institution:** University of Camerino (UNICAM) - Master of Science in Computer Science
**Partner University:** Reykjavik University (Iceland)
**Supervisor:** Michela Quadrini
**Co-Supervisor:** Giovanni Apruzzese

---

## 1. Preliminary Title
**Resilience and Critical Failure Points of Federated Learning: A Stress-Test on Data Heterogeneity and Privacy Noise**

## 2. Abstract
The collaborative training of AI models in healthcare is frequently hindered by data isolation and strict privacy regulations like GDPR and HIPAA. Federated Learning (FL) addresses these challenges by enabling local model training without raw data sharing. However, the interplay between data heterogeneity (non-IID) and privacy-preserving noise can significantly reduce model utility. This thesis investigates the operational limits of a secure FL framework to identify the "failure points" where the system loses clinical utility compared to localized or centralized benchmarks.

## 3. Project Description
Federated Learning mitigates privacy risks by exchanging only model parameters rather than raw patient records. Despite this, the framework remains vulnerable to Inference and Reconstruction attacks. Implementing Differential Privacy (DP) is essential for compliance but introduces a trade-off between privacy and diagnostic accuracy. 

Furthermore, healthcare data is inherently heterogeneous across different institutions, leading to "client drift". This research develops a privacy-preserving FL framework (integrating **FedProx** and **DP**) to simulate real-world clinical scenarios and identify the specific thresholds where global model performance degrades below simple local training or centralized benchmarks.

## 4. Objectives
1.  **Combined Sensitivity Testing:** Varying data heterogeneity (via Dirichlet Concentration Parameter α) and privacy noise (via ε) to map model resilience.
2.  **Finding Breaking Points:** Determining thresholds where extreme class imbalance or excessive noise suppresses diagnostic signals. **Log-loss** will be used to detect model instability and unreliable confidence levels.
3.  **Clinical Utility Benchmarking:** Comparing the federated framework against centralized and average local models to quantify the "cost of privacy".

## 5. Workpackages (WPs)
* **WP1:** Literature Review and Environment Setup
* **WP2:** Framework Development (FedProx + DP)
* **WP3:** Initial Experimental Phase
* **WP4:** Mid-term Evaluation (Progress assessment in July)
* **WP5:** Testing and Benchmarking
* **WP6:** Thesis Finalization
* **WP7:** Continuous Thesis Drafting

## 6. Important Milestones
* **Registration Date:** 11/04/2026
* **Mid-term Evaluation:** 06/07/2026
* **Finalized Thesis:** 01/10/2026
* **Graduation Day:** 14/10/2026