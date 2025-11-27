# Graph-based Few-shot **Multivariate** Time Series Anomaly Detection (with Lightweight Explanations)

> A compact alternative to TAMA that models cross-metric structure with a graph encoder, supports few-shot anomaly **typing**, and produces short textual explanations — while reusing TAMA-style window aggregation and evaluation utilities where helpful.

---

## 1) Overview

- **Task:** Multivariate time series anomaly detection (window & point level) + few-shot anomaly **typing** + short **explanations**.
- **Key idea:** Build a **metric graph** (edges from correlation/MI), encode windowed signals with a **temporal block + 1 graph layer**, score anomalies per window, and **aggregate** over overlapping windows to get pointwise scores (TAMA-style).
- **Why not LMMs?** No proprietary multimodal LMM is required. Explanations start as **templates**; optionally polish with a **small open-source** LM if desired.
- **Few-shot:** With only K labeled anomalies per type (K∈{1,5}), use **prototype** classification in the learned embedding space.

**Minimal equations (plain-text):**
- `z_w = f_theta(X[w], G)`  (temporal + graph)
- `s_w = sigmoid(w^T z_w + b)`  (window score)
- `S(t) = mean_{w: t in w} s_w`  (pointwise aggregation, cf. TAMA)
- Prototypes: `p_k = mean_{w in S_k} z_w`; typing: `argmin_k || z_w - p_k ||_2^2`

---

## 2) Environment

```bash
# Python 3.10+ recommended
pip install -r requirements.txt
