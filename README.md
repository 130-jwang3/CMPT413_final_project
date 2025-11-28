# See it, Think it, Sorted: Large Multimodal Models are Few-shot Time Series Anomaly Analyzers

Extended implementation of **TAMA** for the paper:
[See it, Think it, Sorted: Large Multimodal Models are Few-shot Time Series Anomaly Analyzers](https://arxiv.org/abs/2411.02465)

![](imgs/Flow-Chart.png)

---

## 0) Quick Summary (what actually works best right now)

* **Image pipeline (`main_cli.py`)** → use a **vision model** (e.g., `llava`) via **Ollama** ✅

* **Text pipeline (`main_cli_text.py`)** → use a **text-only instruction model** (e.g., `mistral:instruct`) via **Ollama** ✅
  *(llava can accept text, but in practice it’s much worse at structured JSON; use a text model)*

* If you **haven’t built EH_GAM_EGAN indices** for a dataset yet, run:

  ```bash
  python3 -m Datasets.Dataset
  ```

  once before converting datasets.

* **Evaluation tooling** currently expects a generic log name; **rename/symlink** your log file(s) as noted in §5.

---

## 1) Environment Setup

### OS options

* **Ubuntu 22.04+** (native) — recommended
* **Windows + WSL2 (Ubuntu)** — also works. If Ollama runs on Windows and Python runs in WSL, set `OLLAMA_HOST` to a reachable address (see §2).

### Python

```bash
pip install -r requirements.txt
```

---

## 2) Local LLMs with Ollama

Install Ollama:

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh
# macOS: brew install ollama  (or use the official pkg)
# Windows: install from https://ollama.com
```

Pull the models we use:

```bash
ollama pull llava            # for IMAGE pipeline
ollama pull mistral:instruct # for TEXT pipeline (instruction-tuned)
```

**.env** (create at repo root):

```env
# If Ollama & code run on the same machine:
OLLAMA_HOST=http://localhost:11434

# If running code in WSL and Ollama on Windows host,
# try localhost first; if it fails, use the Windows host's LAN/WSL-bridge IP.
# Example:
# OLLAMA_HOST=http://172.24.80.1:11434

# Default model (you can override per run)
# For image runs (main_cli.py):
# OLLAMA_MODEL=llava
# For text runs (main_cli_text.py):
# OLLAMA_MODEL=mistral:instruct

# optional: generous token-per-minute limit for our rate limiter bookkeeping
OLLAMA_TPM=200000
```

Sanity check:

```bash
curl $OLLAMA_HOST/api/tags
```

> **Why two models?**
>
> * `llava` is good at reasoning over images (used by `main_cli.py`).
> * `mistral:instruct` follows instructions and emits JSON reliably for text (used by `main_cli_text.py`).
>   Using `llava` for text often yields empty or unstructured answers.

---

## 3) (Optional) Cloud APIs

If you also want to try cloud LLMs, add keys to `BigModel/api_keys.yaml`:

```yaml
openai:
  api_key: 'YOUR_KEY'
chatglm:
  api_key: 'YOUR_KEY'
```

*(Ollama does not require this.)*

---

## 4) Prepare Datasets

We evaluate across several domains:

| Dataset   | Domain         | Source                                                             |
| --------- | -------------- | ------------------------------------------------------------------ |
| UCR       | Industry       | [Paper](https://arxiv.org/abs/2009.13807)                          |
| NASA-SMAP | Industry       | [DOI](https://doi.org/10.1145/3219819.3219845)                     |
| NASA-MSL  | Industry       | [DOI](https://doi.org/10.1145/3219819.3219845)                     |
| NormA     | Industry       | [DOI](https://doi.org/10.1007/s00778-021-00655-8)                  |
| SMD       | Web service    | [DOI](https://doi.org/10.1145/3292500.3330672)                     |
| Dodgers   | Transportation | [UCI](https://archive.ics.uci.edu/dataset/157/dodgers+loop+sensor) |
| ECG       | Health care    | [DOI](https://doi.org/10.14778/3529337.3529354)                    |
| Synthetic | —              | Google Drive link in original README                               |

Directory layout (recommended):

```
./data/
├── UCR
│   ├── 135_labels.npy
│   ├── 135_test.npy
│   └── ...
└── ...
```

**If this is your first time on a dataset (no prior EH_GAM_EGAN artifacts):**

```bash
python3 -m Datasets.Dataset
```

Convert sequences to **image** and **text** windows (example: UCR, 600/200):

```bash
# IMAGE modality
python3 make_dataset.py --dataset UCR --mode train --modality image --window_size 600 --stride 200
python3 make_dataset.py --dataset UCR --mode test  --modality image --window_size 600 --stride 200

# TEXT modality
python3 make_dataset.py --dataset UCR --mode train --modality text  --window_size 600 --stride 200
python3 make_dataset.py --dataset UCR --mode test  --modality text  --window_size 600 --stride 200
```

---

## 5) Run

### A) Image pipeline (works well)

```bash
# Ensure .env uses: OLLAMA_MODEL=llava
python3 main_cli.py --dataset UCR --normal_reference 3 --LLM ollama
```

Logs are saved to:

```
./log/<subtask>/UCR_ollama_image_log.yaml
```

### B) Text pipeline (works, but still being tuned)

```bash
# Ensure .env uses: OLLAMA_MODEL=mistral:instruct
python3 main_cli_text.py --dataset UCR --normal_reference 1 --LLM ollama
```

Logs are saved to:

```
./log/<subtask>/UCR_ollama_text_log.yaml
```

#### One-sample smoke test

```bash
python3 main_cli_text.py --dataset UCR --data_id_list 138 --normal_reference 1 --LLM ollama
```

> **Note**: If you see empty `normal_pattern` in the text logs, double-check that `OLLAMA_MODEL` is set to an instruction-tuned **text** model (e.g., `mistral:instruct`). Using `llava` for text commonly causes this.

> **Note**: The current main_cli_text is not fully set up yet, so you might see weird/random looking model output that does not detect anything
---

## 6) Results Analysis

The current `evaluation.py` and `ablation_eval.py` expect a generic filename (e.g., `UCR_log.yaml`).
Until we unify naming, do **one** of the following:

* **Rename/symlink** the produced log:

  ```bash
  # for image
  ln -sf log/UCR_ollama_image_log.yaml log/UCR_log.yaml
  # for text
  ln -sf log/UCR_ollama_text_log.yaml  log/UCR_log.yaml
  ```

* **OR** edit the expected path in `Datasets/Dataset.py` / `evaluation.py` / `ablation_eval.py` to match
  `UCR_ollama_image_log.yaml` or `UCR_ollama_text_log.yaml`.

Then run:

```bash
# evaluation
python3 evaluation.py

# ablation study
python3 ablation_eval.py
```

> **Status:** The text pipeline still needs minor tuning in `main_cli_text.py`.
> `evaluation.py` / `ablation_eval.py` aren’t yet wired to auto-detect the `*_text_log.yaml`/ `*_image_log.yaml`name—use the symlink above.

---

## 7) Troubleshooting

* **WSL + Ollama connection**
  If `requests` time out or you see `Invalid request` messages, confirm:

  ```bash
  echo $OLLAMA_HOST
  curl $OLLAMA_HOST/api/tags
  ```

  Try `http://localhost:11434` first; if that fails from WSL, use the Windows host bridge IP.

* **Model choice matters**

  * Image pipeline → `llava`
  * Text pipeline → `mistral:instruct` (recommended). Small/base models often ignore strict JSON.

* **Empty JSON fields**
  Ensure the model is instruction-tuned. If needed, lower `temperature` in `main_cli_text.py`.

---