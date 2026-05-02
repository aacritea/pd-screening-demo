---
title: PD Screening Demo
emoji: 🧠
colorFrom: blue
colorTo: red
sdk: docker
app_file: app.py
pinned: false
---

# Parkinson's Disease Multimodal Screening — Interactive Research Demo

> **⚠️ Research demonstration only. Not a clinical tool.**

Interactive Streamlit app demonstrating the attention-based multimodal fusion framework from:

> *"Attention-Based Multimodal Fusion of Voice and Gait for Parkinson's Disease Detection"*  
> Aakriti Jain, Ujjawal Gaur, Pragya Singh — ICESAIA 2026 *(Under Review, IEEE)*

**[Live Demo →](https://huggingface.co/spaces/aacritea/pd-screening-demo)**  
*(Replace with your actual HF Spaces URL after deployment)*

---

## What This App Does

| Page | Description |
|---|---|
| 🧠 About the Research | Paper summary, architecture, performance comparison |
| 🎙️ Voice-Only Demo | Run inference using 22 UCI acoustic features (CSV or sliders) |
| 🚶 Gait-Only Demo | Run inference on PhysioNet VGRF signals (upload or sample) |
| 🔀 Multimodal Fusion | Full attention-based fusion with attention weight visualization + dropout simulation |

The signature feature is the **per-sample attention weight visualization** — showing how the model dynamically distributes focus between voice and gait for each individual input.

---

## Key Results (from the paper)

| Method | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Voice-Only | 86.67% | 0.967 | 0.977 |
| Gait-Only | 89.13% | 0.927 | 0.991 |
| Concatenation Fusion | 90.28% | 0.932 | 0.957 |
| **Attention Fusion (Ours)** | **92.26%** | **0.942** | **0.986** |

---

## Project Structure

```
pd-screening-demo/
├── app.py                    # Main Streamlit app (4 pages)
├── models/
│   ├── architecture.py       # Model definitions (matches paper exactly)
│   ├── __init__.py
│   ├── voice_encoder.pth     # ← NOT in repo, copy manually
│   ├── gait_encoder.pth      # ← NOT in repo, copy manually
│   ├── attention_module.pth  # ← NOT in repo, copy manually
│   ├── classifier_head.pth   # ← NOT in repo, copy manually
│   ├── scaler_voice.pkl      # ← NOT in repo, copy manually
│   └── scaler_gait.pkl       # ← NOT in repo, copy manually
├── utils/
│   ├── model_loader.py       # Cached weight loading
│   ├── preprocessing.py      # Voice + gait preprocessing pipelines
│   ├── inference.py          # Model forward pass wrappers
│   └── ui_components.py      # Plotly charts + result cards
├── samples/
│   ├── sample_pd_gait.npy    # Optional: public PhysioNet test sample
│   └── sample_healthy_gait.npy
├── export_weights.py         # Run in private training repo to export weights
├── export_samples.py         # Run in private training repo to export samples
├── requirements.txt
├── .gitignore                # Keeps .pth and .pkl out of git
└── README.md
```

---

## Setup (Local)

```bash
git clone https://github.com/aacritea/pd-screening-demo
cd pd-screening-demo
pip install -r requirements.txt
```

### Adding Model Weights (required for live inference)

The trained weights live in your **private** training repository.
Use the provided export script there:

```bash
# Inside your PRIVATE training repo:
python export_weights.py \
    --checkpoint path/to/best_checkpoint.pt \
    --out /path/to/pd-screening-demo/models
```

Then optionally export sample files:

```bash
python export_samples.py \
    --out /path/to/pd-screening-demo/samples
```

The weight files are in `.gitignore` and will **never** be committed to this repo.

### Running

```bash
streamlit run app.py
```

The app runs in **demo mode** if weight files are absent — showing
placeholder outputs so the UI is always functional.

---

## Deploy to Hugging Face Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
   - SDK: **Streamlit**
   - Visibility: Public

2. Push this repo:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/pd-screening-demo
   git push hf main
   ```

3. Upload weights via the HF web UI or CLI (**not via git**):
   ```bash
   pip install huggingface_hub
   python - <<'EOF'
   from huggingface_hub import HfApi
   api = HfApi()
   for f in ["voice_encoder.pth", "gait_encoder.pth",
             "attention_module.pth", "classifier_head.pth",
             "scaler_voice.pkl", "scaler_gait.pkl"]:
       api.upload_file(
           path_or_fileobj=f"models/{f}",
           path_in_repo=f"models/{f}",
           repo_id="YOUR_HF_USERNAME/pd-screening-demo",
           repo_type="space",
       )
   EOF
   ```

   HF Spaces stores these as Space files, not in git history.

---

## Datasets Used (Public)

- **UCI Parkinson's Voice Dataset** — Little (2007), DOI: 10.24432/C59C74  
- **PhysioNet Gait in Parkinson's Disease** — Goldberger et al. (2000)

---

## Citation

If you use this demo or the underlying research:

```
@inproceedings{jain2026parkinson,
  title     = {Attention-Based Multimodal Fusion of Voice and Gait for Parkinson's Disease Detection},
  author    = {Jain, Aakriti and Gaur, Ujjawal and Singh, Pragya},
  booktitle = {ICESAIA 2026},
  year      = {2026},
  note      = {Under Review, IEEE}
}
```

---

## Disclaimer

This application is a research demonstration. It is not approved for clinical use
and must not be used for medical diagnosis or treatment decisions.
