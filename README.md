<div align="center">

<img src="fig/logo/ehtp.jpeg" height="70"> &nbsp;&nbsp;&nbsp; <img src="fig/logo/dgm.jpeg" height="70">

# 🛰️ Satellite Convection Labeling

**Real-time segmentation of convective clouds over Morocco from MSG/SEVIRI imagery, using a U-Net.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras%203-orange)
![CI](https://github.com/msaadi15/Satellite_Convection_Labeling/actions/workflows/ci.yml/badge.svg)

<img src="fig/RGB_convection.png" width="32%"> <img src="fig/unet-arc.png" width="32%"> <img src="fig/nombre_episodes_convection_maroc.png" width="32%">

*Convection RGB input · U-Net · 2023 convection-episode map produced by the model*

<img src="fig/unet_architecture.png" width="85%">

*U-Net architecture (Conv 3×3 → BatchNorm → ReLU blocks, skip connections). Input is a 512×512×3 convection RGB image;
output is a 512×512×1 binary mask, resized back to the original scene size.*

</div>

---

## Overview

Manually outlining convective systems on satellite images is slow and subjective. This project trains a
**U-Net** to label convection on the EUMETSAT *convection RGB* composite of MSG/SEVIRI over Morocco
(−20°E…5°E, 20°N…40°N), and ships it as a small, monitored, real-time pipeline.

| Composite channel | SEVIRI difference | Highlights |
|---|---|---|
| Red | WV 6.2 − WV 7.3 µm | Upper-level moisture |
| Green | IR 3.9 − IR 10.8 µm | Small ice particles / strong updrafts |
| Blue | NIR 1.6 − VIS 0.6 µm | Cloud-top phase |

Details of the method, the expert labelling protocol and the comparison with an OpenCV/HSV baseline are in
[`docs/report.pdf`](docs/report.pdf).

## Results (2023, 2 552 scenes vs. expert masks)

| Method | Mean IoU | Mean Dice | Pixel acc. |
|---|---|---|---|
| OpenCV baseline | 0.563 | 0.601 | 0.993 |
| **U-Net** | **0.734** | **0.754** | 0.993 |

Computed from [`metrics/metrics_opencv_unet_vs_expert_2023.csv`](metrics/metrics_opencv_unet_vs_expert_2023.csv).
Scenes with no convection in both prediction and expert mask count as a perfect score, so the mean is
flattered by empty scenes; see the notebooks for the per-scene distributions (Bland-Altman, box plots).

## Pipeline

```mermaid
flowchart LR
    A[EUMETSAT Data Store<br/>MSG HRSEVIRI .nat] --> B[satpy<br/>convection RGB + crop Morocco]
    B --> C[U-Net 512x512x3]
    C --> D[mask + overlay PNG]
    C --> E[monitoring<br/>coverage, brightness, drift]
    D --> F[FastAPI /latest]
    E --> G[(JSONL log / MLflow)]
```

## Repository layout

```
src/convection_seg/   package: satellite, opencv_baseline, comparison, preprocessing, model, realtime, monitoring, evaluate, api, cli
configs/default.yaml  region, model, real-time and monitoring settings
tests/                pytest suite (no TensorFlow needed)
notebooks/            01-05: analysis, training and evaluation notebooks (Colab)
docs/                 report.pdf, main.pdf
fig/                  figures used by the report and this README
metrics/              per-scene OpenCV vs U-Net vs expert metrics
Dockerfile, docker-compose.yml, .github/workflows/ci.yml, Makefile
```

## Quick start

```bash
git clone https://github.com/msaadi15/Satellite_Convection_Labeling.git
cd Satellite_Convection_Labeling
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e . --no-deps
```

### Model weights

The checkpoint (`model_all_best.keras`, ~370 MB) is too large for Git. Put it at
`model/model_all_best.keras` (path configurable in `configs/default.yaml`).

### Segment local images

```bash
convseg predict data/image/prod_2023-01-01T12-00-00.000Z.png --out outputs/predict
```

### Compare U-Net vs OpenCV

```bash
convseg compare data/image/prod_2023-01-01T12-00-00.000Z.png --masks data/mask_manual
convseg opencv data/image outputs/opencv_masks      # OpenCV (HSV) baseline only
```

`compare` writes a panel *RGB | Expert | U-Net | OpenCV | agreement* (green = both, red = U-Net only,
blue = OpenCV only) and prints IoU/Dice between methods and against the expert. `convseg evaluate` reports
both methods side by side, and the real-time loop saves `latest_compare.png` for every scan.

### Real-time segmentation of Morocco

1. Create a free account on the [EUMETSAT Data Store](https://data.eumetsat.int) and export your API key:
   ```bash
   export EUMETSAT_KEY=...  EUMETSAT_SECRET=...      # or JSON file ~/.eumetsat_api_key
   ```
   > Through the API (`POST /realtime`) the server never uses its own credentials: each user sends their
   > own key/secret in the `X-EUMETSAT-Key` / `X-EUMETSAT-Secret` headers (get them at
   > <https://api.eumetsat.int/api-key>). They are used for that request only and are not stored or logged.
   > Use HTTPS in front of the service so headers are encrypted in transit.
2. Run the loop (one cycle per SEVIRI repeat cycle, 15 min):
   ```bash
   convseg realtime          # or: convseg realtime --once
   ```
   Each cycle downloads the newest scan, builds the convection RGB, segments it and writes
   `outputs/realtime/latest_{rgb,mask,overlay}.png` plus timestamped copies.

### API

```bash
convseg serve                                   # http://localhost:8000/docs
curl -F file=@scene.png "localhost:8000/predict?output=overlay" -o overlay.png
curl localhost:8000/latest -o latest.png        # newest product written by `convseg realtime`
# Real-time through the API: every caller uses THEIR OWN EUMETSAT credentials
curl -X POST "localhost:8000/realtime?kind=compare" \
     -H "X-EUMETSAT-Key: <your key>" -H "X-EUMETSAT-Secret: <your secret>" -o realtime.png
curl localhost:8000/stats                       # monitoring records
```

### Docker

```bash
docker compose up --build        # api on :8000 + real-time worker (needs EUMETSAT_KEY/SECRET)
```

## MLOps

| Concern | How |
|---|---|
| Reproducible config | One YAML, any key overridable with `CONVSEG__SECTION__KEY` env vars |
| CI | GitHub Actions: lint (`ruff`), unit tests, Docker build on every push |
| Model quality gate | `convseg evaluate --min-dice 0.5` exits non-zero if mean Dice regresses — usable as a release check |
| Experiment tracking | Set `monitoring.mlflow_enabled: true` to log evaluation runs and per-scene stats to MLflow |
| Production monitoring | Every prediction appends coverage / brightness / probability stats to a JSONL log |
| Drift detection | `convseg make-reference` freezes reference stats; later predictions with \|z\| above the threshold are flagged in the log |
| Deployment | Slim Docker image, weights mounted as a volume, `restart: unless-stopped` worker |

## Tests

```bash
make test      # pytest
make lint      # ruff
```

## Status & limitations

- The real-time path (EUMETSAT download → satpy RGB → U-Net) has been written against the same composite and
  region used for the training images, but needs your EUMETSAT credentials to run end to end; the
  `nat_to_rgb` orientation/cropping should be checked once against a known scene.
- Scenes (about 600×521) are resized to 512×512 for the model and the mask is resized back, so mask edges are slightly interpolated.
- The model input size is read from the checkpoint; if it differs from `model.input_size` in the config, a warning is logged and the checkpoint's size wins.
- Training/evaluation notebooks were written for Google Colab and contain Drive paths.

## Authors

Affiliations (see logos above): EHTP and DGM, Morocco.
Report: [`docs/report.pdf`](docs/report.pdf).
