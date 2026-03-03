# Folsom Image-Modality Physics Pretraining

This repository contains a first-stage pretraining pipeline for sky-image modality in multimodal PV forecasting:

1. Compute solar geometry and clear-sky irradiance from `(latitude, longitude, timestamp)` with `pvlib`.
2. Use a CNN+GRU encoder to predict sequence cloud parameters from fisheye sky images.
3. Use a physics-guided irradiance head to reconstruct measured GHI from:
   - predicted cloud parameters,
   - meteorological sequence,
   - solar features (`azimuth`, `zenith`, `cos_zenith`, `cs_ghi`).
4. Train image encoder with indirect physical supervision (GHI reconstruction).

## Project layout

- `train_image_pretrain.py`: training entry.
- `folsom_pretrain/data.py`: dataset and sequence loader.
- `folsom_pretrain/solar.py`: `pvlib` solar features.
- `folsom_pretrain/models.py`: image encoder + physical head.
- `folsom_pretrain/losses.py`: reconstruction + temporal smoothness loss.
- `configs/pretrain_image.yaml`: runnable config.

## Expected CSV format

`train_csv` and `val_csv` should include at least:

- `timestamp`
- `image_path` (relative to `image_root`)
- `ghi` (measured irradiance target)
- weather columns listed in `weather_cols`, e.g.:
  - `air_temperature`
  - `relative_humidity`
  - `wind_speed`

## Ascend NPU notes

This code auto-detects `torch_npu` when `device: npu` or `auto`.

Typical environment setup (example only):

- Install Ascend CANN runtime and compatible `torch` + `torch_npu`.
- Verify NPU visibility before training.

Run:

```bash
python train_image_pretrain.py --config configs/pretrain_image.yaml
```

Checkpoints:

- `outputs/pretrain_image/last.pt`
- `outputs/pretrain_image/best.pt`

## Next integration stage

After this pretraining stage, you can export `encoder` or sequence cloud parameters and feed them into your multimodal fusion model.
