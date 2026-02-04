# FlowLoop ML

PyTorch models and training scripts for emotion detection and attention estimation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Training Pipeline

1. Prepare FER2013 dataset:
   ```bash
   python data/prepare_fer2013.py --output_dir ./data/processed/fer2013
   ```

2. Train EmotionNet:
   ```bash
   python training/train_emotion.py --data_dir ./data/processed/fer2013
   ```

3. Export to ONNX:
   ```bash
   python export/export_onnx.py --model emotion --checkpoint ./outputs/best_model.pt
   ```
