
# Deep Learning Assignment 2

Deepfake detection with an EfficientNet-based model. This repo includes a Streamlit app for inference and a training script.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

| Path | Purpose |
| --- | --- |
| app.py | Streamlit UI for running inference |
| model_train_efficientnet.py | Training script |
| model/deepfake_detector.pth | Trained model weights |
| Dataset.md | Dataset notes |
| requirements.txt | Python dependencies |

## Setup

Create a virtual environment and install dependencies.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

## Train the Model

```bash
python model_train_efficientnet.py
```

## Notes

- Ensure model weights exist at `model/deepfake_detector.pth` before running the app.
- Update dataset paths in the scripts if your dataset location differs.