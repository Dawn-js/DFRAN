# Advancing Chinese Travel Sentiment Analysis: A Novel Dataset and DFRAN Approach for Missing Modalities
# Prerequisites

- Python 3.9
- PyTorch 2.0.0
- CUDA 11.8

# Datasets

1. **CTRS Dataset**: You can download it from the following link: [CTRS Dataset](https://www.kaggle.com/datasets/borain/ctrsd).
2. **MVSA Dataset**: Available at: [MVSA Dataset](https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/).

# Training

First, download the required dataset. Once downloaded, you need to update the path to the dataset in the code.

Make sure you have installed all the necessary dependencies. Next, run the following command in the terminal against your dataset:

- For the **CTRS** dataset:

  ```bash
  python train_model_ctrs.py
  
- For the **MVSA** dataset:
  ```bash
  python train_model.py

