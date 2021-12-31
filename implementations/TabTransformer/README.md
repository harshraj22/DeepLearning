## TabTransformer: [Paper](https://arxiv.org/pdf/2012.06678v1.pdf)


Following project template from [pytorch-template repo](https://github.com/victoresque/pytorch-template)

1. Download the `Blastchar dataset` csv[[1](https://www.kaggle.com/blastchar/telco-customer-churn)] and place it in `data/` directory.
2. Run `python3 train.py`
3. Add arguments using `+` and overwrite arguments using `=`. eg. `python3 train.py params.num_epochs=1 +store='saved/models/v1.pth'`. Hyperparameter management has been done using [Hydra](https://www.youtube.com/watch?v=tEsPyYnzt8s).

See training charts and logs on [weights and biases](https://wandb.ai/harshraj22/Tab-Transformer?workspace=user-harshraj22).              
<strong>Note:</strong> The model was trained on CPU. Deviations from original paper mainly include differences in hyperparameters, LR Schedular, Optimizer etc. The model structure is same as proposed in the original paper.

Directory Structure:
```
├── README.md
├── conf
│   └── config.yaml
├── data
│   └── Telco-Customer-Churn.csv
├── data_loader
│   ├── __pycache__
│   └── datasets.py
├── models
│   ├── __pycache__
│   ├── transformer.py
│   └── transformer_block.py
├── outputs
│   └── 2021-12-30
│       ├── 22-24-21
│       │   └── train.log
├── requirements.txt
├── saved
│   └── models
│       ├── v1.pth
├── train.py
├── utils
│   ├── __pycache__
│   └── utils.py
└── wandb
```

![image](https://user-images.githubusercontent.com/46635452/145711600-22e3ccf9-f45a-49b6-b029-2315e8767b80.png)