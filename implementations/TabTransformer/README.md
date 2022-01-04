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
│   └── config.yaml         <-- config file
├── data                    <-- data directory  (for data files)
│   └── Telco-Customer-Churn.csv
├── data_loader
│   ├── __pycache__
│   └── datasets.py
├── model_interpret.py       <-- model interpret script to understand input features influence on model's output
├── models                   <-- model directory  (for model files)
│   ├── __pycache__
│   ├── transformer.py
│   └── transformer_block.py
├── outputs                  <-- output directory (created by Hydra) 
├── requirements.txt
├── saved
│   └── models
│       └── 400epochs.pth    <-- saved model weights
├── train.py                 <-- training script
├── utils
│   ├── __pycache__
│   └── utils.py
└── wandb                    <-- Logging experiments to weights and biases
```

### Checks
- [x] Easy to setup repo, with well structured directory partition
- [x] Clean and readable codes
- [x] Unit tests
- [ ] Augmentations, EarlyStopping
- [ ] Reproducable results (links to weights of the model)
- [x] Configuration management using well tested tools like Hydra
- [x] wandb for tracking experiments
- [x] captum for model interpretability

![image](https://user-images.githubusercontent.com/46635452/145711600-22e3ccf9-f45a-49b6-b029-2315e8767b80.png)
