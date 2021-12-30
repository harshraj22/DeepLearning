## TabTransformer: [Paper](https://arxiv.org/pdf/2012.06678v1.pdf)


Following project template from [pytorch-template repo](https://github.com/victoresque/pytorch-template)

1. Download the `Blastchar dataset` csv and place it in `data/` directory.
2. Run `python3 train.py`
3. Add arguments using `+` and overwrite arguments using `=`. eg. `python3 train.py params.num_epochs=1 +store='saved/models/v1.pth'`

Directory Structure:
```
├── README.md
├── conf
│   └── config.yaml
├── data
│   └── Telco-Customer-Churn.csv
├── data_loader
│   ├── __pycache__
│   │   ├── datasets.cpython-38.pyc
│   │   └── datasets.cpython-39.pyc
│   └── datasets.py
├── models
│   ├── __pycache__
│   │   ├── transformer.cpython-38.pyc
│   │   ├── transformer.cpython-39.pyc
│   │   ├── transformer_block.cpython-38.pyc
│   │   └── transformer_block.cpython-39.pyc
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
│       └── v2.pth
├── train.py
├── utils
│   ├── __pycache__
│   │   ├── utils.cpython-37.pyc
│   │   ├── utils.cpython-38.pyc
│   │   └── utils.cpython-39.pyc
│   └── utils.py
└── wandb
    ├── debug-internal.log
    ├── debug.log
    ├── latest-run
    │   ├── files
    │   │   ├── conda-environment.yaml
    │   │   ├── config.yaml
    │   │   ├── output.log
    │   │   ├── requirements.txt
    │   │   ├── wandb-metadata.json
    │   │   └── wandb-summary.json
    │   ├── logs
    │   │   ├── debug-internal.log
    │   │   └── debug.log
    │   ├── run-1a6ex8wa.wandb
    │   └── tmp
    │       └── code
    └── run-20211230_231607-1a6ex8wa
        ├── files
        │   ├── conda-environment.yaml
        │   ├── config.yaml
        │   ├── output.log
        │   ├── requirements.txt
        │   ├── wandb-metadata.json
        │   └── wandb-summary.json
        ├── logs
        │   ├── debug-internal.log
        │   └── debug.log
        ├── run-1a6ex8wa.wandb
        └── tmp
            └── code
```

![image](https://user-images.githubusercontent.com/46635452/145711600-22e3ccf9-f45a-49b6-b029-2315e8767b80.png)
