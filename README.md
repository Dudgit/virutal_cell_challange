# ELTE Virtual Cell Analysis
This is a github repository for virtual cell analysis, to understand protein perturbation effects.

# Guidelines

Please follow the following folder structure:

```text
├── configs/
│   └── x_config.yaml      -> configuration (if needed) for a specific run.
├── data/
│   ├── .gitkeep           -> Please do not upload the training data to github. We will figure out a way to share it.
│   └── traindata/
│       └── *.h5ad
├── logs/                  -> If you are using wandb or tensorboard please log them here  
├── misc/                  -> When I'm using some extra libraries I usually download it here  
├── notebooks/             -> If you are using notebooks you can run them from here. I usually use os.chdir(..) to put myself into the root. 
├── outs/                  -> Output of training and/or nohup outputs.  
├── README.md
├── trainers/              -> Store here different kind of training routines, everything you work on.  
└── utils/                 -> All the extra libraries you want to use.  
│   ├── *templates.py           -> If we want to user unified training routines/dataset routines we can put them into classes here.  
│   └── version1/               -> If you wourk on a specific approach, give it a name and include it here.  
│       └── model.py
│       └── dataset.py
│       └── *extra_utils.py
│   └── version2/
│       └── model.py
│       └── dataset.py
│       └── *extra_utils.py

TODO: More descriptive readme

