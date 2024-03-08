# Perovskite Phase Prediction

Before running any code, make sure to have the following two files:
* `data/2D_dataset.csv`: full dataset of the candidate reactions
* `data/Reaction_Data_Inputs_021424.csv`: dataset of reactions that were already attempted (subset of the candidate reactions)

## Generate datset
All flags below are set to their default values (equivalent to running `python src/create_data.py`).
```
python src/create_data3.py --candidate_reactions_path=data/2D_Dataset_030424.csv --attempted_reactions_path=data/Reaction_Data_Inputs_030424.csv --output_path=data/
```

## Train a single model
All flags below are set to their default values (equivalent to running `python src/train.py`).
```
python src/train.py --data_path=data/labelled_dataset.npz --train_ratio=0.8 --seed=42 --num_epochs=400 --lr=3e-4 --conf_mat=False
```

## Train an ensemble of models
