"""
Creates dataset for training. Requires csv files for entire reaction space and attempted reactions.

The dataset is saved as a numpy file with the following keys
    - X: ndarrary of shape [num_data, num_features], features (qualitative features always come in front of quantitative features)
    - Y: ndarrary of shape [num_data], labels (0 for 100 phase, 1 for 110 phase)
    - X_labels: ndarrary of shape [num_features], feature names
    - X_qual_num_classes: ndarrary of shape [num_qual_features], number of classes for each qualitative feature
    - reaction_ids: ndarrary of shape [num_data], reaction names
"""

import os
import sys
import numpy as np
import pandas as pd
import rdkit.Chem
import argparse

sys.path.append("../")
sys.path.append(os.getcwd())

from src.utils.constants import DATA_DIR, RXN_VARS, CATEGORICAL_MAPPINGS

def create_dataset(candidate_reactions_path, attempted_reactions_path, output_path):
    print("Creating dataset from candidate and attempted reactions...")
    print("Candidate reactions path:", candidate_reactions_path)
    print("Attempted reactions path:", attempted_reactions_path)

    # Candidate combinations (space we are exploring)
    candidate_reactions = pd.read_csv(candidate_reactions_path, index_col=0)
    candidate_reactions = candidate_reactions.drop_duplicates()


    # Attempted combinations (space we have already explored, is a subset of candidate combinations)
    attempted_reactions = pd.read_csv(attempted_reactions_path)

    
    # Normalize continuous and discrete variables of candidate reaction space to [0, 1]
    for (data_type, field_name) in RXN_VARS.values():
        if data_type != "categorical": # discrete or continuous
            candidate_reactions[field_name] = ((candidate_reactions[field_name] - candidate_reactions[field_name].min())
                                    / (candidate_reactions[field_name].max() - candidate_reactions[field_name].min()))
            


    attempted_reactions = attempted_reactions[["Metal", "Halide", "Ligand", "Type"]]

    
    # Drop rows with NaN values
    # .dropna(inplace=False, how="all")
    attempted_reactions = attempted_reactions.dropna(inplace=False, how="any", axis=0)

    # Drop rows with Type = 0.0
    attempted_reactions = attempted_reactions[attempted_reactions["Type"] != 0.0]


    # Prepare dataset for training
    all_rows = []
    all_y = []
    reaction_ids = []
    for (index, row) in attempted_reactions.iterrows():
        metal = row["Metal"]
        halide = row["Halide"]
        ligand = row["Ligand"]
        phase = row["Type"]

        # Convert SMILES to canonical form
        ligand = rdkit.Chem.CanonSmiles(ligand)

        # Get reaction name to retrieve info from candidate reactions
        rxn_name = metal + " + " + halide + " + " + ligand
        all_info_row = candidate_reactions.loc[candidate_reactions['Rxn_Name'] == rxn_name]
        reaction_ids.append(rxn_name)

        # Check if there are any duplicates
        try:
            assert len(all_info_row) == 1
        except AssertionError:
            print(rxn_name)
            print(all_info_row)
            print(len(all_info_row))
            print()
            continue

        
        all_rows.append(all_info_row)
        all_y.append(int(phase == 110.0))

    all_x = pd.concat(all_rows, axis=0)
    all_y = np.array(all_y)


    # Deal with qualitative and quantitative features separately (assign class labels to qualitative features)
    all_features = []
    all_feature_names = []
    all_qual_feature_num = []
    # all_precursor_names = []

    index = 0
    for (_, row) in all_x.iterrows():
        qual_features = []
        qual_feature_names = []
        qual_feature_num = []

        quant_features = []
        quant_feature_names = []

        for (key, items) in RXN_VARS.items():
            data_point = row[items[1]]
            if items[0] == "categorical":
                categorical_mapping = CATEGORICAL_MAPPINGS[key]
                class_label = categorical_mapping[data_point]
                num_classes = len(list(categorical_mapping.keys()))
                qual_features.append(class_label)
                
                if index == 0:
                    qual_feature_names.append(key)
                    qual_feature_num.append(num_classes)

            else:
                quant_features.append(data_point)
                
                if index == 0:
                    quant_feature_names.append(key)

        all_features.append(qual_features + quant_features)
        if index == 0:
            all_feature_names = qual_feature_names + quant_feature_names
            all_qual_feature_num = qual_feature_num
        # all_precursor_names = np.array(all_precursor_names)
            
        index += 1


    # Save data dict to numpy file
    dataset_path = os.path.join(output_path, "dataset.npz")
    print("Saving dataset to", dataset_path)
    np.savez(dataset_path, 
             X=all_features, 
             Y=all_y, 
             X_labels=all_feature_names, 
             X_qual_num_classes=all_qual_feature_num,
             reaction_ids=reaction_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset for training. Requires csv files for entire reaction space and attempted reactions.")

    candidate_reactions_path = os.path.join(DATA_DIR, "Reaction_Data_Inputs_021424.csv")
    attempted_reactions_path = os.path.join(DATA_DIR, "2D_Dataset.csv")

    parser.add_argument("--candidate_reactions_path", 
                        type=str, 
                        default=candidate_reactions_path, 
                        help="Path to csv file containing candidate reactions (reaction space).")
    
    parser.add_argument("--attempted_reactions_path",
                        type=str,
                        default=attempted_reactions_path,
                        help="Path to csv file containing attempted reactions (subset of reaction space).")
    
    parser.add_argument("--output_path",
                        type=str,
                        default=DATA_DIR,
                        help="Path to save final dataset to.")
    

    args = parser.parse_args()
    create_dataset(args.candidate_reactions_path, args.attempted_reactions_path, args.output_path)

    print("Done")