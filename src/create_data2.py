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




def parse_data_files(candidate_reactions_path, attempted_reactions_path):
    """
    Takes in the paths to the candidate and attempted reactions csv files and returns the parsed dataframes.
    The bulk of the work is filling in the missing columns of the attempted_reactions dataframe with the 
    info from the candidate_reactions dataframe. None of the values are changed from the original data.

    Returns:
        labelled_reactions: Dataframe of reactions that have been attempted (classified as 100 or 110)
            - Has extra column for 'perovskite_type'
        unlabelled_reactions: Dataframe of reactions that have not been attempted (unclassified)
        total_reactions: Dataframe of all reactions (combined)
    """
    print("Parsing data files...")
    candidate_reactions = pd.read_csv(candidate_reactions_path, index_col=0)
    # Drop duplicates
    candidate_reactions = candidate_reactions.drop_duplicates()

    attempted_reactions = pd.read_csv(attempted_reactions_path)
    attempted_reactions = attempted_reactions[["Metal", "Halide", "Ligand", "Type"]]
    # Drop rows with NaN values
    attempted_reactions = attempted_reactions.dropna(inplace=False, how="any", axis=0)
    # Drop rows with Type = 0.0
    attempted_reactions = attempted_reactions[attempted_reactions["Type"] != 0.0]
    # Drop duplicates
    attempted_reactions = attempted_reactions.drop_duplicates()

    # Change column name of attempted_reactions to match candidate_reactions
    attempted_reactions = attempted_reactions.rename(
        columns={"Metal": "Ion", "Ligand": "SMILES", "Type": "perovskite_type"}
    )
    # Add Rxn_Name column to attempted_reactions
    attempted_reactions.insert(0, "Rxn_Name", "")

    # Add all other columns to attempted_reactions that is not in candidate_reactions
    for col in candidate_reactions.columns:
        if col not in attempted_reactions.columns:
            attempted_reactions[col] = np.nan

    # Fill in Rxn_Name of attempted_reactions
    for i, row in attempted_reactions.iterrows():
        metal = row["Ion"]
        halide = row["Halide"]
        ligand = row["SMILES"]

        # Convert SMILES to canonical form
        ligand = rdkit.Chem.CanonSmiles(ligand)

        # Get reaction name to retrieve info from candidate reactions
        rxn_name = metal + " + " + halide + " + " + ligand
        attempted_reactions.at[i, "Rxn_Name"] = rxn_name
    
    # Fill in the rest of the columns of attempted_reactions with info from candidate_reactions
    for i, row in attempted_reactions.iterrows():
        rxn_name = row["Rxn_Name"]
        candidate_row = candidate_reactions[candidate_reactions["Rxn_Name"] == rxn_name]

        for col in candidate_row.columns:
            attempted_reactions.at[i, col] = candidate_row[col].values[0]
    
    labelled_reactions = attempted_reactions
    unlabelled_reactions = candidate_reactions[~candidate_reactions["Rxn_Name"].isin(labelled_reactions["Rxn_Name"])]
    total_reactions = candidate_reactions

    unlabelled_reactions = unlabelled_reactions.drop_duplicates()
    labelled_reactions = labelled_reactions.drop_duplicates()
    total_reactions = total_reactions.drop_duplicates()

    return labelled_reactions, unlabelled_reactions, total_reactions



def normalize_data(labelled_reactions, unlabelled_reactions, total_reactions):
    """
    Takes in the labelled, unlabelled, and total reactions dataframes and modifies values to be able to be
    used in training and inference. This includes normalizing quantitative features to [0, 1] and converting
    categorical features to numerical values.

    Returns labelled_data and unlabelled_data in the form of dictionaries with the following keys:
        X: X data for attempted reactions (classified as 100 or 110)
            - ndarrays of shape [num_data, num_features]
            - qualitative features always come in front of quantitative features
            
        X_labels: List of strings, feature names
            - qualitative features always come in front of quantitative features

        X_qual_num_classes: List of integers, number of classes for each qualitative feature

        reaction_ids: List of strings, reaction names

        Y: Y data for attempted reactions (classified as 0 for 100 and 1 for 110)
            - only for labelled_data
    """
    print("Preparing data for model...")
    labelled_X_quant = []
    labelled_X_qual = []

    unlabelled_X_quant = []
    unlabelled_X_qual = []

    X_quant_labels = []
    X_qual_labels = []
    X_qual_num_classes = []

    # 'feature' is the name used in src.utils.constants, 'feature_name' is the name used in the dataframe 
    for feature, (feature_type, feature_name) in RXN_VARS.items():
        if feature_type == "discrete" or feature_type == "continuous":
            # Normalize quantitative features to [0, 1]
            min_value = total_reactions[feature_name].min()
            max_value = total_reactions[feature_name].max()

            labelled_feature_values = labelled_reactions[feature_name].values 
            unlabelled_feature_values = unlabelled_reactions[feature_name].values
            
            labelled_feature_values = (labelled_feature_values - min_value) / (max_value - min_value)
            unlabelled_feature_values = (unlabelled_feature_values - min_value) / (max_value - min_value)
    
            labelled_X_quant.append(labelled_feature_values)
            unlabelled_X_quant.append(unlabelled_feature_values)
            X_quant_labels.append(feature_name)

        else: # feature_type == "categorical"
            labelled_feature_values = labelled_reactions[feature_name].values
            unlabelled_feature_values = unlabelled_reactions[feature_name].values

            labelled_feature_values = [CATEGORICAL_MAPPINGS[feature][val] for val in labelled_feature_values]
            unlabelled_feature_values = [CATEGORICAL_MAPPINGS[feature][val] for val in unlabelled_feature_values]
            
            labelled_X_qual.append(labelled_feature_values)
            unlabelled_X_qual.append(unlabelled_feature_values)
            X_qual_labels.append(feature_name)

            X_qual_num_classes.append(len(CATEGORICAL_MAPPINGS[feature]))


    labelled_X = np.transpose(labelled_X_qual + labelled_X_quant)
    unlabelled_X = np.transpose(unlabelled_X_qual + unlabelled_X_quant)

    X_labels = X_qual_labels + X_quant_labels

    labelled_Y = labelled_reactions["perovskite_type"].values
    labelled_Y = np.where(labelled_Y == 100, 0, 1)

    labelled_reaction_ids = list(labelled_reactions["Rxn_Name"].values)
    unlabelled_reaction_ids = list(unlabelled_reactions["Rxn_Name"].values)

    labelled_data = {
        "X": labelled_X,
        "Y": labelled_Y,
        "X_labels": X_labels,
        "X_qual_num_classes": X_qual_num_classes,
        "reaction_ids": labelled_reaction_ids
    }

    unlabelled_data = {
        "X": unlabelled_X,
        "X_labels": X_labels,
        "X_qual_num_classes": X_qual_num_classes,
        "reaction_ids": unlabelled_reaction_ids
    }

    return labelled_data, unlabelled_data


def main(args):
    # Candidate combinations (space we are exploring)
    candidate_reactions_path = args.candidate_reactions_path

    # Attempted combinations (space we have already explored, is a subset of candidate combinations)
    attempted_reactions_path = args.attempted_reactions_path

    labelled_reactions, unlabelled_reactions, total_reactions = parse_data_files(candidate_reactions_path, attempted_reactions_path)
    labelled_data, unlabelled_data = normalize_data(labelled_reactions, unlabelled_reactions, total_reactions)

    labelled_data_path = os.path.join(args.output_path, "labelled_dataset.npz")
    unlabelled_data_path = os.path.join(args.output_path, "unlabelled_dataset.npz")

    np.savez(labelled_data_path, 
                X=labelled_data['X'], 
                Y=labelled_data['Y'], 
                X_labels=labelled_data['X_labels'], 
                X_qual_num_classes=labelled_data['X_qual_num_classes'], 
                reaction_ids=labelled_data['reaction_ids'])
    
    np.savez(unlabelled_data_path,
                X=unlabelled_data['X'],
                X_labels=unlabelled_data['X_labels'],
                X_qual_num_classes=unlabelled_data['X_qual_num_classes'],
                reaction_ids=unlabelled_data['reaction_ids'])

    print("Done")


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

    main(args)

    