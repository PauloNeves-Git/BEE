from sklearn.metrics import confusion_matrix, r2_score
import json
import time
import math
import numpy as np

def split_data(df, n):
    """
    Partitions a dataframe into n equal sized blocks, outputs a list with respective blocks
    """
    block_size = math.ceil(len(df)/n)
    return [df.iloc[i:i + block_size] for i in range(0, len(df), block_size)]

def convert_strlst2lstfloat(stringlist):
    """
    Function to use when after loading a list in a dataframe ast.literal_eval() is not capable of processing syntax of recorded list
    """
    equivalents_float = []
    equivalents_str2list = stringlist.split(",")
    try:
        for i, element in enumerate(equivalents_str2list):
            if i == 0 and element[1:] == "nan":
                equivalents_float.append(-10)
            elif i == 0 and element[1:] != "nan":
                equivalents_float.append(float(element[1:]))
            elif i == len(equivalents_str2list)-1 and element[1:] == "nan]":
                equivalents_float.append(-10)
            elif i == len(equivalents_str2list)-1 and element[1:] != "nan]":
                equivalents_float.append(float(element[1:-1]))
            elif element[1:] == "nan":
                equivalents_float.append(-10)
            elif element[1:] != "nan":
                equivalents_float.append(float(element[1:]))
    except:
        equivalents_float = np.nan
    return equivalents_float

def replace_sep(row):
    """
    Replace token . used to indicate an ionic bond by ^ to differentiate from the token . used to separate molecules
    """
    separators = []
    ordered_reactant_ids = row["reactant_ids_order"]
    for i in range(len(ordered_reactant_ids)-1):
        if int(ordered_reactant_ids[i]) == int(ordered_reactant_ids[i+1]):
            separators.append("^")
        else:
            separators.append(".")
        
    smiles_sep = [row["reaction_smiles_std"].split(".")[0]]
    for mol, sep in zip(row["reaction_smiles_std"].split(".")[1:], separators):
        smiles_sep.append(sep)
        smiles_sep.append(mol)       
        
    return ''.join(smiles_sep)

def add_photocat_react_ids(row, ID2ROLE):
    """
    Adds the photocalyst ids to the ordered reactant ids list if they are missing.
    """
    reactant_ids_order_size = len(row.reactant_ids_order)
    reactant_ids_order = row.reactant_ids_order
    intended_id = int(list(ID2ROLE.keys())[list(ID2ROLE.values()).index("photocalysts")])
    if intended_id not in row.reactant_ids_order:
        for i in range(row.reaction_smiles_std.count(".")+1- reactant_ids_order_size):
            reactant_ids_order.append(intended_id+i*0.1)
    return reactant_ids_order

def nested_cross_validation(df_photoredox, col_interest, num_splits, base_model):

    for i in range(num_splits):
        photoredox_splitted = split_data(df_photoredox, num_splits)
        val_test = photoredox_splitted.pop(i)
        train = pd.concat(photoredox_splitted)
        val = val_test.iloc[int(len(val_test)/2):]
        test = val_test.iloc[:int(len(val_test)/2)]

        dropout = 0.75
        l_r = 0.00008
        model_args = {"save_steps": 10000, "save_epochs": 10,
                        'num_train_epochs': 160, 'overwrite_output_dir': True,
                        'learning_rate': l_r, 'gradient_accumulation_steps': 1,
                        'regression': True, "num_labels":1, "fp16": False,
                        "evaluate_during_training": True, 'manual_seed': 42,
                        "max_seq_length": 350, "train_batch_size": 16*4,"warmup_ratio": 0.00,
                        "config" : {'hidden_dropout_prob': dropout}}

        model_path = f"../reaction_transformers/pretrained_models/transformers/bert_{base_model}"
        if "equivalents" in col_interest:
            eqv_option = "eqvs"
        else:
            eqv_option = "NOeqvs"
        output_dir = f"../finetuned_models/{base_model}_photoredox_{eqv_option}_ncv_{i}"

        pretrained_bert = SmilesClassificationModel("bert", model_path, num_labels=1, args=model_args, use_cuda=torch.cuda.is_available())
        pretrained_bert.train_model(train[col_interest], output_dir=output_dir, show_running_loss=False, eval_df=val[col_interest])

def moving_avg(signal, sig_range):
    """
    Smoothing function to visualy analyze convergence plots with less noise
    """
    #used such that selection of epoch in early stopping is less suspectible to noise
    return [np.mean(signal[i:i+sig_range]) for i in range(len(signal)-sig_range+1)]

def epochs_calc_r2(df_test, model_dir, checkpoit_multiple=22, epoch_offset = 0, final_epoch = 200, stride = 10):
    r2_scores = []
    epochs = []
    for i in range(epoch_offset, final_epoch+stride, stride):
        model_path = model_dir + f"/checkpoint-{i*checkpoit_multiple}-epoch-{i}"
        trained_yield_bert = SmilesClassificationModel('bert', model_path, num_labels=1, args={"regression": True}, use_cuda=torch.cuda.is_available())
        if "equivalents" in df_test.columns:
            print("Equivalents used.")
            lcms_predicted = trained_yield_bert.predict(df_test.text.values, equivalents = df_test.equivalents.values)[0]
        else:
            lcms_predicted = trained_yield_bert.predict(df_test.text.values)[0]

        lcms_true = df_test.labels.values
        epochs.append(f"epoch_{i}")
        r2_scores.append(r2_score(lcms_true, lcms_predicted))
    r2_scores = {k:v for k,v in zip(epochs, r2_scores)}
    return r2_scores

def r2score_stats(r2_scores_all):
    """
    Combine results from each nested cross validation split and calculate std for the results plot
    """
    r2_scores_all_avg = []
    r2_scores_all_min = []
    r2_scores_all_max = []
    r2_scores_all_std = []

    for i in range(len(list(r2_scores_all.values())[0])):
        all_scores = [r2_scores_all[key][list(r2_scores_all[key].keys())[i]] for key in r2_scores_all]
        r2_scores_all_avg.append(np.mean(all_scores))
        r2_scores_all_min.append(np.min(all_scores))
        r2_scores_all_max.append(np.max(all_scores))
        r2_scores_all_std.append(np.std(all_scores))

    return r2_scores_all_avg, r2_scores_all_min, r2_scores_all_max, r2_scores_all_std

def save_ncv_testresults(df_photoredox, col_interest, num_splits, base_model):
    """
    Save nested cross validation r2 score results into a json file
    """
    r2_scores_all = {}
    max_r2_scores_all = {}
    for i in range(num_splits):
        photoredox_splitted = split_data(df_photoredox, num_splits)
        val_test = photoredox_splitted.pop(i)
        test = val_test.iloc[:int(len(val_test)/2)]
        
        if "equivalents" in col_interest:
            eqv_option = "eqvs"
        else:
            eqv_option = "NOeqvs"
        model_dir = f"../finetuned_models/{base_model}_photoredox_{eqv_option}_ncv_{i}"
        r2_scores = epochs_calc_r2(test[col_interest], model_dir, checkpoit_multiple=22, epoch_offset = 10, final_epoch = 160, stride = 10)
        r2_scores_all[f"ncv_{i}"] = r2_scores
        max_r2_scores_all[f"ncv_{i}"] = max(moving_avg(list(r2_scores.values()), sig_range = 3))

    data = [max_r2_scores_all, r2_scores_all]
    data_json = json.dumps(data, indent=2)

    with open(f"../results/{base_model}_photoredox_ncv_{eqv_option}_results.json", "w") as outfile:
        outfile.write(data_json)
        
def add_canonical_rxn_smiles(df):
    """
    Generate canonical sorted reaction smiles by canonicalizing each entity separated by a special token
    """
    reactions_standardized = []

    for rxn_separated in df["reaction_smiles_sep"].values:
        rxn_separated = rxn_separated.replace(".","|")
        rxn_separated = rxn_separated.replace("^",".")
        reactants = rxn_separated.split(">>")[0]
        products = rxn_separated.split(">>")[1]
        reactants_canonical = []
        products_canonical = []

        try:
            for reactant in reactants.split("|"):
                reactants_canonical.append(MolToSmiles(MolFromSmiles(reactant)))

            reactants_canonical_sorted = sort_reactants(reactants_canonical)
            reactants_canonical_sorted = ".".join(reactants_canonical_sorted)

            products_canonical = MolToSmiles(MolFromSmiles(products))

            rxn_stdd = reactants_canonical_sorted + ">>" + products_canonical
        except:
            rxn_stdd = "invalid_structure"

        reactions_standardized.append(rxn_stdd)

    df["rxns_canonical_rdkit"] = reactions_standardized

    return df   

def get_folder_names(dir_name, num):
    """
    Get all folder names with epoch values in the name which correspond to a set limit defined by 'num'
    """

    folder_names = {}
    epoch_list = [i for i in range(1,num)]
    for folder in os.listdir(dir_name):
        try:
            if int(folder.split("-")[-1]) in epoch_list:
                folder_names[int(folder.split("-")[-1])] = folder
        except:
            pass
    return folder_names

def sort_reactants(lista):
    list_size=[]   
    for ele in lista:
         list_size.append(len(ele))     
    sortedindex = np.argsort(list_size)  
    listb = [" " for i in range(len(lista))]  

    for i in range(len(lista)):    
        listb[i] = lista[sortedindex[i]]     

    return listb

