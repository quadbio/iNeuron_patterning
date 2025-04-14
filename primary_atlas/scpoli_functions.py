##################################################################################################
#########################################scPoli functions#########################################

# LOAD LIBRARIES
import sys
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from scarches.models.scpoli import scPoli
import wandb
import random
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import scvi
scvi.settings.seed = 0

##################################################################################################
#########################################scPoli functions#########################################

#TRAINING

def train_scpoli_model(source_adata,condition_key,cell_type_key,
                       hid_layer_size, lat_dim, emb_dim,
                       nepoch,epoch_ratio,eta,alpha_epoch_anneal,
                       file_path,
                       early_stopping_kwargs,
                       prototype_training=True,
                       unlabeled_prototype_training=True,
                       recon_loss='nb'):
    #mse', 'nb' or 'zinb for recon_loss
    
    scpoli_model = scPoli(
        adata=source_adata, #normally use source adata
        condition_keys=condition_key,
        cell_type_keys=cell_type_key,
        hidden_layer_sizes=hid_layer_size,
        latent_dim=lat_dim,
        embedding_dims=emb_dim,
        recon_loss=recon_loss,
    )


    scpoli_model.train(
        n_epochs=nepoch,
        pretraining_epochs=int(epoch_ratio*nepoch),
        early_stopping_kwargs=early_stopping_kwargs,
        eta=eta,
        prototype_training=True,
        unlabeled_prototype_training=True,
        alpha_epoch_anneal=alpha_epoch_anneal
    )

    import pickle
    # Open the file in binary write mode and save the object using pickle.dump
    with open(file_path, 'wb') as file:
        pickle.dump(scpoli_model, file)
    return(scpoli_model)


def score_scpoli_model(scpoli_model,
                       target_adata,
                       cell_type_key,
                       file_path_query,
                      file_path_classification):
    scpoli_query = scPoli.load_query_data(
        adata=target_adata,
        reference_model=scpoli_model,
        labeled_indices=[],
    )

    scpoli_query.train(
        n_epochs=50,
        pretraining_epochs=40,
        eta=10
    )

    results_dict = scpoli_query.classify(target_adata, scale_uncertainties=True)

    for i in range(len(cell_type_key)):
        preds = results_dict[cell_type_key]["preds"]
        results_dict[cell_type_key]["uncert"]
        classification_df = pd.DataFrame(
            classification_report(
                y_true=target_adata.obs[cell_type_key],
                y_pred=preds,
                output_dict=True)).transpose()


    # Specify the file path where you want to save the object
    import pickle
    # Open the file in binary write mode and save the object using pickle.dump
    with open(file_path_query, 'wb') as file:
        pickle.dump(scpoli_query, file)
    classification_df.to_csv(file_path_classification,sep="\t")

    acc_f1 = classification_df.loc['accuracy','f1-score']
    mavg_f1 = classification_df.loc['macro avg','f1-score']
    wavg_f1 = classification_df.loc['weighted avg','f1-score']

    return(scpoli_query,acc_f1,mavg_f1,wavg_f1,results_dict)


def get_latent_scpoli(scpoli_query,
                      target_adata,
                      source_adata,
                      cell_type_key,
                      results_dict,
                      file_path):

    #get latent representation of reference data
    scpoli_query.model.eval()
    data_latent_source = scpoli_query.get_latent(
        source_adata,
        mean=True
    )

    adata_latent_source = sc.AnnData(data_latent_source)
    adata_latent_source.obs = source_adata.obs.copy()

    #get latent representation of query data
    data_latent= scpoli_query.get_latent(
        target_adata,
        mean=True)
    
    adata_latent = sc.AnnData(data_latent)
    adata_latent.obs = target_adata.obs.copy()


    #get label annotations
    adata_latent.obs['cell_type_pred'] = results_dict[cell_type_key]['preds'].tolist()
    adata_latent.obs['cell_type_uncert'] = results_dict[cell_type_key]['uncert'].tolist()
    adata_latent.obs['classifier_outcome'] = (
        adata_latent.obs['cell_type_pred'] == adata_latent.obs[cell_type_key]
    )

    #get prototypes
    labeled_prototypes = scpoli_query.get_prototypes_info()
    labeled_prototypes.obs['study'] = 'labeled prototype'
    unlabeled_prototypes = scpoli_query.get_prototypes_info(prototype_set='unlabeled')
    unlabeled_prototypes.obs['study'] = 'unlabeled prototype'

    #join adatas
    adata_latent_full = adata_latent_source.concatenate(
        [adata_latent, labeled_prototypes, unlabeled_prototypes],
        batch_key='query'
    )
    adata_latent_full.obs['cell_type_pred'][adata_latent_full.obs['query'].isin(['0'])] = np.nan
    sc.pp.neighbors(adata_latent_full, n_neighbors=15)
    sc.tl.umap(adata_latent_full)

    adata_latent_full.write_h5ad(file_path)
    return(adata_latent_full)
    
    
    
##################################################################################################
#########################################scPoli functions#########################################

#MAPPING
def prepare_adata(adata,primary_genes,training_genes):
    import anndata as ad

    common_genes = list(set(primary_genes)&set(adata.var_names))
    print("genes common in adata and in model")
    print(len(common_genes))
    missing_genes = [x for x in primary_genes if x not in common_genes]
    print("genes missing in adata from model")
    print(len(missing_genes))
    
    meta_bk = adata.obs.copy()
    print('replacing missing genes with 0')
    # If there are missing genes, add them with values of 0
    if missing_genes:
        # Create a matrix of zeros to add to the new AnnData
        zero_values = np.zeros((adata.shape[0], len(missing_genes)))

        # Create an AnnData object for the missing genes with zero values
        missing_genes_adata = ad.AnnData(zero_values, obs=adata.obs, var=pd.DataFrame(index=missing_genes))

        # Concatenate the new AnnData with the missing genes to the subset
        adata = ad.concat([adata, missing_genes_adata], axis=1)

    adata = adata[:,primary_genes]
    adata.obs = meta_bk.loc[adata.obs_names]
    
    print('log normalizing adata on primary genes')
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    print('subsetting adata to training genes')
    adata = adata[:,training_genes]

    return(adata)

def map_data(target_adata,scpoli_model, condition_key,cell_type_key):
    print('adding mapping keys to adata')
    target_adata.obs[cell_type_key] = 'unknown'
    target_adata.obs[condition_key] = 'target'
    target_adata.X = target_adata.X.astype(np.float32)
    
    scpoli_query = scPoli.load_query_data(
        adata=target_adata,
        reference_model=scpoli_model,
        labeled_indices=[],
        freeze=True,
        freeze_expression=True,
    )
        
    scpoli_query.train(
        n_epochs=50,
        pretraining_epochs=40,
        eta=10
    )
    
    results_dict = scpoli_query.classify(target_adata, scale_uncertainties=True)

    return(scpoli_query,results_dict)


def get_latent_scpoli(scpoli_query,
                      target_adata,
                      source_adata,
                      cell_type_key,
                      results_dict,
                      file_path):

    #get latent representation of reference data
    scpoli_query.model.eval()
    data_latent_source = scpoli_query.get_latent(
        source_adata,
        mean=True
    )

    adata_latent_source = sc.AnnData(data_latent_source)
    adata_latent_source.obs = source_adata.obs.copy()

    #get latent representation of query data
    data_latent= scpoli_query.get_latent(
        target_adata,
        mean=True)
    
    adata_latent = sc.AnnData(data_latent)
    adata_latent.obs = target_adata.obs.copy()


    #get label annotations
    adata_latent.obs['cell_type_pred'] = results_dict[cell_type_key]['preds'].tolist()
    adata_latent.obs['cell_type_uncert'] = results_dict[cell_type_key]['uncert'].tolist()
    adata_latent.obs['classifier_outcome'] = (
        adata_latent.obs['cell_type_pred'] == adata_latent.obs[cell_type_key]
    )

    #get prototypes
    labeled_prototypes = scpoli_query.get_prototypes_info()
    labeled_prototypes.obs['study'] = 'labeled prototype'
    unlabeled_prototypes = scpoli_query.get_prototypes_info(prototype_set='unlabeled')
    unlabeled_prototypes.obs['study'] = 'unlabeled prototype'

    #join adatas
    adata_latent_full = adata_latent_source.concatenate(
        [adata_latent, labeled_prototypes, unlabeled_prototypes],
        batch_key='query'
    )
    adata_latent_full.obs['cell_type_pred'][adata_latent_full.obs['query'].isin(['0'])] = np.nan
    sc.pp.neighbors(adata_latent_full, n_neighbors=15)
    sc.tl.umap(adata_latent_full)

    adata_latent_full.write_h5ad(file_path)
    return(adata_latent_full)


##################################################################################################
#########################################scPoli functions#########################################

#PARSING

def convert_age(lst,unit='days',born_date=266):
    print('using '+str(born_date)+' as birth time ('+unit+')')
    converted_list = []
    import re
    c = 0
    for item in lst:
        if 'unknown' in item:
            item = re.sub('unknown','0',item)
            if c == 0:
                print('unknown values detected')
                print('keeping unit, converting to 0')
                c+=1
                
        if item.endswith('_PCW'):
            value = int(item.split('_')[0]) * 7
        elif 'PCWd' in item:
            value = int(item.split('_')[0]) * 7 + int(item.split('d')[-1])
        elif item.endswith('_PCD'):
            value = int(item.split('_')[0])
        elif item.endswith('_PNY'):
            value = int(item.split('_')[0]) * 365 + 266
        elif item.endswith('_PNM'):
            value = int(item.split('_')[0]) * 30 + 266
        elif item.endswith('_PND'):
            value = int(item.split('_')[0]) + 266
        else:
            raise ValueError("Unsupported format: {}".format(item))
        converted_list.append(value)
        
    return converted_list

def assign_age_to_extra_bin(age_in_days):
    import numpy as np

    # Define the bins in years with extra categories for the first two years
    extra_bins = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 12, 18, 60, 100]
    extra_bin_labels = [
    "Q1", "Q2", "Q3", "Q4",
    "Q5", "Q6", "Q7", "Q8",
    "Child(2-12)", "Teen(12-18)", 
    "Adult(18-60)", "Senior(60-100)"
    ]

    # Convert days to years
    age_in_years = age_in_days / 365.25
    
    # Determine the bin index
    bin_index = np.digitize(age_in_years, extra_bins) - 1
    
    # Ensure the index is within the valid range
    if bin_index < 0:
        bin_index = 0
    elif bin_index >= len(extra_bin_labels):
        bin_index = len(extra_bin_labels) - 1
    
    # Return the corresponding bin label
    return extra_bin_labels[bin_index]

