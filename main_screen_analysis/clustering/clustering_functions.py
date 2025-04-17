import numpy as np
import pandas as pd
import scanpy as sc
from tqdm.notebook import tqdm
import re


def compute_cluster_averages(adata, clustering_key):
    """Compute average gene expression per cluster.

    Parameters:
        adata (AnnData): Annotated data matrix.
        clustering_key (str): Key in adata.obs with cluster labels.

    Returns:
        DataFrame: Gene x Cluster matrix of average expression.
    """
    cluster_means = adata.to_df().groupby(adata.obs[clustering_key]).mean().T
    return cluster_means


def compute_markers_with_closest_cluster(adata, clustering_key, cluster_corr_matrix):
    """Compute differential expression between each cluster and its most similar cluster.

    Parameters:
        adata (AnnData): Annotated data matrix.
        clustering_key (str): Key in adata.obs with cluster labels.
        cluster_corr_matrix (DataFrame): Correlation matrix of cluster profiles.

    Returns:
        DataFrame: Combined DE result for all clusters.
    """
    all_results = []

    for cluster in tqdm(cluster_corr_matrix.columns, desc="Processing clusters"):
        # Identify closest cluster
        most_similar = cluster_corr_matrix[cluster].nlargest(2).index[1]

        # Subset and preprocess
        subset = adata[adata.obs[clustering_key].isin([cluster, most_similar])].copy()
        sc.pp.normalize_total(subset, target_sum=1e4)
        sc.pp.log1p(subset)

        key_added = f"closest_cluster_{cluster}"
        sc.tl.rank_genes_groups(subset, clustering_key, method='wilcoxon', key_added=key_added)

        # Build dataframe
        results = []
        for group in subset.uns[key_added]['names'].dtype.names:
            tmp = pd.DataFrame({
                'gene': subset.uns[key_added]['names'][group],
                'score': subset.uns[key_added]['scores'][group],
                'pval': subset.uns[key_added]['pvals'][group],
                'pval_adj': subset.uns[key_added]['pvals_adj'][group],
                'logfc': subset.uns[key_added]['logfoldchanges'][group],
                'cluster': group
            })
            results.append(tmp.set_index('gene'))

        result_df = pd.concat(results)
        result_df['contrast'] = key_added
        result_df['corr'] = cluster_corr_matrix[cluster].nlargest(2)[1]
        all_results.append(result_df)

    final_df = pd.concat(all_results)
    final_df['orig_cluster'] = final_df['contrast'].str.replace("closest_cluster_", "")
    return final_df


def summarize_cluster_markers(marker_df, informative_genes, pval_thresh=1e-10, logfc_thresh=2):
    """Summarize DEGs per cluster for decision-making on merging.

    Parameters:
        marker_df (DataFrame): DE results.
        informative_genes (List[str]): List of key genes (e.g., ion channels, TFs).
        pval_thresh (float): P-value cutoff.
        logfc_thresh (float): Log fold change threshold.

    Returns:
        DataFrame: Summary stats per cluster.
    """
    summaries = []
    for cluster in marker_df['orig_cluster'].unique():
        cluster_df = marker_df.query("orig_cluster == @cluster and cluster != @cluster")
        sig = cluster_df[(cluster_df['pval_adj'] < pval_thresh) & (cluster_df['logfc'].abs() > logfc_thresh)]

        summary = {
            'cluster': cluster,
            'mcluster': cluster_df['cluster'].iloc[0],
            'ndeg': sig.shape[0],
            'ndegs_info': len(set(sig.index) & set(informative_genes)),
            'corr': cluster_df['corr'].iloc[0]
        }
        summaries.append(summary)

    return pd.DataFrame(summaries)


def merge_indistinguishable_clusters(adata, clustering_key, merge_summary):
    """Merge similar clusters based on the merge summary table.

    Parameters:
        adata (AnnData): Annotated data matrix.
        clustering_key (str): Column in obs to modify.
        merge_summary (DataFrame): Contains cluster, mcluster pairs to merge.

    Returns:
        AnnData: Updated AnnData with merged clusters.
    """
    cluster_map = {}
    current_labels = adata.obs[clustering_key].astype(str)
    existing_numeric_labels = [int(label) for label in current_labels.unique() if label.isdigit()]
    next_id = max(existing_numeric_labels) + 100 if existing_numeric_labels else 300

    for _, row in merge_summary.iterrows():
        c1, c2 = row['cluster'], row['mcluster']
        if c1 in cluster_map:
            new_id = cluster_map[c1]
        elif c2 in cluster_map:
            new_id = cluster_map[c2]
        else:
            new_id = next_id
            next_id += 1

        cluster_map[c1] = new_id
        cluster_map[c2] = new_id

    new_labels = adata.obs[clustering_key].astype(str).copy()
    for old, new in cluster_map.items():
        new_labels.loc[new_labels == old] = str(new)

    # Reassign clean sequential labels
    unique_labels = sorted(set(new_labels))
    label_mapping = {label: str(i) for i, label in enumerate(unique_labels)}
    adata.obs[clustering_key] = new_labels.map(label_mapping).astype('category')
    return adata


def iterative_cluster_merging(adata, base_resolution, info_genes, max_iter=20, pval_thresh=1e-10, logfc_thresh=2, save_deg_summary=True):
    """Main pipeline for iterative merging of clusters.

    Parameters:
        adata (AnnData): Annotated data matrix.
        base_resolution (str): Key in .obs for initial clustering.
        info_genes (List[str]): Informative genes used for clustering similarity.
        max_iter (int): Maximum number of iterations.
        pval_thresh (float): P-value threshold for DEGs.
        logfc_thresh (float): Log fold change threshold for DEGs.
        save_deg_summary (bool): Whether to save the final DEG summary to CSV (default: True).

    Returns:
        AnnData: Updated object with merged clusters in the same key.
    """
    info_genes = [g for g in info_genes if g in adata.var_names]
    print(f"Using {len(info_genes)} informative genes present in the dataset.")

    adata.obs['merged_clusters'] = adata.obs[base_resolution].astype(str)
    i = 0
    merge_summary = [0]

    while len(merge_summary) > 0 and i < max_iter:
        i += 1
        print(f"Iteration {i}")

        cluster_means = compute_cluster_averages(adata, 'merged_clusters')
        norm_means = np.log1p(cluster_means / cluster_means.sum() * 1e6)
        corr_matrix = norm_means.loc[info_genes].corr(method='spearman')

        print("Calculating DEGs...")
        markers_df = compute_markers_with_closest_cluster(adata, 'merged_clusters', corr_matrix)

        print("Summarizing clusters...")
        summary_df = summarize_cluster_markers(markers_df, info_genes, pval_thresh=pval_thresh, logfc_thresh=logfc_thresh)
        merge_summary = summary_df.query("ndeg < 5 or ndegs_info < 3")

        print("Merging clusters...")
        adata = merge_indistinguishable_clusters(adata, 'merged_clusters', merge_summary)

    if save_deg_summary:
        deg_summary_path = "deg_summary_final.csv"
        markers_df_info = markers_df.loc[markers_df.index.isin(info_genes)].copy()
        markers_df_info.to_csv(deg_summary_path)
        print(f"Final DEG summary saved to {deg_summary_path}")

    print(f"Completed in {i} iterations")
    print(f"Original clusters: {len(set(adata.obs[base_resolution]))}")
    print(f"Merged clusters: {len(set(adata.obs['merged_clusters']))}")

    return adata