import pandas as pd
import logging
from os.path import join
import sys
import tensorflow as tf
sys.path.append("/home/sunzehui/GeneNet/")
from data_utils.pathways.reactome import Reactome, ReactomeNetwork
from config_path import *
import tensorflow as tf
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.activations import relu, sigmoid, softmax, tanh
import tensorflow_datasets.public_api as tfds

# Cached_data
cached_data = {}

# Path Define
data_path = DATA_PATH
processed_path = join(DATA_PATH, 'processed')
response_filename = join(DATA_PATH, 'processed/response.csv')
cnv_burden_filename = join(DATA_PATH, 'processed/CNV_burden.csv')
gene_important_mutations_only = join(DATA_PATH, "processed/somatic_mutations_important_only.csv")
cnv_filename = join(DATA_PATH, "processed/CNV.csv")
gene_filename = join(GENE_PATH, "selected_gene")

def load_data(filename, selected_genes=None) -> tuple[pd.DataFrame, pd.Series, pd.Index, list]:
    """This a load_data generic form

    Args:
        filename (String): data file_path
        selected_genes (String, optional): Select specific genes. Defaults to None.

    Returns:
        tuple: (x: pd.DataFrame
                response: pd.Series
                samples: pd.index
                genes: list, after intersection
                )
        
    """
    filename = join(processed_path, filename)
    logging.info(f'loading GeneNet_data from {filename},')
    if filename in cached_data:
        logging.info('loading from memory cached_data')
        data = cached_data[filename]
    else:
        data = pd.read_csv(filename, index_col=0)
        cached_data[filename] = data
    logging.info(f"The loaded dataframe's shape is {data.shape}")

    if 'response' in cached_data:
        logging.info('loading from memory cached_data')
        labels = cached_data['response']
    else:
        labels = get_response()
        cached_data['response'] = labels

    # join with the labels
    _all = data.join(labels, how='inner')
    # 去除response=NA
    _all = _all[~_all['response'].isnull()]

    response = _all['response']
    samples = _all.index

    del _all['response']
    x = _all
    genes = list(_all.columns)

    if selected_genes is not None:
        intersect = set.intersection(set(genes), selected_genes)
        if len(intersect) < len(selected_genes):
            logging.warning('some genes dont exist in the original dataset')
        x = x.loc[:, list(intersect)]
        genes = list(intersect)
    logging.info('loaded GeneNet_data %d samples, %d variables, %d responses ' % (x.shape[0], x.shape[1], response.shape[0]))
    logging.info(len(genes))
    return x, response, samples, genes

def load_data_type(data_type='mut_important', cnv_levels=5, 
                   cnv_filter_single_event=True, mut_binary=False, 
                   selected_genes=None) -> tuple[pd.DataFrame, pd.Series, pd.Index, list]:
    """ for each data_type, do particular data preprocessing
    utilize the previous load_data function

    Args:
        data_type (str, optional): choose from
                                   ['mut_important', 'cnv', 'cnv_indel', 'cnv_amp']
                                   ['cnv_single_del', 'cnv_single_amp', 'cnv_deep_del', 'cnv_deep_amp']
                                   ['cnv_burden']. Defaults to 'mut_important'.
        cnv_levels (int, optional): _description_. Defaults to 5.
        cnv_filter_single_event (bool, optional): _description_. Defaults to True.
        mut_binary (bool, optional): _description_. Defaults to False.
        selected_genes (str, optional): selected_gene path. Defaults to None.

    Returns:
        tuple: (x: pd.DataFrame, pure data in the table
                response: pd.Series, label it as metastatic or primary
                samples: pd.index, sample_id, barcodes for the patients
                genes: list, after intersection, gene name
                )
    """
    
    
    assert(data_type in ['mut_important', 'cnv', 'cnv_indel', 'cnv_amp', 
                         'cnv_single_del', 'cnv_single_amp', 'cnv_deep_del', 
                         'cnv_deep_amp', 'cnv_burden'])
    logging.info('loading {}'.format(data_type))
    
    if data_type == 'mut_important':
        x, response, info, genes = load_data(gene_important_mutations_only, selected_genes)
        if mut_binary:
            logging.info('mut_binary = True')
            x[x > 1.] = 1.

    if data_type == 'cnv':
        x, response, info, genes = load_data(cnv_filename, selected_genes)
        if cnv_levels == 3:
            logging.info('cnv_levels = 3')
            # remove single amplification and single delteion, they are usually noisy
            if cnv_levels == 3:
                if cnv_filter_single_event:
                    x[x == -1.] = 0.0
                    x[x == -2.] = 1.0
                    x[x == 1.] = 0.0
                    x[x == 2.] = 1.0
                else:
                    x[x < 0.] = -1.
                    x[x > 0.] = 1.
        return (x, response, info, genes)

    if data_type == 'cnv_del':
        x, response, info, genes = load_data(cnv_filename, selected_genes)
        x[x >= 0.0] = 0.
        if cnv_levels == 3:
            if cnv_filter_single_event:
                x[x == -1.] = 0.0
                x[x == -2.] = 1.0
            else:
                x[x < 0.0] = 1.0
        else:  # cnv == 5 , use everything
            x[x == -1.] = 0.5
            x[x == -2.] = 1.0
        return (x, response, info, genes)

    if data_type == 'cnv_amp':
        x, response, info, genes = load_data(cnv_filename, selected_genes)
        x[x <= 0.0] = 0.
        if cnv_levels == 3:
            if cnv_filter_single_event:
                x[x == 1.0] = 0.0
                x[x == 2.0] = 1.0
            else:
                x[x > 0.0] = 1.0
        else:  # cnv == 5 , use everything
            x[x == 1.] = 0.5
            x[x == 2.] = 1.0
        return (x, response, info, genes)

    if data_type == 'cnv_single_del':
        x, response, info, genes = load_data(cnv_filename, selected_genes)
        x[x == -1.] = 1.0
        x[x != -1.] = 0.0
        return (x, response, info, genes)
    
    if data_type == 'cnv_single_amp':
        x, response, info, genes = load_data(cnv_filename, selected_genes)
        x[x == 1.] = 1.0
        x[x != 1.] = 0.0
        return (x, response, info, genes)
        
    if data_type == 'cnv_high_amp':
        x, response, info, genes = load_data(cnv_filename, selected_genes)
        x[x == 2.] = 1.0
        x[x != 2.] = 0.0
        return (x, response, info, genes)
        
    if data_type == 'cnv_deep_del':
        x, response, info, genes = load_data(cnv_filename, selected_genes)
        x[x == -2.] = 1.0
        x[x != -2.] = 0.0
        return (x, response, info, genes)

    if data_type == 'cnv_burden':
        x, response, info, genes = load_data(cnv_burden_filename, None)
        return (x, response, info, genes)
    return (x, response, info, genes)

def get_response() -> pd.DataFrame:
    logging.info(f'loading response from {response_filename}')
    labels = pd.read_csv(join(processed_path, response_filename))
    labels = labels.set_index('id')
    return labels

# all_features: make sure all the data_types have the same set of features_processing (genes)
def combine(x_list, y_list, rows_list, cols_list, data_type_list
            , combine_type, use_coding_genes_only=False):
    cols_list_set = [set(list(c)) for c in cols_list]

    if combine_type == 'intersection':
        cols = set.intersection(*cols_list_set)
    else:
        cols = set.union(*cols_list_set)

    if use_coding_genes_only:
        f = join(data_path, 'genes/HUGO_genes/protein-coding_gene_with_coordinate_minimal.txt')
        coding_genes_df = pd.read_csv(f, sep='\t', names=['chr', 'start', 'end', 'name'])
        coding_genes = set(coding_genes_df['name'].unique())
        cols = cols.intersection(coding_genes)

    # the unique (super) set of genes
    all_cols = list(cols)

    all_cols_df = pd.DataFrame(index=all_cols)  # Empty dataframe in which index is genes

    df_list = []
    for x, y, r, c in zip(x_list, y_list, rows_list, cols_list):
        df = pd.DataFrame(x, columns=c, index=r)  # cols are genes, index is id, right join instances
        df = df.T.join(all_cols_df, how='right')  # df.T index are genes, cols are ids
        df = df.T  # index is id, cols are genes
        # print(f"df.columns are {df.columns}, df.index is {df.index}")
        df = df.fillna(0)  # fill it with 0
        df_list.append(df)
    #  这一步的目的是为了将各种不同类型的数据整合在一起，形成一个以index=gene, data_type为multi-index，
    all_data = pd.concat(df_list, keys=data_type_list, join='inner', axis=1, )

    # 整合完毕之后形成一个columns为多重索引[data_type, gene_name], 在这之后用swaplevel函数将其倒换
    # put genes on the first level and then the GeneNet_data type
    all_data = all_data.swaplevel(i=0, j=1, axis=1)

    # order the columns based on genes, 按照基因字母序进行列排序
    order = all_data.columns.levels[0]  # type: ignore #选取columns的多重索引的level=0的索引，即gene
    all_data = all_data.reindex(columns=order, level=0)
    # print(f"df.columns are {all_data.columns}, df.index is {all_data.index}")

    x = all_data.values  # 去除掉表头等标签属性，只提取纯数据

    # prepare response
    reordering_df = pd.DataFrame(index=all_data.index)
    y = reordering_df.join(y, how='left')

    # print(f"df.columns are {y.columns}, df.index is {y.index}")

    y = y.values
    cols = all_data.columns  # [gene, data_type]
    rows = all_data.index  # [id]
    logging.info(
        'After combining, loaded GeneNet_data %d samples, %d variables, %d responses ' % (x.shape[0], x.shape[1], y.shape[0])
    )

    return (x, y, rows, cols)

def split_cnv(x_df):
    genes = x_df.columns.levels[0]
    x_df.rename(columns={'cnv': 'CNA_amplification'}, inplace=True)
    for g in genes:
        x_df[g, 'CNA_deletion'] = x_df[g, 'CNA_amplification'].replace({-1.0: 0.5, -2.0: 1.0})
        x_df[g, 'CNA_amplification'] = x_df[g, 'CNA_amplification'].replace({1.0: 0.5, 2.0: 1.0})
    x_df = x_df.reindex(columns=genes, level=0)
    return x_df

class Dataset:
    """DatasetClass

    Returns:
        _type_: _description_
    """
    # This class will facilitate the creation of GeneNet Dataset
    def __init__(self, training, data_type='cnv', cnv_levels=5,
                 cnv_filter_single_event=True, mut_binary=False,
                 selected_genes=gene_filename, combine_type='intersection',
                 use_coding_genes_only=False):
        # Download the CNV and SNV data and convert to a dataset
        self.split = "train" if training else "test"
        self.data_type = data_type
        self.cnv_levels = cnv_levels
        
        # Selected_gene
        if selected_genes is not None:
            if type(selected_genes) == list:
                # list of genes
                selected_genes = selected_genes
            else:
                # file that will be used to load list of genes
                selected_genes_file = join(data_path, 'genes')
                selected_genes_file = join(selected_genes_file, selected_genes)
                df = pd.read_csv(selected_genes_file, header=0)
                selected_genes = list(df['genes'])
        self.selected_genes = selected_genes
        
        if isinstance(self.data_type, list):
            x_list = []
            y_list = []
            rows_list = []
            cols_list = []

            for t in data_type: 
                x, y, rows, cols = load_data_type(t, cnv_levels, cnv_filter_single_event, mut_binary, selected_genes)
                x_list.append(x)
                y_list.append(y)
                rows_list.append(rows)
                cols_list.append(cols)
            x, y, rows, cols = combine(x_list, y_list, rows_list, cols_list, data_type, combine_type,
                                       use_coding_genes_only)
            x = pd.DataFrame(x, columns=cols)
        else:
            x, y, rows, cols = load_data_type(data_type, cnv_levels, cnv_filter_single_event, mut_binary,
                                              selected_genes)
        
        self._data = x
        self.response = y
        self.barcodes = rows
        self.gene_features = cols
        
    def shape(self):
        return self._data.shape
        
    def statistics(self):
        print(f"This dataset includes {len(self.barcodes)} samples")
        print(f"This dataset includes {len(self.gene_features)} genes features")
        print(f"This dataset includes {self.data_type} datatypes")
        if isinstance(self._data.columns, pd.MultiIndex):
            print(f"Select {len(self.selected_genes)} different genes, after combination and intersection {len(self._data.columns.levels[0])}")
            print(f"RefSeq and raw_dataset only have partial overlaps")
        else:
            print(f"Select {len(self.selected_genes)} different genes, after combination and intersection {len(self._data.columns)} genes left")
            print(f"RefSeq and raw_dataset only have partial overlaps")
            
    def prepare_tf_dataset(self, split=None):
        features = tf.constant(self._data.values)
        labels = tf.constant(self.response.values)
        return tf.data.Dataset.from_tensor_slices((features, labels))

# if __name__ == "__main__":
#     d = Dataset(training=True)
#     for ele in d.prepare_tf_dataset():
#         print(ele)