import itertools
import logging
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization, multiply
from tensorflow.keras.regularizers import l2
from data_utils.pathways.reactome import ReactomeNetwork

# layer_dict:
# {pathway1: [gene1, gene2, gene3, gene4, ...], 
#  pathway2: [gene1, gene2, gene3, gene4, ...]}
def get_map_from_layer(layer_dict):
    pathways = layer_dict.keys()
    print('''pathways' number''', len(pathways))
    genes = list(itertools.chain.from_iterable(layer_dict.values())) # values = [[g1,g2], [g3,g4]]
    genes = list(np.unique(genes))
    print('unique genes included in reactome mapps', len(genes))

    n_pathways = len(pathways)
    n_genes = len(genes)

    mat = np.zeros((n_pathways, n_genes))
    for p, gs in layer_dict.items():
        g_inds = [genes.index(g) for g in gs]
        p_ind = pathways.index(p)
        mat[p_ind, g_inds] = 1

    df = pd.DataFrame(mat, index=pathways, columns=genes)
    return df.T


def get_layer_maps(genes, n_levels, direction, add_unk_genes):
    reactome_layers = ReactomeNetwork().get_layers(n_levels, direction)
    filtering_index = genes
    maps = []
    for i, layer in enumerate(reactome_layers[::-1]):
        print ('layer #', i)
        mapp = get_map_from_layer(layer)
        filter_df = pd.DataFrame(index=filtering_index)
        print ('filtered_map', filter_df.shape)
        filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='left')
        print ('filtered_map', filter_df.shape)

        if add_unk_genes:
            print('UNK ')
            filtered_map['UNK'] = 0
            ind = filtered_map.sum(axis=1) == 0
            filtered_map.loc[ind, 'UNK'] = 1

        filtered_map = filtered_map.fillna(0)
        print('filtered_map', filter_df.shape)
        filtering_index = filtered_map.columns
        logging.info('layer {} , # of edges  {}'.format(i, filtered_map.sum().sum()))
        maps.append(filtered_map)
    return maps


def shuffle_genes_map(mapp):
    logging.info('shuffling')
    ones_ratio = np.sum(mapp) / np.prod(mapp.shape)
    logging.info('ones_ratio {}'.format(ones_ratio))
    mapp = np.random.choice([0, 1], size=mapp.shape, p=[1 - ones_ratio, ones_ratio])
    logging.info('random map ones_ratio {}'.format(ones_ratio))
    return mapp


def get_genenet(inputs, features, genes, n_hidden_layers, direction, activation, activation_decision, w_reg,
             w_reg_outcomes, dropout, sparse, add_unk_genes, batch_normal, kernel_initializer, use_bias=False,
             shuffle_genes=False, attention=False, dropout_testing=False, non_neg=False, sparse_first_layer=True):
    feature_names = {}
    n_features = len(features)
    n_genes = len(genes)

    if not isinstance(w_reg, list):
        w_reg = [w_reg] * 10

    if not isinstance(w_reg_outcomes, list):
        w_reg_outcomes = [w_reg_outcomes] * 10

    if not isinstance(dropout, list):
        dropout = [w_reg_outcomes] * 10

    w_reg0 = w_reg[0]
    w_reg_outcome0 = w_reg_outcomes[0]
    w_reg_outcome1 = w_reg_outcomes[1]
    reg_l = l2
    constraints = {}
    if non_neg:
        from keras.constraints import nonneg
        constraints = {'kernel_constraint': nonneg()}
        # constraints= {'kernel_constraint': nonneg(), 'bias_constraint':nonneg() }
    if sparse:
        if shuffle_genes == 'all':
            ones_ratio = float(n_features) / np.prod([n_genes, n_features])
            logging.info('ones_ratio random {}'.format(ones_ratio))
            mapp = np.random.choice([0, 1], size=[n_features, n_genes], p=[1 - ones_ratio, ones_ratio])
            layer1 = SparseTF(n_genes, mapp, activation=activation, W_regularizer=reg_l(w_reg0),
                              name='h{}'.format(0), kernel_initializer=kernel_initializer, use_bias=use_bias,
                              **constraints)
        else:
            layer1 = Diagonal(n_genes, input_shape=(n_features,), activation=activation, W_regularizer=l2(w_reg0),
                              use_bias=use_bias, name='h0', kernel_initializer=kernel_initializer, **constraints)


    else:
        if sparse_first_layer:
            layer1 = Diagonal(n_genes, input_shape=(n_features,), activation=activation, W_regularizer=l2(w_reg0),
                              use_bias=use_bias, name='h0', kernel_initializer=kernel_initializer, **constraints)
        else:
            layer1 = Dense(n_genes, input_shape=(n_features,), activation=activation, W_regularizer=l2(w_reg0),
                           use_bias=use_bias, name='h0', kernel_initializer=kernel_initializer)
    outcome = layer1(inputs)
    if attention:
        attention_probs = Diagonal(n_genes, input_shape=(n_features,), activation='sigmoid', W_regularizer=l2(w_reg0),
                                   name='attention0')(inputs)
        outcome = multiply([outcome, attention_probs], name='attention_mul')

    decision_outcomes = []

    decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(0), W_regularizer=reg_l(w_reg_outcome0))(
        inputs)
  
    # testing
    if batch_normal:
        decision_outcome = BatchNormalization()(decision_outcome)
    decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(1),
                             W_regularizer=reg_l(w_reg_outcome1 / 2.))(outcome)
    drop2 = Dropout(dropout[0], name='dropout_{}'.format(0))

    outcome = drop2(outcome, training=dropout_testing)

    # testing
    if batch_normal:
        decision_outcome = BatchNormalization()(decision_outcome)

    decision_outcome = Activation(activation=activation_decision, name='o{}'.format(1))(decision_outcome)
    decision_outcomes.append(decision_outcome)

    if n_hidden_layers > 0:
        maps = get_layer_maps(genes, n_hidden_layers, direction, add_unk_genes)
        layer_inds = range(1, len(maps))
        w_regs = w_reg[1:]
        w_reg_outcomes = w_reg_outcomes[1:]
        dropouts = dropout[1:]
        for i, mapp in enumerate(maps[0:-1]):
            w_reg = w_regs[i]
            w_reg_outcome = w_reg_outcomes[i]
            # dropout2 = dropouts[i]
            dropout = dropouts[1]
            names = mapp.index
            # names = list(mapp.index)
            mapp = mapp.values
            if shuffle_genes in ['all', 'pathways']:
                mapp = shuffle_genes_map(mapp)
            n_genes, n_pathways = mapp.shape
            logging.info('n_genes, n_pathways {} {} '.format(n_genes, n_pathways))
            # print 'map # ones {}'.format(np.sum(mapp))
            layer_name = 'h{}'.format(i + 1)
            if sparse:
                hidden_layer = SparseTF(n_pathways, mapp, activation=activation, W_regularizer=reg_l(w_reg),
                                        name=layer_name, kernel_initializer=kernel_initializer,
                                        use_bias=use_bias, **constraints)
            else:
                hidden_layer = Dense(n_pathways, activation=activation, W_regularizer=reg_l(w_reg),
                                     name=layer_name, kernel_initializer=kernel_initializer, **constraints)

            outcome = hidden_layer(outcome)

            if attention:
                attention_probs = Dense(n_pathways, activation='sigmoid', name='attention{}'.format(i + 1),
                                        W_regularizer=l2(w_reg))(outcome)
                outcome = multiply([outcome, attention_probs], name='attention_mul{}'.format(i + 1))

            decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(i + 2),
                                     W_regularizer=reg_l(w_reg_outcome))(outcome)
          
            if batch_normal:
                decision_outcome = BatchNormalization()(decision_outcome)
            decision_outcome = Activation(activation=activation_decision, name='o{}'.format(i + 2))(decision_outcome)
            decision_outcomes.append(decision_outcome)
            drop2 = Dropout(dropout, name='dropout_{}'.format(i + 1))
            outcome = drop2(outcome, training=dropout_testing)

            feature_names['h{}'.format(i)] = names
        i = len(maps)
        feature_names['h{}'.format(i - 1)] = maps[-1].index
    return outcome, decision_outcomes, feature_names



class sparse_decoder(tf.keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        
    def call(self, inputs, **kwargs):
        return super().call(inputs, **kwargs)
    