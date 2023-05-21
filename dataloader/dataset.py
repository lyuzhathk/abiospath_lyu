import os
import random
import numpy as np
import pandas as pd
import networkx as nx

class Dataset(object):
    def __init__(self, data_dir,directed):
        self.data_dir = data_dir
        self.data_dict = self._data_loading()
        self.disease_list, self.gene_list, self.drug_list,self.drug_list_in_ehr = self._statistics()
        self.graph,self.dataframe = self._network_building()
        self.directed = directed

    def get_network(self):
        return self.graph
    def get_dataframe(self):
        return self.dataframe
    def get_ehr_drug(self):
        return self.drug_list_in_ehr
    def get_disease_list(self):
        return self.disease_list
    # def get_drug_disease_associations(self):
    #     positive_drug_disease_pd = self.data_dict['positive_drug_disease']
    #     negative_drug_disease_pd = self._negative_sampling()
    #     positive_drug_disease_pd['label'] = [1] * len(positive_drug_disease_pd)
    #     negative_drug_disease_pd['label'] = [0] * len(negative_drug_disease_pd)
    #     print('{} positive pairs and {} negative pairs.'.format(len(positive_drug_disease_pd), len(negative_drug_disease_pd)))
    #     drug_disease_pd = pd.concat([positive_drug_disease_pd, negative_drug_disease_pd])
    #     return drug_disease_pd

    def get_drug_target(self):
        return self.data_dict['drug_target']

    def get_disease_target(self):
        return self.data_dict['disease_target']

    def _data_loading(self):
        '''Network'''
        # gene regulatory network with columns as from_entrez and target_entrez
        # grn_pd = pd.read_csv(os.path.join(self.data_dir, 'grn.csv'))
        # protein-protein interaction network with columns as protein1_entrez and protein2_entrez
        ppi_pd = pd.read_csv(os.path.join(self.data_dir, 'mapped_network','ppi_mapped.csv'))
        ppi_pd['weight'] = int(1)
        # protein-chemical interaction network with columns as chemical_pubchemcid and protein_entrez
        # pci_pd = pd.read_csv(os.path.join(self.data_dir, 'pci.csv'))
        # chemical-chemical interaction network with columns as chemical1_pubchemcid and chemical2_pubchemcid
        # cci_pd = pd.read_csv(os.path.join(self.data_dir, 'cci.csv'))
        commorbidity_pd = pd.read_csv(os.path.join(self.data_dir),'mapped_network','commorbidity_mapped.csv')
        commorbidity_order = ['source','target','weight']
        commorbidity_pd = commorbidity_pd[commorbidity_order].copy()
        '''Drug and Disease Info'''
        # the columns are target_entrez and disease_icd10

        disease_target_pd = pd.read_csv(os.path.join(self.data_dir,'mapped_network','dpi_mapped.csv'))
        if self.directed:
            disease_target_pd['weight'] = int(1)
        else:
            disease_target_pd['weight'] = float(0.5)
        disease_target_order = ["ICD","entrez",'weight']
        disease_target_pd=disease_target_pd[disease_target_order].copy()
        # the columns are drug_pubchemcid and target_entrez
        drug_target_pd = pd.read_csv(os.path.join(self.data_dir,'mapped_network','drug_target_mapped.csv'))
        drug_target_pd['weight'] = int(1)
        drug_target_order=["cid","entrez",'weight']
        drug_target_pd=drug_target_pd[drug_target_order]
        # the columns are drug_pubchemcid and disease_icd10
        # positive_drug_disease_pd = pd.read_csv(os.path.join(self.data_dir, 'drug_disease.csv'))
        '''weight changing'''
        # commorbidity_weight = list(set(commorbidity_pd['weight']))
        # print(max(commorbidity_weight),min(commorbidity_weight))
        # interval = (max(commorbidity_weight)-min(commorbidity_weight))/5
        # from itertools import groupby
        # for k, g in groupby(sorted(commorbidity_weight), key=lambda x: x//interval):
        #     print('{}-{}: {}'.format(k*interval, (k+1)*interval-1, len(list(g))))
        '''data dict'''
        data_dict = {'ppi': ppi_pd,
                     'disease_target': disease_target_pd, 'drug_target': drug_target_pd,
                     'commorbidity':commorbidity_pd}
        return data_dict

    def _statistics(self):
        # node_list = list(pd.read_csv(os.path.join(self.data_dir, 'mapping_file', 'node_map.csv'))['map'])
        disease_list = list(pd.read_csv(os.path.join(self.data_dir, 'mapping_file', 'Icd9_map.csv'))['map'])
        # chemical_list = list(pd.read_csv(os.path.join(self.data_dir, 'mapping_file', 'chemical_map.csv'))['map'])
        gene_list = list(pd.read_csv(os.path.join(self.data_dir, 'mapping_file', 'protein_map.csv'))['map'])
        drug_list = list(pd.read_csv(os.path.join(self.data_dir, 'mapping_file', 'drug_map.csv'))['map'])
        drug_list_in_ehr = list(pd.read_csv(os.path.join(self.data_dir,'mapping_file','drug_used_in_ehr.csv'))['map'])
        return disease_list, gene_list, drug_list,drug_list_in_ehr

    def display_data_statistics(self):
        print('{} nodes and {} edges in the directed ppi network'.format(
            len(self.gene_list), len(self.data_dict['ppi'])
                ))
        print('{} drugs and {} diseases in our dataset'.format(
            len(self.drug_list), len(self.disease_list)))
        print('Drugs have {} targets and disease have {} targets'.format(
            len(self.data_dict['drug_target']), len(self.data_dict['disease_target'])))


    #TODO negative sampling
    # def _negative_sampling(self):
    #     print('Negative sampling...')
    #     positive_pd = self.data_dict['positive_drug_disease']
    #     drug_list = list(set(positive_pd['drug_pubchemcid']))
    #     disease_list = list(set(positive_pd['disease_icd10']))
    #     negative_drug_list, negative_disease_list = [], []
    #     for drug in drug_list:
    #         positive_disease_list = list(positive_pd[positive_pd['drug_pubchemcid']==drug]['disease_icd10'])
    #         disease_left_list = list(set(disease_list) - set(positive_disease_list))
    #         # random select the diseases with the same number of that of positive diseases
    #         negative_disease_list += random.sample(disease_left_list, min(len(positive_disease_list), len(disease_left_list)))
    #         negative_drug_list += [drug] * min(len(positive_disease_list), len(disease_left_list))
    #     negative_pd = pd.DataFrame({'drug_pubchemcid': negative_drug_list,
    #                                 'disease_icd10': negative_disease_list})
    #     print('For {0} drugs, {1} negative samples are generated.'.format(len(drug_list), len(negative_disease_list)))
    #     return negative_pd
    
    def _df_column_switch(self, df_name):
        df_copy = self.data_dict[df_name].copy()
        df_copy.columns = ['from', 'target','weight']
        df_switch = self.data_dict[df_name].copy()
        df_switch.columns = ['target', 'from','weight']
        df_concat = pd.concat([df_copy, df_switch])
        df_concat.drop_duplicates(subset=['from', 'target','weight'], inplace=True)
        return df_concat

    def _network_building(self):
        ppi_directed = self._df_column_switch(df_name='ppi')
        # pci_directed = self._df_column_switch(df_name='pci')
        # cci_directed = self._df_column_switch(df_name='cci')
        # the direction in grn is from-gene -> target-gene
        # grn_directed = self.data_dict['grn'].copy()
        # grn_directed.columns = ['from', 'target']
        commorbidity_bidi = self._df_column_switch(df_name="commorbidity")

        # the direction in drug-target network is drug -> target
        drug_target_directed = self.data_dict['drug_target'].copy()
        drug_target_directed.columns = ['from', 'target','weight']
        # orginial dataframe is [ICD9 code, entrez weight]
        # the direction in disease-target network is entrez -> diseasea
        if self.directed:
            disease_target_directed = self.data_dict['disease_target'].copy()
            disease_target_directed.columns = ['target', 'from','weight']
        else:
            disease_target_directed = self._df_column_switch(df_name="disease_target")
        
        graph_directed = pd.concat([ppi_directed,
                                    drug_target_directed, disease_target_directed,commorbidity_bidi])
        graph_directed.drop_duplicates(subset=['from', 'target','weight'], inplace=True)

        graph_nx = nx.from_pandas_edgelist(graph_directed, source='from', target='target',edge_attr=['weight'],
                                           create_using=nx.DiGraph())
        
        return graph_nx,graph_directed

from numpy.lib.function_base import select
import torch
import numpy as np
from scipy import sparse
from torch.utils.data import Dataset
import itertools
class PathDataset(Dataset):
    def __init__(self, drug_disease_array, total_path_dict, type_dict,
                        patient_tags_list,patient_drugs_pd,patient_icd_pd,max_path_length=8, max_path_num=8, rng=None):
        self.drug_disease_array = drug_disease_array
        self.total_path_dict = total_path_dict
        self.type_dict = type_dict
        self.max_path_length = max_path_length
        self.max_path_num = max_path_num
        self.rng = rng
        self.patient_tags_list = patient_tags_list
        self.patient_drugs_pd = patient_drugs_pd
        self.patient_icd_pd = patient_icd_pd
    def __len__(self):
        return len(self.patient_tags_list)

    def __getitem__(self, index):
        print("1")
        patid, tags = self.patient_icd_pd.iloc[index,:]
        print("2")
        drug_list = list(set(self.patient_drugs_pd[self.patient_drugs_pd['PATID']==patid]['cid']))
        disease_list = list(set(self.patient_icd_pd[self.patient_icd_pd['PATID']==patid]['ICD']))
        full_path_type_array_list = []
        full_path_lengths_array_list = []
        full_path_mask_array_list = []
        full_path_array_list = []
        drug_disease_list = list(itertools.product(drug_list, disease_list))
        drug_disease_indict = list(set(drug_disease_list).intersection(set(self.drug_disease_array)))
        '''find drug path'''
        if len(drug_disease_indict) > 0:
            for drug_disease_tuple in drug_disease_indict:
                path_list = self.total_path_dict[drug_disease_tuple]
                # print(drug_disease_tuple)
                path_array_list = []
                type_array_list = []
                lengths_list = []
                mask_list = []
                for path in path_list:
                    path = path[:self.max_path_length]
                    pad_num = max(0, (self.max_path_length - len(path)))
                    path_array_list.append(path + [0] * pad_num)
                    type_array_list.append([self.type_dict[n] for n in path] + [0] * pad_num)
                    lengths_list.append(len(path))
                    mask_list.append([1] * len(path) + [0] * pad_num)
                replace = len(path_array_list) < self.max_path_num
                select_idx_list = [idx for idx in self.rng.choice(len(path_array_list), size=self.max_path_num, replace=replace)]
                path_array = [path_array_list[idx] for idx in select_idx_list]
                type_array = [type_array_list[idx] for idx in select_idx_list]
                lengths_array = [lengths_list[idx] for idx in select_idx_list]
                mask_array = [mask_list[idx] for idx in select_idx_list]
                full_path_array_list = [*full_path_array_list, *path_array]
                full_path_type_array_list = [*full_path_type_array_list, *type_array]
                full_path_lengths_array_list = [*full_path_lengths_array_list, *lengths_array]
                full_path_mask_array_list = [*full_path_mask_array_list, *mask_array]
        '''find the disease path'''
        disease_array_list = []
        disease_type_list = []
        disease_lengths_list = []
        disease_mask_list = []
        for disease in disease_list:
            disease = [disease]
            disease_array_list.append(disease + [0] * 7)
            disease_lengths_list.append(1)
            disease_mask_list.append([1] + [0] * 7)
            disease_type_list.append([self.type_dict[n] for n in disease] + [0] * 7)
        full_array_list = [*full_path_array_list, *disease_array_list]
        full_type_list = [*full_path_type_array_list, *disease_type_list]
        full_lengths_list = [*full_path_lengths_array_list, *disease_lengths_list]
        disease_location_list = [*([0] * len(full_path_array_list)), *disease_lengths_list]
        path_location_list = [*([1]*len(full_path_array_list)),*([0]*len(disease_lengths_list))]
        full_mask_list = [*full_path_mask_array_list, *disease_mask_list]
        full_array = np.array(full_array_list)
        full_type = np.array(full_type_list)
        full_lengths = np.array(full_lengths_list)
        full_mask = np.array(full_mask_list)
        disease_location_array = np.array(disease_location_list)
        path_location_array = np.array(path_location_list)
        path_feature = torch.from_numpy(full_array).type(torch.LongTensor)
        type_feature = torch.from_numpy(full_type).type(torch.LongTensor)
        label = torch.from_numpy(np.array([tags])).type(torch.FloatTensor)
        lengths = torch.from_numpy(full_lengths).type(torch.LongTensor)
        mask = torch.from_numpy(full_mask).type(torch.ByteTensor)
        disease_location = torch.from_numpy(disease_location_array).type(torch.LongTensor)
        path_location = torch.from_numpy(path_location_array).type(torch.LongTensor)
        path_num = path_location.sum()
        # if path_num!=0:
        #     path_location = path_location/pad_num

        return path_feature, type_feature, lengths, mask, label, patid, disease_location, path_location, path_num


# b=Dataset("zhiheng_network")

import os
import pickle
import torch
import random
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm # 进度条
from scipy import sparse #创建稀疏矩阵
from pathlib import Path
from base import BaseDataLoader
# from data_loader.dataset import Dataset
# from data_loader.path_dataset import PathDataset

class PathDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size,
                       max_path_length=8, max_path_num=8, random_state=0, recreate=False, directed=True, use_disease_seed=False,
                       shuffle=True, validation_split=0.1, test_split=0.2, num_workers=1, partial_pair=True,training=True):
        random.seed(0)
        self.data_dir = Path(data_dir)
        self.partial_pair=partial_pair
        self.max_path_length = max_path_length
        self.max_path_num = max_path_num
        self.random_state = random_state
        self.recreate = recreate
        self.use_disease_seed = use_disease_seed

        self.rng = np.random.RandomState(random_state)

        self.graph = self._data_loader()
        self.node_num = self.graph.number_of_nodes()
        self.type_dict = self._get_type_dict()
        self.path_dict = self._load_path_dict()
        self.dataset = self._create_dataset()
        self.directed = directed

        super().__init__(self.dataset, batch_size, shuffle, validation_split, test_split, num_workers)
    def get_node_num(self):
        return self.node_num

    def get_type_num(self):
        return 3

    def get_sparse_adj(self):
        def adj_normalize(mx):
            '''Row-normalize sparse matrix'''
            rowsum = np.array(mx.sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)]=0.
            r_mat_inv = sparse.diags(r_inv)
            mx = r_mat_inv
            return mx

        def sparse_mx_to_torch_sparse_tensor(sparse_mx):
            '''convert a scipy sparse matrix into a torch sparse tensor'''
            sparse_mx=sparse_mx.tocoo().astype(np.float32)
            indices = torch.from_numpy(np.vstack((sparse_mx.row,sparse_mx.col)).astype(np.int64))
            values = torch.from_numpy(sparse_mx.data)
            shape = torch.Size(sparse_mx.shape)

            return torch.sparse.FloatTensor(indices,values,shape)

        print('Get sparse adjacency matrix in csr format')

        # csr matrix, note that if there is a link from node A to B, then the nonzero value in the adjacency matrix is (A, B)
        #where A is the row number and B is the column number

        csr_adjmatrix = nx.adjacency_matrix(self.graph, nodelist =sorted(list(range(1,self.node_num+1))))

        #add virtual node (index is 0) for padding
        virtual_col = sparse.csr_matrix(np.zeros([self.node_num,1]))
        csr_adjmatrix=sparse.hstack([virtual_col,csr_adjmatrix])
        virtual_row = sparse.csr_matrix(np.zeros([1,self.node_num+1]))
        csr_adjmatrix=sparse.vstack([virtual_row,csr_adjmatrix])
        row_num, col_num = csr_adjmatrix.shape
        print('{} edges among {} possible pairs.' .format(csr_adjmatrix.getnnz(),row_num*col_num))

        adj = csr_adjmatrix.tocoo()
        adj=adj_normalize(adj+sparse.eye(row_num))
        adj_tensor = sparse_mx_to_torch_sparse_tensor(adj)

        return adj_tensor
    def _data_loader(self):
        print('Load graph and other basic data....')
        if self.data_dir.joinpath('processed','full_graph.pkl').exists():
            print("Loading existing file")
            with open(self.data_dir.joinpath('processed','full_graph.pkl'),'rb') as f:
                graph = pickle.load(f)
        else:
            print("Creating files")
            graph_dataset, graph_dataframe= Dataset(data_dir=os.path.join(self.data_dir),directed=self.directed)
            graph = graph_dataset.get_network()
            graph_path = self.data_dir.joinpath('processed','full_graph.pkl')
            os.makedirs(os.path.dirname(graph_path), exist_ok=True)
            with self.data_dir.joinpath('processed','full_graph.pkl').open('wb') as f:
                pickle.dump(graph,f)
        return graph
    def _get_type_dict(self):
        type_mapping_dict = {'gene':1,'drug':2,'disease':3}
        if self.data_dir.joinpath('processed','type.csv').is_file():
            print('Loading existing file')
            type_pd=pd.read_csv(self.data_dir.joinpath('processed','type.csv'))
            type_dict={row['node']:type_mapping_dict[row['type']] for index,row in type_pd.iterrows()}

        else:
            gene_map_pd = pd.read_csv(self.data_dir.joinpath('mapping_file','protein_map.csv'))
            durg_map_pd = pd.read_csv(self.data_dir.joinpath('mapping_file','drug_map.csv'))
            disease_map_pd = pd.read_csv(self.data_dir.joinpath('mapping_file','Icd9_map.csv'))

            gene_map_list = {node:'gene' for node in list(gene_map_pd['map'])}
            drug_map_list = {node:'drug' for node in list(durg_map_pd['map'])}
            disease_map_list = {node:'disease' for node in list(disease_map_pd['map'])}
            type_dict = {**gene_map_list,**drug_map_list,**disease_map_list}
            type_dict_pd = pd.DataFrame({'node':list(type_dict.keys())
                                        ,'type':list(type_dict.values())})
            type_dict_pd.to_csv(self.data_dir.joinpath('processed','type.csv'),index=False,header=True)
            type_dict = {node: type_mapping_dict[type_string] for node, type_string in type_dict.items()}

        return type_dict

    # TODO negetative sampling!

    def _load_path_dict(self):
        if not self.recreate and self.partial_pair and self.data_dir.joinpath('processed','path_dict_partial.pkl').is_file():
            print('Load existing path_dict_partial.pkl....')
            with self.data_dir.joinpath('processed','path_dict_partial.pkl').open('rb') as f:
                path_dict = pickle.load(f)
        elif not self.recreate and not self.partial_pair and self.data_dir.join('processed','path_dict_possible.pkl').is_file():

            print("Load existing path_dict_possible.pkl.....")
            with self.data_dir.joinpath('processed','path_dict_possible.pkl').open('rb') as f:

                path_dict = pickle.load(f)
        else:
            print("Starting creating path_dict....")
            print('Loading drug_path_dict.pkl....')
            with open(self.data_dir.joinpath('path_new','drug_path_dict.pkl'),'rb') as f:
                drug_path_dict = pickle.load(f)
            print('drug path Loaded')
            print('loading disease path now')
            with open(self.data_dir.joinpath('path_new','disease_path_dict.pkl'),'rb') as f:
                disease_path_dict = pickle.load(f)
            print('disease path loaded')
            drug_target_pd = pd.read_csv(self.data_dir.joinpath('mapped_network','drug_target_mapped.csv'))
            disease_target_pd = pd.read_csv(self.data_dir.joinpath('mapped_network','dpi_mapped.csv'))
            partial_pair = self.partial_pair

            path_dict, drug_disease_pd = self._create_path(drug_path_dict,disease_path_dict,drug_target_pd,disease_target_pd,partial_pair)
            if self.partial_pair:
                drug_disease_pd.to_csv(self.data_dir.joinpath('processed','partial_drug_disease.csv'),index=False)
                with self.data_dir.joinpath('processed','path_dict_partial.pkl').open('wb') as f:
                    pickle.dump(path_dict,f)
            else:
                drug_disease_pd.to_csv(self.data_dir.joinpath('processed','possible_drug_disease.csv'),index=False)
                with self.data_dir.joinpath('processed','path_dict_possible.pkl').open('wb') as f:
                    pickle.dump(path_dict,f)
        return path_dict



    def _create_path(self,drug_path_dict,disease_path_dict,drug_target_pd,disease_target_pd,partial_pair):
        print('creating the path between ehr drug and commorbidity diseases...')
        if partial_pair:
            drug_disease_pd = pd.read_csv(self.data_dir.joinpath("mapped_network","drug_disease_partial.csv"),header=0)
        else:
            drug_disease_pd = pd.read_csv(self.data_dir.joinpath("mapped_network","drug_disease_possible.csv"),header=0)

        path_dict=dict()
        for idx,row in tqdm(drug_disease_pd.iterrows()):
            drug,disease = row['cid'],row['ICD']
            drug_target_list = list(set(drug_target_pd[drug_target_pd["cid"]==drug]["entrez"]))
            disease_target_list = list(set(disease_target_pd[disease_target_pd["ICD"]==disease]["entrez"]))

            if len(drug_path_dict[drug])==0 or len(disease_path_dict[disease])==0:
                continue
            drug_path_list=[drug_path_dict[drug][t]+[disease] for t in disease_target_list if t in drug_path_dict[drug]]
            disease_path_list = [disease_path_dict[disease][t]+[drug] for t in drug_target_list if t in disease_path_dict[disease]]
            disease_path_list = [path[::-1] for path in disease_path_list]

            path_list = drug_path_list+disease_path_list
            if len(path_list)==0:
                continue
            path_dict[tuple([drug,disease])] = path_list
        return path_dict,drug_disease_pd

    def _create_dataset(self):
        print("creating tensor dataset....")
        drug_disease_array = list(self.path_dict.keys())
        patient_tags_pd = pd.read_csv('zhiheng_network/mapped_network/patient_tags.csv', header=0)
        patient_tags_list = [tuple(patient_tag) for patient_tag in patient_tags_pd.values]
        patient_drug_pd = pd.read_csv('zhiheng_network/mapped_network/patient_drug.csv', header=0)
        patient_icd_pd = pd.read_csv('zhiheng_network/mapped_network/patient_icd.csv', header=0)
        dataset = PathDataset(drug_disease_array=drug_disease_array, total_path_dict=self.path_dict,
                              type_dict=self.type_dict, patient_tags_list=patient_tags_list, patient_drugs_pd=patient_drug_pd,
                              patient_icd_pd=patient_icd_pd, max_path_length=self.max_path_length,
                              max_path_num=self.max_path_num, rng=self.rng)
        return dataset
# class PathDataset(Dataset):
#
# drug_disease_array = list(self.path_dict.keys())
# total_path_dict = self.path_dict
# max_path_length = self.max_path_length
# type_dict = self.type_dict
# rng=self.rng
# index = 1
#
# patid,tags = patient_tags_pd.iloc[index,:]
# drug_list = list(set(patient_drug_pd[patient_drug_pd['PATID']==patid]['cid']))
# disease_list = list(set(patient_icd_pd[patient_icd_pd['PATID']==patid]['ICD']))
# import itertools
# full_path_list = []
# drug_disease_list = list(itertools.product(drug_list,disease_list))
# drug_disease_indict = list(set(drug_disease_list).intersection(set(drug_disease_array)))
# if len(drug_disease_indict)>0:
#     for drug_disease_tuple in drug_disease_indict:
#         path_list = total_path_dict[drug_disease_tuple]
#         path_array_list = []
#         type_array_list = []
#         lengths_list=[]
#         mask_list=[]
#         for path in path_list:
#             path=path[:max_path_length]
#             pad_num = max(0, max_path_length - len(path))
#             path_array_list.append(path + [0] * pad_num)
#             type_array_list.append([type_dict[n] for n in path] + [0] * pad_num)
#             lengths_list.append(len(path))
#             mask_list.append([1] * len(path) + [0] * pad_num)
#         replace = len(path_array_list) < self.max_path_num
#         select_idx_list = [idx for idx in rng.choice(len(path_array_list), size=max_path_num, replace=replace)]
#         path_array = [path_array_list[idx] for idx in select_idx_list]
#         type_array = np.array([type_array_list[idx] for idx in select_idx_list])
#         lengths_array = np.array([lengths_list[idx] for idx in select_idx_list])
#         mask_array = np.array([mask_list[idx] for idx in select_idx_list])
        # full_path_list = full_path_list.append(path_array)
#     def __init__(self,drug_disease_array, total_path_dict, type_dict,
#                  max_path_length=8, max_path_num=8, rng=None):
#         self.drug_disease_array = drug_disease_array
#         self.total_path_dict = total_path_dict
#         self.type_dict = type_dict
#         self.max_path_length = max_path_length
#         self.max_path_num = max_path_num
#         self.rng = rng
#         self.patient_ehr = get_patient_ehr()
#         self.patient_drug = get_patient_drug()
#
#     def __len__(self):
#         return len(self.len(drug_disease_array))
#     def get_patient_ehr(self):
#         patient_ehr = pd.read_csv("")
#
#     def __getitem__(self, item):

a = PathDataLoader("zhiheng_network",batch_size = 512, max_path_length = 8, max_path_num = 156, random_state=0
                   , recreate=False, use_disease_seed=True
                   , shuffle = True, validation_split = 0.1
                   , test_split = 0.2, num_workers = 2)
a.dataset[1]
print(a)
# def get_shortest_path(G, source_list, targest_list):
#     shortest_path = []
#     source_l = source_list,
#     print(source_list)
#     target_l = targest_list,
#     print(targest_list)
#     source_len=len(source_l)
#     target_len=len(target_l)
#     for i in range(source_len):
#         for j in range(target_len):
#             print(source_list[i],targest_list[j])
#
#
#             # path = nx.shortest_path(G, source_l[i], target_l[j])
#             # shortest_path.append(path)
#
#     return shortest_path
# for i in b.disease_list:
#     print(i)
# Path=get_shortest_path(b.graph,b.drug_list_in_ehr,b.disease_list)
# a = PathGenetorDataset.get_dataframe()