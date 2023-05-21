import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler #use for the split of validation dataset and test dataset
import pandas as pd

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """


    def __init__(self, dataset,patient_tags, oversampling_factor,undersampling_factor,random,random_state_data_loader,batch_size, shuffle
                 , validation_split, test_split, num_workers, oversampling, undersampling,collate_fn=default_collate):
        self.validation_split = validation_split
        self.test_split = test_split
        self.shuffle = shuffle
        self.oversampling_factor = oversampling_factor
        self.undersampling_factor = undersampling_factor
        self.rng = random
        print("random_state_Base_loader",self.rng)
        self.batch_idx = 0
        self.n_samples = len(dataset)
        self.patient_tags = patient_tags
        self.random_state_data_loader = random_state_data_loader
        self.sampler, self.valid_sampler, self.test_sampler,\
        self.train_idx,self.valid_idx,self.test_idx,self.n_samples = self._split_sampler(oversampling,undersampling)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self,oversampling=True, undersampling=False):
        idx_full = np.arange(self.n_samples)
        # print(self.rng)
        np.random.seed(self.random_state_data_loader)
        print(self.random_state_data_loader)

        # print("spliter",np.random.seed(self.rng*2))
        np.random.shuffle(idx_full)
        if isinstance(self.validation_split, int) or isinstance(self.test_split, int):
            # assert self.validation_split >=0 or self.test_split >=0
            assert self.validation_split < self.n_samples or self.test_split < self.n_samples, \
                "validation set size or test set size is configured to be larger than entire dataset."
            len_valid = self.validation_split
            len_test  = self.test_split
            valid_idx = idx_full[0:len_valid]
            test_idx = idx_full[len_valid: (len_valid + len_test)]
            train_idx = np.delete(idx_full, np.arange(0, len_valid + len_test))
        elif self.validation_split > 0 and self.test_split > 0:
            len_valid = int(self.n_samples * self.validation_split)
            len_test  = int(self.n_samples * self.test_split)
            valid_idx = idx_full[0:len_valid]
            test_idx = idx_full[len_valid: (len_valid + len_test)]
            train_idx = np.delete(idx_full, np.arange(0, len_valid + len_test))

        elif self.validation_split == 0 and self.test_split == 0:
            len_valid = self.n_samples
            len_test = self.n_samples
            valid_idx = idx_full
            test_idx = idx_full
            train_idx = idx_full
        # valid_idx = idx_full[0:len_valid]
        # test_idx  = idx_full[len_valid: (len_valid+len_test)]
        # train_idx = np.delete(idx_full, np.arange(0, len_valid+len_test))
        train_full = self.patient_tags.iloc[train_idx].copy()
        valid_full = self.patient_tags.iloc[valid_idx].copy()
        test_full = self.patient_tags.iloc[test_idx].copy()
        train_positive = train_full[train_full['Tags']==1].copy().reset_index().astype(int)
        valid_positive = valid_full[valid_full['Tags']==1].copy().reset_index().astype(int)
        valid_negative = valid_full[valid_full['Tags']==0].copy().reset_index().astype(int)
        test_positive = test_full[test_full['Tags']==1].copy().reset_index().astype(int)
        test_negative = test_full[test_full['Tags']==0].copy().reset_index().astype(int)
        print(f"for valid, positive are {len(valid_positive)}, negative are {len(valid_negative)}")
        print(f"for test, positive{len(test_positive)}, negative are{len(test_negative)}")

        train_negative = train_full[train_full['Tags']==0].copy().reset_index().astype(int)
        train_positive_idx = list(train_positive['index'].copy())
        print(f"positive samples are {len(train_positive_idx)}")
        train_negative_idx = list(train_negative['index'].copy())
        print(f"negative samples are {len(train_negative_idx)}")
        if oversampling:
            replace = len(train_positive)<len(train_negative)
            select_patient_index = [idx for idx in self.rng.choice(len(train_positive)
                                                                   ,size=int(len(train_negative)*self.oversampling_factor),replace=replace)]
            new_train_positive = pd.DataFrame([train_positive.loc[idx] for idx in select_patient_index]).copy()
            train_positive_idx = list(new_train_positive['index'].copy())
            print(f"osampling len is {len(train_positive_idx)}")


        if undersampling:
            replace = len(train_negative)>len(train_positive)
            select_patient_index = [idx for idx in self.rng.choice(len(train_negative),
                                                                   size=int(len(train_negative)*self.undersampling_factor),replace=replace)]
            new_train_negative = pd.DataFrame([train_negative.loc[idx] for idx in select_patient_index]).copy()
            train_negative_idx = list(new_train_negative['index'].copy())
            print(f"undersampling len is {len(train_negative_idx)}")

        train_idx = np.array(train_positive_idx + train_negative_idx)
        # np.random.shuffle(train_idx)
        print(f"aftersampling, training length is {len(train_idx)}")
        if self.validation_split!=0:
            full_len = len(train_idx)+len(valid_idx)+len(test_idx)

            print('after oversampling the lengths is ',full_len)
        elif self.validation_split==0:
            full_len = len(train_idx)
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler  = SubsetRandomSampler(test_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler, test_sampler, train_idx,valid_idx,test_idx,full_len

    def split_dataset(self, valid=False, test=False):
        if valid:
            assert len(self.valid_sampler) != 0, "validation set size ratio is not positive"
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
        if test:
            assert len(self.test_sampler) != 0, "test set size ratio is not positive"
            return DataLoader(sampler=self.test_sampler, **self.init_kwargs)
