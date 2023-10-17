import os
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class ImagePandasDataset(Dataset):
    def __init__(self, df,
                 img_key, label_key, img_root="",
                 transform=None,
                 target_transform=None,
                 loader=default_loader):
        '''
        df: pandas dataframe of this dataset
        img_key: column name of storing image path in the dataframe
        label_key: column name of storing labels in the daraframe
        transform: preprpcessing for img
        target_transform: preprocessing for labels
        '''
        self.df = df.sort_values(by=[img_key])
        self.img_key = img_key
        self.label_key = label_key
        self.img_root = img_root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = default_loader
        self.num_classes = len(set([self.get_label_idx(i) for i in range(len(self))]))

    def __getitem__(self, i):
        '''
        get (img,label_idx) pair of i-th data point
        img is already preprocessed
        label_idx start from 0 incrementally 
        That is, they can be used for cnn input directly
        '''
        return {"input": self.get_img(i), "label": self.get_label_idx(i)}

    def get_img_path(self, i):
        '''
        get img_path of i-th data point
        '''
        return os.path.join(self.img_root, str(self.df.iloc[i][self.img_key]))

    def get_img(self, i):
        '''
        get img array of i-th data point
        self.transform is applied if exists
        '''
        img = self.loader(self.get_img_path(i))
        if self.transform is not None:
            img = self.transform(img)
        return img

    def get_label(self, i):
        '''
        get label of i-th data point as it is. 
        '''
        return self.df.iloc[i][self.label_key]

    def get_label_idx(self, i):
        '''
        get label idx, which start from 0 incrementally
        self.target_transform is applied if exists
        '''
        label = self.get_label(i)
        if self.target_transform is not None:
            if isinstance(self.target_transform, dict):
                label_idx = self.target_transform[label]
            else:
                label_idx = self.target_transform(label)
        else:
            label_idx = int(label)
        return label_idx

    def __len__(self):
        return len(self.df)
