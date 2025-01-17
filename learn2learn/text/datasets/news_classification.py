import io
import os
import zipfile

import pandas as pd
import requests
import torch
from torch.utils.data import Dataset


class NewsClassificationDataset(Dataset):
    """

    [[Source]]()

    **Description**

    **References**

    * TODO: Cite ...

    **Arguments**

    **Example**

    """

    def __init__(self, root, train=True, transform=None, download=False):
        self.labels_list = {'QUEER VOICES': 0, 'GREEN': 1, 'STYLE': 2, 'BUSINESS': 3, 'CULTURE & ARTS': 4,
                            'WEDDINGS': 5, 'ARTS': 6, 'HEALTHY LIVING': 7,
                            'LATINO VOICES': 8, 'ENVIRONMENT': 9, 'FIFTY': 10, 'COMEDY': 11, 'BLACK VOICES': 12,
                            'TRAVEL': 13, 'ENTERTAINMENT': 14, 'TASTE': 15,
                            'CRIME': 16, 'WOMEN': 17, 'TECH': 18, 'PARENTING': 19, 'SCIENCE': 20, 'WORLD NEWS': 21,
                            'WORLDPOST': 22, 'POLITICS': 23,
                            'ARTS & CULTURE': 24, 'RELIGION': 25, 'IMPACT': 26, 'MEDIA': 27, 'STYLE & BEAUTY': 28,
                            'SPORTS': 29, 'WEIRD NEWS': 30,
                            'HOME & LIVING': 31, 'THE WORLDPOST': 32, 'MONEY': 33, 'EDUCATION': 34, 'DIVORCE': 35,
                            'PARENTS': 36, 'GOOD NEWS': 37,
                            'FOOD & DRINK': 38, 'WELLNESS': 39, 'COLLEGE': 40}
        if train:
            self.path = os.path.join(root, 'train_sample.csv')
        else:
            self.path = os.path.join(root, 'test_sample.csv')
        self.transform = transform
        if transform == 'roberta':
            self.roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')

        if download:
            download_file_url = 'https://www.dropbox.com/s/g8hwl9pxftl36ww/test_sample.csv.zip?dl=1'
            if train:
                download_file_url = 'https://www.dropbox.com/s/o71z7fq7mydbznc/train_sample.csv.zip?dl=1'

            r = requests.get(download_file_url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(path=root)

        if root:

            if os.path.exists(self.path):
                origin_df = pd.read_csv(self.path)#缺少对数据的预处理，比如会出现nan导致训练出错
                origin_df = origin_df.dropna(
                axis=0,     # 0: 对行进行操作; 1: 对列进行操作
                how='any',   # 'any': 只要存在 NaN 就 drop 掉; 'all': 必须全部是 NaN 才 drop
                subset=['headline'] #只删除headline列中出现空值的行，其他的不做处理
                )
                self.df_data =origin_df.reset_index(drop=True)#重置索引
            else:
                raise ValueError("Please download the file first.")

    def __len__(self):
        return self.df_data.shape[0]

    def __getitem__(self, idx):
        if self.transform:
            return self.roberta.encode(self.df_data['headline'][idx]), \
                   self.labels_list[self.df_data['category'][idx]]

        return self.df_data['headline'][idx], self.labels_list[self.df_data['category'][idx]]
