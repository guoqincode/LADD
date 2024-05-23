import os
import numpy as np
from torch.utils.data import Dataset
import warnings
import torch
warnings.filterwarnings('error', category=UserWarning, module='PIL')


import pickle

class ADDDataset(Dataset):
    def __init__(self,
                 data_root,
                pkl_name,
                 ):

        self.data_root = data_root
        self.pkl_name = pkl_name
        
        self._load_flist()
        self._length = len(self.flist)
        self.retries = 20
        


    def __len__(self):
        return self._length

    def _load_flist(self):
        print('loading file info...')
        with open(os.path.join(self.data_root, self.pkl_name), 'rb') as f:
            self.flist = pickle.load(f)

        print('file info loaded. %d images in total' % len(self.flist))




    def __getitem__(self, ind):
        cur_retry = 0
        while cur_retry < self.retries: 
            try:
                latent_path, noise_path, txt_emb_path = self.flist[ind]

                noise = torch.load(os.path.join(self.data_root, noise_path)).float()
                latent = torch.load(os.path.join(self.data_root, latent_path)).float()
                txt_emb = torch.load(os.path.join(self.data_root, txt_emb_path))
                txt_emb['encoder_hidden_states'] = txt_emb['encoder_hidden_states'].float()

            except Exception as e:
                print('error when loading file: %s' % latent_path)
                print(e)
                print('retrying...')
                if cur_retry < self.retries:
                    cur_retry += 1
                    ind = np.random.randint(0, len(self.flist)-1)
                    continue
            break
        return (latent, noise, txt_emb)
        
