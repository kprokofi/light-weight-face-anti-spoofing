'''MIT License

Copyright (C) 2020 Prokofiev Kirill
 
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom
the Software is furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.'''

from torch.utils.data import Dataset
from .celeba_spoof import CelebASpoofDataset
from .lcc_fasd import LCFAD

class MultiDataset(Dataset):
    def __init__ (self, LCCFASD_root, Celeba_root, train=True, 
                  transform=None, LCFASD_train_protocol='combine_all', 
                  LCFASD_val_protocol='val_test'):
        if train:
            self.dataset_celeba = CelebASpoofDataset(Celeba_root, test_mode=False, 
                                                     transform=transform, multi_learning=False)
            self.dataset_lccfasd = LCFAD(LCCFASD_root, protocol=LCFASD_train_protocol, 
                                         transform=transform)
        else:
            self.dataset_celeba = CelebASpoofDataset(Celeba_root, test_mode=True, 
                                                     transform=transform,  multi_learning=False)
            self.dataset_lccfasd = LCFAD(LCCFASD_root, protocol=LCFASD_val_protocol, 
                                         transform=transform)
        self.celeba_index = set(range(len(self.dataset_celeba)))
        self.lccfasd_index = set(range(len(self.dataset_celeba), 
                                       len(self.dataset_celeba) + len(self.dataset_lccfasd)))

    def __len__(self):
        return len(self.dataset_celeba) + len(self.dataset_lccfasd)

    def __getitem__(self, indx):
        if indx in self.celeba_index:
            return self.dataset_celeba[indx]
        assert indx in self.lccfasd_index
        indx = indx - len(self.dataset_celeba)
        return self.dataset_lccfasd[indx]
