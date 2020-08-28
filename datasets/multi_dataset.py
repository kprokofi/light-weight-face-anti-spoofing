from torch.utils.data import Dataset
from datasets import CelebASpoofDataset, LCFAD

class MultiDataset(Dataset):
    def __init__ (self, LCCFASD_root, Celeba_root, train=True, transform=None):
        if train:
            self.dataset_celeba = CelebASpoofDataset(Celeba_root, test_mode=False, transform=transform, test_dataset=False)
            self.dataset_lccfasd = LCFAD(LCCFASD_root, protocol='combine_all', transform=transform)
        else:
            self.dataset_celeba = CelebASpoofDataset(Celeba_root, test_mode=True, transform=transform, test_dataset=False)
            self.dataset_lccfasd = LCFAD(LCCFASD_root, protocol='val', transform=transform)

        self.celeba_index = set(range(len(self.dataset_celeba)))
        self.lccfasd_index = set(range(len(self.dataset_celeba), len(self.dataset_celeba) + len(self.dataset_lccfasd)))

    def __len__(self):
        return len(self.dataset_celeba) + len(self.dataset_lccfasd)

    def __getitem__(self, indx):
        if indx in self.celeba_index:
            return self.dataset_celeba[indx]
        else:
            assert indx in self.lccfasd_index
            indx = indx - len(self.dataset_celeba)
            return self.dataset_lccfasd[indx]