from torch.utils.data import Dataset
from datasets import CelebASpoofDataset, LCFAD

class MultiDataset(Dataset):
    def __init__ (self, LCCFASD_root, Celeba_root, train=True, transform=None):
        if train:
            self.dataset_celeba = CelebASpoofDataset(Celeba_root, test_mode=False, transform=transform, test_dataset=False)
            self.dataset_lccfasd = LCFAD(LCCFASD_root, train=True, transform=transform)
        else:
            self.dataset_celeba = CelebASpoofDataset(Celeba_root, test_mode=True, transform=transform, test_dataset=False)
            self.dataset_lccfasd = LCFAD(LCCFASD_root, train=False, transform=transform)

    def __len__(self):
        return len(self.dataset_celeba) + len(self.dataset_lccfasd)

    def __getitem__(self, indx):
        celeba_index = set(range(len(self.dataset_celeba)))
        lccfasd_index = set(range(len(self.dataset_celeba), len(self.dataset_celeba) + len(self.dataset_lccfasd)))
        if indx in celeba_index:
            return self.dataset_celeba[indx]
        else:
            assert indx in lccfasd_index
            indx = indx - len(self.dataset_celeba)
            return self.dataset_lccfasd[indx]