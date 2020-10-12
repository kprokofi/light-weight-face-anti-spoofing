import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TAC():
    def __init__(self, classes):
        self.classes = classes
        if self.classes != 2:
            raise NotImplementedError
        self.embedings = torch.tensor([for i in range(self.classes)]).reshape(self.classes,1,1,1,1)
    def update(self, emb, target):
        target_index = target.argmax(dim=1)
        emb = emb.mean(0) # [classes, emb, 1, 1]
        for i, e in enumerate(emb):
            self.embedings[target[i]] = torch.mean(self.embedings[target[i]], e)
    def get_emb(self, target):
        target_index = target.argmax(dim=1)
        emb_matrix = torch.tensor([])
        p=[1/self.classes + 1/(self.classes**2)]*self.classes
        for ind in target_index:
            p[target_index] = 0
            index = np.random.choice(self.classes, target.size(0), p=p)
            p[target_index] = 1/(self.classes**2)
            contrast_emb = self.embedings[index]
            torch.cat((emb_matrix, contrast_emb))
        return emb_matrix

'''
for imges, target in dataloader:
    z <-- model.get_emb(imges) # new embeding
    avg_z <-- TAC.get_emb
    new_z <-- alpha*z + (1-alpha)*avg_z
    logits <-- model.make_logits(new_z)
    loss <-- loss_func(logits, target)
    loss.backward()
'''
