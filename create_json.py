import os
import json
from tqdm import tqdm


def create_json(mode='train', root_folder='/home/prokofiev/pytorch/antispoofing/CelebA_Spoof'):
    if mode == 'test':
        list_path = os.path.join(root_folder, 'metas/intra_test/test_label.json')
        save_file = os.path.join(root_folder, 'metas/intra_test/items_test.json')
    else:
        assert mode == 'train'
        list_path = os.path.join(root_folder, 'metas/intra_test/train_label.json')
        save_file = os.path.join(root_folder, 'metas/intra_test/items_train.json')
    indx=0
    items = {}
    with open(list_path, 'r') as f:
        data = json.load(f)
        for path in tqdm(data, 'Reading dataset info...', leave=False):
            labels = data[path] # create list with labels
            bbox_path = os.path.join(root_folder, os.path.splitext(path)[0] + '_BB.txt')
            bbox_f = open(bbox_path, 'r')
            bbox_info = bbox_f.readline().strip().split()[0:4]
            bbox = [int(x) for x in bbox_info] # create bbox with labels
            if len(bbox) < 4 or bbox[2] < 3 or bbox[3] < 3: # filter not existing or too small boxes
                print('Bad bounding box: ', bbox, path)
                continue
            items[indx] = {'path':path, 'labels':labels, 'bbox':bbox}
            indx += 1
    with open(save_file, 'w') as f:
        json.dump(items, f, indent = 4)

        
if __name__ == '__main__':
    create_json(mode='train')
    create_json(mode='test')
