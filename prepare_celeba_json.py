'''MIT License
Copyright (C) 2020 Prokofiev Kirill, Intel Corporation
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

import argparse
import json
import os

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='CelebA_Spoof json preparation')
    parser.add_argument('--root', type=str, default=None, required=True,
                        help='path to root folder of the CelebA_Spoof')
    args = parser.parse_args()
    create_json(mode='train', root_folder=args.root)
    create_json(mode='test', root_folder=args.root)

def create_json(mode, root_folder):
    if mode == 'test':
        list_path = os.path.join(root_folder, 'metas/intra_test/test_label.json')
        save_file = os.path.join(root_folder, 'metas/intra_test/items_test.json')
    else:
        assert mode == 'train'
        list_path = os.path.join(root_folder, 'metas/intra_test/train_label.json')
        save_file = os.path.join(root_folder, 'metas/intra_test/items_train.json')
    indx=0
    items = {}
    with open('./datasets/small_crops.txt', 'r') as f:
        small_crops = map(lambda x: x.strip(), f.readlines())
        set_ = set(small_crops)
    with open(list_path, 'r') as f:
        data = json.load(f)
        print('Reading dataset info...')
        for indx, path in tqdm(enumerate(data), leave=False):
            labels = data[path] # create list with labels
            bbox_path = os.path.join(root_folder, os.path.splitext(path)[0] + '_BB.txt')
            bbox_f = open(bbox_path, 'r')
            bbox_info = bbox_f.readline().strip().split()[0:4]
            bbox = [int(x) for x in bbox_info] # create bbox with labels
            if len(bbox) < 4 or bbox[2] < 3 or bbox[3] < 3: # filter not existing or too small boxes
                print('Bad bounding box: ', bbox, path)
                continue
            if path in set_:
                print('Bad img cropp: ', path)
            items[indx] = {'path':path, 'labels':labels, 'bbox':bbox}
    with open(save_file, 'w') as f:
        json.dump(items, f)

if __name__ == '__main__':
    main()
