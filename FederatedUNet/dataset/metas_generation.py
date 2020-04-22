import pandas as pd
import os
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", required=True)
    args = parser.parse_args()

    image_path = '/nmnt/x3-hdd/data/DA/CC359/originalScaled'
    target_path = '/nmnt/x3-hdd/data/DA/CC359/Silver-standard-MLScaled'
    if len(os.listdir(image_path)) != len(os.listdir(target_path)):
        raise AssertionError('File numbers in image and target paths are different')

    img_names = sorted(os.listdir(image_path))
    class_names = [img.split('_')[1] + '-' + img.split('_')[4][0] for img in img_names]

    target_names = sorted(os.listdir(target_path))
    meta = pd.DataFrame(columns=['id', 'img', 'target', 'class'])
    meta['img'], meta['target'], meta['class'] = img_names, target_names, class_names
    for img_class in np.unique(class_names):
        partial_meta = meta[meta['class'] == img_class].copy()
        partial_meta.reset_index(drop=True, inplace=True)
        partial_meta['id'] = partial_meta.index
        partial_meta.drop('class', axis=1, inplace=True)
        partial_meta.to_csv(os.path.join(args.save_path, 'meta_' + img_class + '.csv'), index=False)
    # meta.to_csv(args.save_path, index=False)
if __name__ == '__main__':
    main()