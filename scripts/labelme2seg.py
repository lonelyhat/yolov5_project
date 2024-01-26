import os, cv2, json
import numpy as np
print(cv2.__file__)

classes = ['Parking Stand Area', 'Apron', 'Buildings', 'Taxiway', 'Runway', 'Runway Marking', 'Blast pad/ Stopway']

base_path = '../dataset/labelme_dataset'
path_list = [i.split('.')[0] for i in os.listdir(base_path)]

path_list_new = []

for path in path_list:
    if path != '':
        path_list_new.append(path)

print(path_list_new)

for path in path_list_new:
    print(f'{base_path}/{path}.jpg')
    image = cv2.imread(f'{base_path}/{path}.jpg')
    h, w, c = image.shape
    with open(f'{base_path}/{path}.json') as f:
        masks = json.load(f)['shapes']
    with open(f'{base_path}/{path}.txt', 'w+') as f:
        for idx, mask_data in enumerate(masks):
            mask_label = mask_data['label']
            if '_' in mask_label:
                mask_label = mask_label.split('_')[0]
            mask = np.array([np.array(i) for i in mask_data['points']])
            mask[:, 0] /= w
            mask[:, 1] /= h
            mask = mask.reshape((-1))
            if idx != 0:
                f.write('\n')
            f.write(f'{classes.index(mask_label)} {" ".join(list(map(lambda x:f"{x:.6f}", mask)))}')