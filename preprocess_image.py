"""
Author: JudePark <judepark@kookmin.ac.kr>
"""
import math

from PIL import Image
from tqdm import tqdm
from itertools import groupby
from typing import *

import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch
import json
import cv2
import os


def image2tensor(image_path:str, mode='fetch') -> torch.Tensor:
    """
    :param image_path:
    :param mode:
        resize: 이미지를 리사이징해서 텐서로 반환할지
        fetch: 그냥 원본 이미지를 반환할지
    :return:
    """

    mode_constant = ['fetch', 'resize']
    if mode not in mode_constant:
        raise ValueError('invalid mode input. check again.')

    if mode == 'fetch':
        img = Image.open(image_path).convert('RGB')
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
             torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        # C X H X W
        return transform(img)

    elif mode == 'resize':
        # TODO => resize method 도 추가할 것
        pass


def process2dataset(image_folder_path: str, json_path: str) -> Union[groupby, Iterator[Tuple[Any, Iterator]]]:
    if image_folder_path[-1] != '/':
        raise ValueError("path should be ended with '/'.")
    image_file_name = (list(filter(lambda x: x.split('.')[1] == 'png', os.listdir(image_folder_path))))

    with open(json_path, 'r') as f:
        dataset_json = json.load(f)
        properties = [feature['properties'] for feature in tqdm(dataset_json['features'], 'features')]

        '''
        examples for use
        for image_id, group in groups:
            print(image_id, list(group))
        '''

        image_id_grouped = groupby(sorted(properties, key=lambda x: x['image_id']), key=lambda property: property['image_id'])
        labels = {}
        dataset = []

        for id, properties in tqdm(image_id_grouped):
            properties = list(properties)
            rbox = []

            for entity in properties:
                (cx, cy), (width, height), theta = cv2.minAreaRect(np.array(entity['bounds_imcoords'].split(','))
                                                  .astype(dtype=np.float32)
                                                  .reshape((-1, 2)))

                if width < height:
                    width, height = height, width
                    theta += 90

                # (cx, cy, width, height, theta) 값을 반환
                rbox.append([cx, cy, width, height, math.radians(theta)])

            type_id = [entity['type_id'] for entity in properties]
            type_name = [entity['type_name'] for entity in properties]
            labels[id] = [rbox, type_id, type_name]

        for name in tqdm(image_file_name):
            img_tensor = image2tensor(image_folder_path + name)

            # 이미지 안에 객체가 있을 경우만 넣도록 하자.
            if name in labels:
                dataset.append((img_tensor, labels[name]))


        # (name, img_tensor, labels[name])
        return dataset


if __name__ == '__main__':
    # tensor = image2tensor('./rsc/sample_images/0.png', 'fetch')
    # print(tensor.permute(1, 2, 0).shape)
    # plt.imshow(tensor.permute(1, 2, 0))
    # plt.show()
    # plt.imshow([0.5, 0.5, 0.5] * tensor.permute(1, 2, 0).numpy() + [0.5, 0.5, 0.5]) # 3000 x 3000 x 3
    # plt.show()
    print(process2dataset('./rsc/sample_images/', './rsc/sample_images/labels.json')[0])
    pass
