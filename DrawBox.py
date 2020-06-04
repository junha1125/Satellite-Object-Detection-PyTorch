#!/usr/bin/env python
# coding: utf-8

from matplotlib import pyplot as plt
import numpy
import argparse
from PIL import Image
import json
import argparse

parser = argparse.ArgumentParser(description='DrawBox')
parser.add_argument('--image_path', type=str, default='boxPrac2')
parser.add_argument('--json_file', type=str , default='labels_original.json')
parser.add_argument('--save_path', type=str, default='results_prac2')

args = parser.parse_args()
image_path = args.image_path + '/'
save_path = args.save_path +'/'




image_dic = {}
with open(image_path  + 'lables.json') as json_file:
  json_datas = json.load(json_file)
  for json_data in json_datas['features']:
    image_id = json_data['properties']['image_id']
    box_position = json_data['properties']['bounds_imcoords']
    try:
      image_dic[image_id]['box_position'].append(box_position)
      image_dic[image_id]['type_id'].append(json_data['properties']['type_id'])
    except KeyError:
      image_dic[image_id] ={}
      image_dic[image_id]['box_position'] = []
      image_dic[image_id]['box_position'].append(box_position)
      image_dic[image_id]['type_id'] = []
      image_dic[image_id]['type_id'].append(json_data['properties']['type_id'])




for image_name in image_dic.keys():
    im = Image.open(image_path  + image_name)
    plt.figure(figsize = (15,15))
    plt.imshow(im)
    color_set = ['r','b','y','w']
    box_position = image_dic[image_name]['box_position']
    for i, box in enumerate(box_position):
        if box == None:
            continue
        x_values = [box[0],box[2],box[4],box[6],box[0]]
        y_values = [box[1],box[3],box[5],box[7],box[1]]
        plt.plot(x_values, y_values, linewidth=3, color = color_set[image_dic[image_name]['type_id'][i] - 1])
    plt.savefig(save_path  + str(image_name))
    plt.close()
    print("save : ", image_name)








