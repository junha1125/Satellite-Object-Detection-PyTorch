from matplotlib import pyplot as plt
import numpy as np
import argparse
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import pandas as pd
from PIL import ImageFont
from PIL import ImageDraw



def save_image_with_box(box_info, imageName , start_index, last_index , 
                        image_path, save_path):
    """
    box_info : panda로 받아드린 csv파일
    image_Name : image_Num.png 파일을 봐야한다. 
    start, last : box_info['file_name'][start ~ last]가  image_Num.png에 대한 정보를 담고 있다. 
    img_path : 이미지가 있는 directory name
    save_path : 이미지가 저장될 directory name
    """
    try:
        im = Image.open(image_path + imageName)

    except :
        print("no such file in directory : ", imageName)
        return
    plt.figure(figsize = (30,30))
    plt.imshow(im)
    color_set = ['r','b','y','g']
    for i in range(start_index, last_index+1):
        point1 = (box_info["point1_x"][i],box_info["point1_y"][i])
        point2 = (box_info["point2_x"][i],box_info["point2_y"][i])
        point3 = (box_info["point3_x"][i],box_info["point3_y"][i])
        point4 = (box_info["point4_x"][i],box_info["point4_y"][i])
        plt.plot([point1[0],point2[0]],[point1[1],point2[1]], linewidth=3, color = color_set[box_info['class_id'][i] - 1])
        plt.plot([point2[0],point3[0]],[point2[1],point3[1]], linewidth=3, color = color_set[box_info['class_id'][i] - 1])
        plt.plot([point3[0],point4[0]],[point3[1],point4[1]], linewidth=3, color = color_set[box_info['class_id'][i] - 1])
        plt.plot([point4[0],point1[0]],[point4[1],point1[1]], linewidth=3, color = color_set[box_info['class_id'][i] - 1])

    plt.savefig(save_path  + imageName)
    plt.close()
    print("saved : ", imageName)


if __name__ == "__main__":
    # 기본 설정
    # image_path = './Alpha-project/images/'
    # save_path = './Alpha-project/images_with_boxs/'
    # csv_path = './Alpha-project/baseline.csv'
    parser = argparse.ArgumentParser(description='draw_rbox_in_images')
    parser.add_argument('--image_path', type=str, default='images')
    parser.add_argument('--save_path', type=str, default='images_with_boxs')
    parser.add_argument('--csv_path', type=str, default='baseline.csv')

    args = parser.parse_args()
    image_path = args.image_path + '/'  # imagepath == images/
    save_path = args.save_path + '/'    # save_path == images_with_boxs/
    csv_path = args.csv_path

    #csv file load
    box_info = pd.read_csv(csv_path)

    #run
    start_index = 0
    for i in range(len(box_info)+1):
        try:
            # i+1번째 파일이 다른 이미지라면, i번째 파일에 대해서 박스가 처진 그림을 그린다. 
            if box_info['file_name'][i][0:-4] != box_info['file_name'][i+1][0:-4]:  
                save_image_with_box(box_info, box_info['file_name'][i] , start_index, i,
                                    image_path ,save_path )
                start_index = i+1
        except:
            # box_info['file_name'][i+1]가 존재하지 않으면 , 즉 999.png 이후를 바라본다면
            save_image_with_box(box_info,box_info['file_name'][i],start_index,i,
                                image_path ,save_path )