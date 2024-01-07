# 使用本脚本将labelme标注的分割数据集转换为yolov5使用的格式   
import os, cv2, json, string
import numpy as np

classes = ['bubble', 'stripe','spot','chromatism']
txt_path = 'E:\detect_looks\datasets\data\labels\\train'
json_path = 'E:\detect_looks\datasets\data\json'
pic_path = 'E:\detect_looks\datasets\data\images\\train'
cnt = 0
path_list = [i.split('.')[0] for i in os.listdir(pic_path)]
for path in path_list:
    image = cv2.imread(f'{pic_path}/{path}.jpg')
    print(pic_path,'/',path,'.jpg')
    # cv2.imshow('image', image)
    # cv2.waitKey()
    h, w, c = image.shape
    print("h, w, c: ", h, w, c)

    with open(f'{json_path}/{path}.json') as f:
        masks = json.load(f)['shapes']
    with open(f'{txt_path}/{path}.txt', 'w+') as f:
        for idx, mask_data in enumerate(masks):
            mask_label  = mask_data['label']
            mask_label = mask_label.rstrip(string.digits)
            print(mask_label)
            mask = np.array([np.array(i) for i in mask_data['points']], dtype=np.float64)
            mask[:, 0] /= w
            mask[:, 1] /= h
            mask = mask.reshape((-1))
            if idx != 0:
                f.write('\n')
            f.write(f'{classes.index(mask_label)} {" ".join(list(map(lambda x:f"{x:.6f}", mask)))}')
    # cnt += 1
    # if cnt >= 5:
    #     break