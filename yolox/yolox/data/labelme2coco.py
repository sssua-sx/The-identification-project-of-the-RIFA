# coding=gbk
import os
import json
import numpy as np
import glob
import shutil
import cv2
from sklearn.model_selection import train_test_split
import labelme
import imgviz

np.random.seed(41)
# rock1
# rock2
# rock3
# rock4
# rock5
# rock6
# rock7
# rock8
# rock9
# rock10
# rock11
# sand_wave1
# sand_wave2
# sand_wave3
# sand_wave4
# sand_wave5
# sand_wave6
# sand_wave7
# sand_wave8
# sand_wave9
# sand_wave10
# sand_wave11

# 0Ϊ����
classname_to_id = {
    "formicary": 1,
    "rifa": 2,
}


# ע�����yxf
# ��Ҫ��1��ʼ�Ѷ�Ӧ��Label����д�룺��������Լ���Lable�����޸�

def Json2Img(json_path):
    label_files = glob.glob(os.path.join(json_path, "*.json"))
    for i, filename in enumerate(label_files):
        print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)

        base = os.path.splitext(os.path.basename(filename))[0]
        img_path = os.path.join(json_path, base + ".jpg")
        if os.path.exists(img_path):
            continue
        img = labelme.utils.img_data_to_arr(label_file.imageData)
        imgviz.io.imsave(img_path, img)

class Lableme2CoCo:

    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)  # indent=2 ����������ʾ

    # ��json�ļ�����COCO
    def to_coco(self, json_path_list):
        self._init_categories()
        for json_path in json_path_list:
            obj = self.read_jsonfile(json_path)
            self.images.append(self._image(obj, json_path))
            shapes = obj['shapes']
            for shape in shapes:
                annotation = self._annotation(shape)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # �������
    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # ����COCO��image�ֶ�
    def _image(self, obj, path):
        image = {}
        from labelme import utils
        img_x = utils.img_b64_to_arr(obj['imageData'])
        h, w = img_x.shape[:-1]
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path).replace(".json", ".jpg")
        return image

    # ����COCO��annotation�ֶ�
    def _annotation(self, shape):
        # print('shape', shape)
        label = shape['label']
        points = shape['points']
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(classname_to_id[label])
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = 1.0
        return annotation

    # ��ȡjson�ļ�������һ��json����
    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

    # COCO�ĸ�ʽ�� [x1,y1,w,h] ��ӦCOCO��bbox��ʽ
    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]


if __name__ == '__main__':
    # ������ԭ�����ߵ�·��
    # labelme_path = "../../../xianjin_data-3/"

    # ����ע�⣺yxf
    # ��Ҫ��labelme_path�޸�Ϊ�Լ���images��json�ļ���·��
    labelme_path = "E:/ͼ����ƽ̨/webyolox-main/yolov5/data/�����ͼ���ע���ݼ�_new/"
    Json2Img(labelme_path)
    # saved_coco_path = "../../../xianjin_data-3/"
    saved_coco_path = "E:/ͼ����ƽ̨/webyolox-main/yolox/datasets/"
    # saved_coco_path = "./"
    # Ҫ��saved_coco_path�޸�Ϊ�Լ�������COCO��·������������ҵ�ǰCOCO���ļ����½�������coco�ļ��С�
    print('reading...')
    # �����ļ�
    if not os.path.exists("%scoco/annotations/" % saved_coco_path):
        os.makedirs("%scoco/annotations/" % saved_coco_path)
    if not os.path.exists("%scoco/train2017/" % saved_coco_path):
        os.makedirs("%scoco/train2017" % saved_coco_path)
    if not os.path.exists("%scoco/val2017/" % saved_coco_path):
        os.makedirs("%scoco/val2017" % saved_coco_path)
    # ��ȡimagesĿ¼�����е�json�ļ��б�
    print(labelme_path + "/*.json")
    json_list_path = glob.glob(labelme_path + "/*.json")
    print('json_list_path: ', len(json_list_path))
    # ���ݻ���,����û������val2017��tran2017Ŀ¼������ͼƬ������imagesĿ¼��
    train_path, val_path = train_test_split(json_list_path, test_size=0.2, train_size=0.8)
    # ����yxf����ѵ��������֤���ı�����8��2�����Ը����Լ���Ҫ�ı����޸ġ�
    print("train_n:", len(train_path), 'val_n:', len(val_path))

    # ��ѵ����ת��ΪCOCO��json��ʽ
    l2c_train = Lableme2CoCo()
    train_instance = l2c_train.to_coco(train_path)
    l2c_train.save_coco_json(train_instance, '%scoco/annotations/instances_train2017.json' % saved_coco_path)
    for file in train_path:
        # shutil.copy(file.replace("json", "jpg"), "%scoco/images/train2017/" % saved_coco_path)
        # print("�������һ��file��"+file)
        img_name = file.replace('json', 'jpg')
        # print("�������һ��img_name��" + img_name)
        temp_img = cv2.imdecode(np.fromfile(img_name,dtype=np.uint8),-1)
        # print(temp_img) ����ͼ���ȡ�Ƿ���ȷ
        try:
            # ��������ԭ�����ߵĴ��룬����֮��train�ļ��������ɵ��ǿյ�
            # cv2.imwrite("{}coco/images/train2017/{}".format(saved_coco_path, img_name.replace('png', 'jpg')),temp_img)
            # ���Լ���trainͼ���·����F:\rockdata\COCO\coco\images\train2017
            img_name_jpg = img_name.replace('png', 'jpg')
            print("jpg����:" + img_name_jpg)
            filenames = img_name_jpg.split("\\")[-1]
            print(filenames)  # �����ǽ�һ��·���е��ļ�������ȡ����
            cv2.imencode('.jpg', temp_img)[1].tofile("{}coco/train2017/{}".format(saved_coco_path,filenames))
            # ���д����䣬�ǽ� X.jpg д�뵽ָ��·��./COCO/coco/images/train2017/X.jpg
        except Exception as e:
            print(e)
            print('Wrong Image:', img_name)
            continue

        print(img_name + '-->', img_name.replace('png', 'jpg'))
        # print("yxf"+img_name)

    for file in val_path:
        # shutil.copy(file.replace("json", "jpg"), "%scoco/images/val2017/" % saved_coco_path)

        img_name = file.replace('json', 'jpg')
        temp_img = cv2.imdecode(np.fromfile(img_name,dtype=np.uint8),-1)
        try:

            # cv2.imwrite("{}coco/images/val2017/{}".format(saved_coco_path, img_name.replace('png', 'jpg')), temp_img)
            img_name_jpg = img_name.replace('png', 'jpg')  # ��png�ļ��滻��jpg�ļ���
            print("jpg����:" + img_name_jpg)
            filenames = img_name_jpg.split("\\")[-1]
            print(filenames)
            cv2.imencode('.jpg', temp_img)[1].tofile("{}coco/val2017/{}".format(saved_coco_path,filenames))

        except Exception as e:
            print(e)
            print('Wrong Image:', img_name)
            continue
        print(img_name + '-->', img_name.replace('png', 'jpg'))

    # ����֤��ת��ΪCOCO��json��ʽ
    l2c_val = Lableme2CoCo()
    val_instance = l2c_val.to_coco(val_path)
    l2c_val.save_coco_json(val_instance, '%scoco/annotations/instances_val2017.json' % saved_coco_path)