import os
import cv2
import torch
import sys
import yaml


this_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_path) if not this_path in sys.path else None   #print('start from this project dir')

# import yolox.exp
from .yolox.data.data_augment import ValTransform
from .yolox.data.datasets import COCO_CLASSES
from .yolox.utils import fuse_model, postprocess, vis
from .model import Exp as myexp


class Colors:
    def __init__(self):
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()


def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness

    c1, c2 = (int(x[0]), int((x[1]))), (int(x[2]), int((x[3])))
    # print(c1,c2)
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], (c1[1] - t_size[1] - 3) if (c1[1] - t_size[1] - 3) > 0 else (c1[1] + t_size[1] + 3)
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2) if (c1[1] - t_size[1] - 3) > 0 else (c1[0], c2[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def draw_bb(img, pred, names, type_limit=None, line_thickness=2):
    if type_limit is None:
        type_limit = names
    for *xyxy, conf0, conf1, cls in pred:
        conf = conf0 * conf1
        # print(cls)
        if names[int(cls)] in type_limit:
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, img, label=label, color=colors(int(cls), True), line_thickness=line_thickness)


class DirCapture(object):
    """read image file from a dir containing images or a image file"""
    __cv2 = cv2
    __support_type = ['jpg', 'jpeg', 'png', 'bmp']
    __img_list = []

    def __init__(self, path: str = None):
        if path is not None:
            self.open(path)

    def open(self, path):
        self.__img_list = []
        if os.path.isdir(path):
            path = path[:-1] if path.endswith('/') else path
            assert os.path.isdir(path)
            from glob import glob

            for img_type in self.__support_type:
                self.__img_list += sorted(glob('%s/*.%s' % (path, img_type)))
        elif os.path.isfile(path) and '.' in path and path.split('.')[-1] in self.__support_type:
            self.__img_list = [path]
        else:
            print('wrong input')
            self.__img_list = []

    def isOpened(self):
        return bool(len(self.__img_list))

    def read(self):
        this_img_name = self.__img_list[0]
        del self.__img_list[0]
        img = self.__cv2.imread(this_img_name)
        success = img.size > 0
        return success, img

    def release(self):
        self.__img_list = []


class Predictor(object):

    def __init__(
            self,
            model,
            exp,
            cls_names=COCO_CLASSES,  # 类型名称
            trt_file=None,  # tensorRT File
            decoder=None,  # tensorRT decoder
            device="cpu",
            fp16=False,  # 使用混合精度评价
            legacy=False,  # 与旧版本兼容
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img: list):

        ratios = [min(self.test_size[0] / image.shape[0], self.test_size[1] / image.shape[1]) for image in img]

        img = torch.cat([torch.from_numpy(self.preproc(image, None, self.test_size)).unsqueeze(0) for image in img], 0)

        img = img.float()
        # print(self.device)
        if self.device == "gpu":
            img = img.cuda()
            # print(self.fp16)
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():

            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            all_output = []
            # print(outputs)
            for index in range(len(outputs)):
                output = outputs[index]
                if output is not None:
                    output[:, 0:4] /= ratios[index]
                    output = output.cpu().numpy()
                else:
                    output = []
                all_output.append(output)

        return all_output


class listString(str):

    def __init__(self, this_string: str = ''):
        super(listString, self).__init__()
        self.__this_string = this_string
        # self.__all__ = ['append', ]

    def __getitem__(self, index):
        return self.__this_string[index]

    def __repr__(self):
        return self.__this_string

    def __len__(self):
        return len(self.__this_string)

    def append(self, add_string):
        self.__this_string += add_string


class Detector(object):
    __model_size_all = ['s', 'm', 'l', 'x', 'tiny', 'nano', 'v3']
    __device = 'cpu'
    __model = None  # 模型
    __model_size = None  # 模型大小（s,m,l,x, tiny, nano, v3）
    __model_path = None  # 模型权重文件位置
    __class_names = COCO_CLASSES  # 类别名称
    __detector = None  # 检测器
    __exp = None
    __fp16 = False
    __fuse = False

    __useTRT = False  # 使用TensorRT
    __legacy = False  # To be compatible with older versions
    __auto_choose_device = True

    __tsize = 640
    __conf = 0.25
    __nms = 0.5

    __result = None
    __img_info = None

    def __init__(
            self,
            model_path: str or None = None,  # 模型权重文件位置
            model_size: str or None = None,  # 模型大小（s,m,l,x,tiny,）
            class_path: str or None = None,  # 类别文件位置
            conf: float or None = None,  # 置信度阈值
            nms: float or None = None,  # 非极大值抑制阈值
            fp16: bool = False,
            input_size: int = 640,
            autoChooseDevice: bool = True,  # 自动选择运行的设备（CPU，GPU）
    ):

        self.__model_path = model_path
        self.__model_size = model_size
        self.__fp16 = fp16
        self.__tsize = input_size
        self.__auto_choose_device = autoChooseDevice
        self.reloadModel = self.loadModel

        if class_path is not None:
            self.__load_class(class_path)
        if conf is not None:
            self.__conf = conf
        if nms is not None:
            self.__nms = nms

        self.__check_input()

        # self.

    #######################################################################################################
    # private function
    def __load_class(self, path):
        if os.path.exists(path):
            data = open(path).readlines()
            classes = []

            [classes.append(this_class[:-1] if this_class.endswith('\n') else this_class)
             if len(this_class) else None
             for this_class in data]

            self.__class_names = classes

    def __check_input(self):
        if self.__model_path is not None:
            this_error = '[model path input error]: Type of model path should be "string"!'
            assert type(self.__model_path) == str, this_error
            # print(self.__model_path)
            assert self.__model_path.endswith('.pth'), '[model path type error]:not a weight file'

        if self.__model_size is not None:
            allSizeStr = listString('[model path input error]: Available model size: ')
            [allSizeStr.append('%s, ' % this_size) for this_size in self.__model_size_all]
            assert self.__model_size in self.__model_size_all, '%s' % allSizeStr

    def __cuda(self):
        assert torch.cuda.is_available()
        assert self.__model is not None
        self.__model.cuda()
        if self.__fp16:
            self.__model.half()
        self.__model.eval()

    def __cpu(self):
        assert self.__model is not None
        self.__model.cpu()
        self.__model.eval()

    #######################################################################################################
    # public function
    def get_all_classes(self):
        return self.__class_names

    ################################################################################
    """
    you can use the following setting functions only before loading model, or you should reload model
    """

    def setModelPath(self, model_path: str) -> None:
        self.__model_path = model_path
        self.__check_input()

    def setModelSize(self, model_size: str) -> None:
        self.__model_size = model_size
        self.__check_input()

    def setClassPath(self, class_path: str) -> None:
        self.__load_class(class_path)

    def setAutoChooseDevice(self, flag: bool) -> None:
        self.__auto_choose_device = flag

    def setFuse(self, flag: bool) -> None:
        self.__fuse = flag

    def setLegacy(self, flag: bool) -> None:
        self.__legacy = flag

    def setDevice(self, device: str) -> None:
        assert device in ['cpu', 'gpu'], '[Device name error]: No device named %s' % device
        self.__device = device

    def setSize(self, size: int) -> None:
        self.__tsize = size

    def setUseTRT(self, flag:bool):
        self.__useTRT = flag

    def setFp16(self, flag:bool):
        self.__fp16 = flag
    ################################################################################
    """
    you can use the following setting functions after loading model
    """

    def setConf(self, conf: float) -> None:
        self.__conf = conf
        if self.__detector is not None:
            self.__detector.confthre = conf

    def setNms(self, nms: float) -> None:
        self.__nms = nms
        if self.__detector is not None:
            self.__detector.nmsthre = nms

    ################################################################################
    def loadModel(self) -> None:
        assert self.__model_size is not None, 'model size not declared'
        assert self.__model_path is not None, 'model path not declared'

        # 载入网络结构
        self.__exp = myexp('yolox-%s' % self.__model_size)  # yolox.exp.build.get_exp_by_name('yolox-%s' % self.__model_size)
        self.__exp.test_conf = self.__conf
        self.__exp.nmsthre = self.__nms
        self.__exp.test_size = (self.__tsize, self.__tsize)
        self.__exp.input_scale = (self.__tsize, self.__tsize)
        self.__exp.num_classes = len(self.__class_names)

        self.__model = self.__exp.get_model()

        if self.__auto_choose_device:
            if torch.cuda.is_available():
                self.__cuda()
                self.__device = "gpu"
            else:
                self.__cpu()
        else:
            if self.__device == 'cpu':
                self.__cpu()
            elif torch.cuda.is_available():
                self.__cuda()
            else:
                print('cuda is not available, use cpu')
                self.__cpu()

        trt_file = None
        decoder = None
        if not self.__useTRT:
            # 载入权重
            pt = torch.load(self.__model_path, map_location="cpu")
            # for name in pt:
            #     print(name)
            pt['classes'] = self.__class_names

            self.__model.load_state_dict(pt["model"])

            if self.__fuse:
                self.__model = fuse_model(self.__model)

        else:
            trt_file = self.__model_path
            self.__model.head.decode_in_inference = False
            decoder = self.__model.head.decode_outputs

        # 预测器
        self.__detector = Predictor(
            self.__model,
            self.__exp,
            self.__class_names,
            trt_file,
            decoder,
            self.__device,
            self.__fp16,
            self.__legacy,
        )

    def predict(self, image):
        is_single_image = False
        if self.__detector is None:
            self.loadModel()
        if not isinstance(image, list):
            image = [image.copy()]
            is_single_image = True
        else:
            image = [img.copy() for img in image]

        # 预测
        self.__result = self.__detector.inference(image)
        return self.__result[0] if is_single_image else self.__result


# setting_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'detect_settings.yaml')
# file_settings = None
# if os.path.isfile(setting_file):
#     file_settings = yaml.load(open(setting_file), yaml.Loader)
# confidence_thres = file_settings['confidence_thres'] if file_settings is not None else 0.4
# nms_thres = file_settings['nms_thres'] if file_settings is not None else 0.5
# device = file_settings['device'] if file_settings is not None else 'gpu'
# input_size = file_settings['input_size'] if file_settings is not None else 640
# auto_choose_device = file_settings['auto_choose_device'] if file_settings is not None else True
# weight_size = file_settings['weight_size'] if file_settings is not None else 's'
# model_path = file_settings['model_path'] if file_settings is not None else './best.pth'
# is_trt_file = file_settings['is_trt_file'] if file_settings is not None else False
# fp16 = file_settings['fp16'] if file_settings is not None else False
# classes_file = file_settings['classes_file'] if file_settings is not None else 'coco_classes.txt'

