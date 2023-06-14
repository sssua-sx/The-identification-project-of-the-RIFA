from .batch_detect import *
from multiprocessing import Process, Manager, freeze_support
import cv2
import subprocess as sp


__all__ = ["multiDetectRtmpServer", "Detector", "draw_bb","create_detector_from_settings"]


def create_detector_from_settings(file_name="detect_settings.yaml") -> Detector:
    setting_file = file_name
    file_settings = None
    if os.path.isfile(setting_file):
        file_settings = yaml.load(open(setting_file), yaml.Loader)
    confidence_thres = file_settings['confidence_thres'] if file_settings is not None else 0.4
    nms_thres = file_settings['nms_thres'] if file_settings is not None else 0.5
    device = file_settings['device'] if file_settings is not None else 'gpu'
    input_size = file_settings['input_size'] if file_settings is not None else 640
    auto_choose_device = file_settings['auto_choose_device'] if file_settings is not None else True
    weight_size = file_settings['weight_size'] if file_settings is not None else 's'
    model_path = file_settings['model_path'] if file_settings is not None else './best.pth'
    is_trt_file = file_settings['is_trt_file'] if file_settings is not None else False
    fp16 = file_settings['fp16'] if file_settings is not None else False
    classes_file = file_settings['classes_file'] if file_settings is not None else 'coco_classes.txt'

    weight_type = weight_size
    detector = Detector(
        model_path=model_path,
        model_size=weight_type,
        class_path=classes_file,
        conf=confidence_thres,
        nms=nms_thres,
        autoChooseDevice=auto_choose_device
    )

    # detector.setConf(0.4)
    # detector.setNms(0.5)
    detector.setDevice(device)
    detector.setSize(input_size)
    detector.setUseTRT(is_trt_file)
    detector.setFp16(fp16)
    # detector.setAutoChooseDevice(True)
    # detector.setModelPath('weights/yolox_s.pth')
    return detector




def get_rtmp_command(url, size, fps, bitrate=7000000):
    sizeStr = str(size[0]) + 'x' + str(size[1])
    command = [
        'ffmpeg',
        '-y',
        '-c', 'copy',  # copy图像的质量会更好
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-max_delay', str(100),
        '-s', sizeStr,
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast',
        '-f', 'flv',
        '-g', '5',
        '-b', '%d' % bitrate,
        url
    ]
    return command


def cap(my_dict, this_source, func=None, *kwargs):
    from time import time

    cam = cv2.VideoCapture(this_source)

    while not my_dict['run']:
        pass

    this_result = []
    frames = []
    fps = my_dict["fps_%s" % this_source]
    rtmp_address = my_dict["rtmp_%s" % this_source]
    bitrate = my_dict["bitrate"]
    is_first = True
    pipe_push = None

    t0 = time()
    while cam.isOpened() and my_dict["run"]:
        success, frame = cam.read()
        if success:
            frames.append(frame)
            my_dict["img_%s" % this_source] = frame
            my_dict["updated_%s" % this_source] = True

            if my_dict["result_updated_%s" % this_source]:
                this_result = my_dict["result_%s" % this_source]
                my_dict["result_updated_%s" % this_source] = False
                use_func = True
            else:
                use_func = False

            if len(this_result):
                while len(frames) > 1:
                    frame = frames.pop(0)
                draw_bb(frame, this_result, my_dict['classes'])

            # cv2.imshow(this_source, frame)
            # cv2.waitKey(1)

            while (time() - t0) * fps < 0.95:
                pass
            t0 = time()

            # rtmp
            if is_first:
                is_first = False
                h, w, _ = frame.shape
                pipe_push = sp.Popen(get_rtmp_command(rtmp_address, (w, h), fps, bitrate), stdin=sp.PIPE, shell=False)

            pipe_push.stdin.write(frame.tostring())

            if func is not None and use_func:
                func(this_result, kwargs)

        else:
            cam.release()
            cam.open(this_source)


def n_detect(my_dict, source_list: list, setting_file_name="detect_settings.yaml"):

    detector = create_detector_from_settings(setting_file_name)
    is_first = True
    image_list = []

    detector.loadModel()
    my_dict["classes"] = detector.get_all_classes()
    my_dict['run'] = True

    while my_dict['run']:  # 如果程序仍需要运行

        if "conf" in my_dict:
            detector.setConf(my_dict["conf"])
        if "nms" in my_dict:
            detector.setNms(my_dict["nms"])

        if is_first:
            is_first = False
            for single_source in source_list:
                while not my_dict["updated_%s" % single_source]:
                    pass
                image_list.append(my_dict["img_%s" % single_source])
                my_dict["updated_%s" % single_source] = False
        else:
            for single_source in source_list:
                if my_dict["updated_%s" % single_source]:
                    image_list[source_list.index(single_source)] = my_dict["img_%s" % single_source]
                    my_dict["updated_%s" % single_source] = False

        result = detector.predict(image_list)  # 推理
        # print("result", result)
        for this_source in source_list:
            my_dict["result_%s" % this_source] = result[source_list.index(this_source)]  # 存储结果
            my_dict["result_updated_%s" % this_source] = True
    # print("detect quit")

class multiDetectRtmpServer:
    __all_sources = []
    __all_fps = []
    __all_rtmp_address = []
    __all_func = []
    __all_kwargs = []
    __flags = []

    __extra_funcs = []
    __extra_args = []

    __bitrate = 3000000

    def __init__(self, bitrate=3000000, detector_setting_file_name="detect_settings.yaml"):
        self.set_bitrate(bitrate)
        self.__detector_setting_file = detector_setting_file_name

    def add_source(self, source: str or int, url: str, fps: int, func=None, args=(), is_process=False):
        if args != ():
            assert isinstance(args, tuple)
            assert func is not None
        self.__all_sources.append(source)
        self.__all_rtmp_address.append(url)
        self.__all_fps.append(fps)
        self.__all_func.append(func)
        self.__all_kwargs.append(args)
        self.__flags.append(is_process)

    def add_extra_process(self, func, args=()):
        self.__extra_funcs.append(func)
        self.__extra_args.append(args)

    def set_bitrate(self, bitrate=3000000):
        self.__bitrate = bitrate

    def run(self):
        freeze_support()
        n_data = Manager().dict()
        n_data["run"] = False
        n_data["bitrate"] = self.__bitrate

        for this_source in self.__all_sources:
            n_data["updated_%s" % this_source] = False
            n_data["result_updated_%s" % this_source] = False
            n_data["fps_%s" % this_source] = self.__all_fps[self.__all_sources.index(this_source)]
            n_data["rtmp_%s" % this_source] = self.__all_rtmp_address[self.__all_sources.index(this_source)]
        Processes = []

        for this_source in self.__all_sources:
            index = self.__all_sources.index(this_source)
            flag = self.__flags[index]
            if flag:
                Processes.append(Process(target=cap, args=(n_data, this_source)))
                Processes.append(Process(target=self.__all_func[index], args=(n_data, *self.__all_kwargs[index])))
            else:
                Processes.append(
                    Process(
                        target=cap,
                        args=(n_data, this_source, self.__all_func[index], *self.__all_kwargs[index])
                    )
                )

        Processes.append(Process(target=n_detect, args=(n_data, self.__all_sources, self.__detector_setting_file)))

        for func, args in zip(self.__extra_funcs, self.__extra_args):
            Processes.append(Process(target=func, args=(n_data, *args)))

        print(len(Processes))

        [process.start() for process in Processes]
        [process.join() for process in Processes]


def main():

    server = multiDetectRtmpServer(bitrate=4200000)

    server.add_source(source="/home/uagv/Videos/test.mp4", url="rtmp://localhost/live/test1", fps=30)
    server.add_source(source="/home/uagv/Videos/car1.avi", url="rtmp://localhost/live/test2", fps=30)
    server.add_source(source="/home/uagv/Videos/6.mp4", url="rtmp://localhost/live/test3", fps=30)

    server.run()


if __name__ == '__main__':
    main()
