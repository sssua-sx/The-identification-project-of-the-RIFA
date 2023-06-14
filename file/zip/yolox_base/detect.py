"""
author: LSH9832
"""
from yolox import Detector, draw_bb
import cv2

if __name__ == '__main__':
    detector = Detector(
        model_path="E:/图像检测平台/webyolox-main/settings/test/output/last.pth",
        model_size="s",
        class_path="E:/图像检测平台/webyolox-main/yolox/datasets/classes.txt",
        conf=0.25,
        nms=0.4,
        input_size=640,
        fp16=True
    )

    """
    or create detector as follows
    """
    # from yolox import create_detector_from_settings
    # detector = create_detector_from_settings("./detect_settings.yaml")

    detector.loadModel()

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if success:
            result = detector.predict(image)

            draw_bb(image, result, detector.get_all_classes())
            cv2.imshow("image", image)
            if cv2.waitKey(1) == 27:    # esc
                cv2.destroyAllWindows()
                break
