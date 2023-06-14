"""
author: LSH9832
reference: https://github.com/Megvii-BaseDetection/YOLOX
"""
import os
# import shutil
from loguru import logger

import tensorrt as trt
import torch
from torch2trt import torch2trt

from .model import Exp
import time


@logger.catch
def convert(
        pth_file,
        output_dir="yolox_output",
        model_size="yolox-s",
        num_classes=80,
        work_space=8,
        batch_size=1,
):
    exp = Exp(exp_name=model_size)
    exp.num_classes = num_classes
    logger.info("class num: %d" % exp.num_classes)

    time.sleep(3)

    model = exp.get_model()
    file_name = os.path.join(output_dir, model_size.replace("-", "_"))
    os.makedirs(file_name, exist_ok=True)

    ckpt = torch.load(pth_file, map_location="cpu")

    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")
    model.eval()
    model.cuda()
    model.head.decode_in_inference = False
    logger.info("test size: (%d, %d)." % (exp.test_size[0], exp.test_size[1]))
    x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
    model_trt = torch2trt(
        model,
        [x],
        fp16_mode=True,
        log_level=trt.Logger.INFO,
        max_workspace_size=((1 << 30) * work_space),
        max_batch_size=batch_size,
    )
    torch.save(model_trt.state_dict(), os.path.join(file_name, "model_trt.pth"))
    logger.info("Converted TensorRT model done.")
    engine_file = os.path.join(file_name, "model_trt.engine")

    with open(engine_file, "wb") as f:
        f.write(model_trt.engine.serialize())

    logger.info("Converted TensorRT model engine file is saved for C++ inference.")


if __name__ == "__main__":
    convert("yolox_s.pth")
