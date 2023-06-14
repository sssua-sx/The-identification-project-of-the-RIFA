"""
author: LSH9832
"""
from yolox.pth2trtengine import convert


if __name__ == '__main__':
    convert(
        pth_file="best.pth",
        output_dir="yolox_output",
        model_size="yolox-s",       # yolox-[m/l/xl/tiny/nano]
        num_classes=80,
        work_space=8,               # GB
        batch_size=1
    )
