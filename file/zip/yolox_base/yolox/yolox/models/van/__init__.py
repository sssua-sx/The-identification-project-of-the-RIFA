from .van import VANBackbone, partial, nn


def get_model(size, img_size):
    base_channels = [64, 256, 512, 1024]
    all_size = {
        "nano": (0.25, [1, 1, 1, 1]),
        "tiny": (0.375, [1, 1, 2, 1]),
        "s": (0.5, [1, 2, 3, 1]),
        "m": (0.75, [1, 2, 3, 2]),
        "l": (1, [2, 2, 2, 3])
    }
    if isinstance(size, str):
        assert size in all_size, "Wrong Model Size!"
    elif isinstance(size, float):
        flag = False
        for name in all_size:
            if all_size[name][0] == size:
                flag = True
                size = name
                break
        assert flag, "Wrong Size Input!"

    # print(size)

    per, depths = all_size[size]
    dims = [channel * per for channel in base_channels]
    model = VANBackbone(
        img_size=img_size,
        embed_dims=dims,
        mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=depths
    )

    return model

