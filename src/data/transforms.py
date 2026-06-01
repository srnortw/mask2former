import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(cfg):
    aug = cfg.augmentation
    t = []

    if aug.train.horizontal_flip.enabled:
        t.append(A.HorizontalFlip(p=aug.train.horizontal_flip.p))

    if aug.train.shift_scale_rotate.enabled:
        s = aug.train.shift_scale_rotate
        t.append(A.ShiftScaleRotate(
            shift_limit=s.shift_limit,
            scale_limit=s.scale_limit,
            rotate_limit=s.rotate_limit,
            border_mode=s.border_mode,
            p=s.p,
        ))

    if aug.train.random_resized_crop.enabled:
        r = aug.train.random_resized_crop
        t.append(A.RandomResizedCrop(
            size=(cfg.data.dataloader.img_size, cfg.data.dataloader.img_size),
            scale=(r.scale_min, r.scale_max),
            p=r.p,
        ))

    t.append(A.PadIfNeeded(
        min_height=cfg.data.dataloader.img_size,
        min_width=cfg.data.dataloader.img_size,
        border_mode=0,
    ))
    t.append(A.Resize(
        height=cfg.data.dataloader.img_size,
        width=cfg.data.dataloader.img_size,
    ))

    if aug.train.random_brightness_contrast.enabled:
        b = aug.train.random_brightness_contrast
        t.append(A.RandomBrightnessContrast(
            brightness_limit=b.brightness_limit,
            contrast_limit=b.contrast_limit,
            p=b.p,
        ))

    if aug.train.hue_saturation_value.enabled:
        h = aug.train.hue_saturation_value
        t.append(A.HueSaturationValue(
            hue_shift_limit=h.hue_shift_limit,
            sat_shift_limit=h.sat_shift_limit,
            val_shift_limit=h.val_shift_limit,
            p=h.p,
        ))

    if aug.train.gaussian_blur.enabled:
        g = aug.train.gaussian_blur
        t.append(A.GaussianBlur(
            blur_limit=(g.blur_limit_min, g.blur_limit_max),
            p=g.p,
        ))

    if aug.train.gauss_noise.enabled:
        gn = aug.train.gauss_noise
        t.append(A.GaussNoise(
            var_limit=(gn.var_limit_min, gn.var_limit_max),
            p=gn.p,
        ))

    if aug.train.random_shadow.enabled:
        t.append(A.RandomShadow(p=aug.train.random_shadow.p))

    t.append(A.Normalize(mean=aug.normalize.mean, std=aug.normalize.std))
    t.append(ToTensorV2())

    return A.Compose(t, additional_targets={"masks": "masks"})


def get_val_transforms(cfg):
    aug = cfg.augmentation
    return A.Compose([
        A.Resize(
            height=cfg.data.dataloader.img_size,
            width=cfg.data.dataloader.img_size,
        ),
        A.Normalize(mean=aug.normalize.mean, std=aug.normalize.std),
        ToTensorV2(),
    ], additional_targets={"masks": "masks"})
