from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class CropDeskewConfig:
    enabled: bool = True
    blur_kernel_size: int = 5
    canny_threshold1: int = 75
    canny_threshold2: int = 200
    morph_kernel_size: int = 20
    min_area_ratio: float = 0.10
    perspective_padding: int = 10
    max_skew_angle: float = 45.0


@dataclass
class CLAHEConfig:
    clip_limit: float = 2.0
    tile_grid_size: list[int] = field(default_factory=lambda: [8, 8])


@dataclass
class BinarizeConfig:
    enabled: bool = False   # False → output grayscale; True → pure black/white
    block_size: int = 11
    c_constant: int = 2


@dataclass
class DenoiseConfig:
    enabled: bool = True
    kernel_size: int = 3


@dataclass
class EnhanceConfig:
    enabled: bool = True
    clahe: CLAHEConfig = field(default_factory=CLAHEConfig)
    binarize: BinarizeConfig = field(default_factory=BinarizeConfig)
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)


@dataclass
class BlankPageConfig:
    enabled: bool = True
    white_pixel_ratio_threshold: float = 0.99


@dataclass
class OrientationConfig:
    enabled: bool = True
    gradient_xy_threshold: float = 0.85  # SobelX/SobelY < this → flag wrong orientation


@dataclass
class BarcodeConfig:
    enabled: bool = True


@dataclass
class DetectConfig:
    blank_page: BlankPageConfig = field(default_factory=BlankPageConfig)
    orientation: OrientationConfig = field(default_factory=OrientationConfig)
    barcode: BarcodeConfig = field(default_factory=BarcodeConfig)


@dataclass
class PreprocessConfig:
    crop_deskew: CropDeskewConfig = field(default_factory=CropDeskewConfig)
    enhance: EnhanceConfig = field(default_factory=EnhanceConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PreprocessConfig:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> PreprocessConfig:
        config = cls()

        if cd := data.get("crop_deskew"):
            config.crop_deskew = CropDeskewConfig(
                enabled=cd.get("enabled", True),
                blur_kernel_size=cd.get("blur_kernel_size", 5),
                canny_threshold1=cd.get("canny_threshold1", 75),
                canny_threshold2=cd.get("canny_threshold2", 200),
                morph_kernel_size=cd.get("morph_kernel_size", 20),
                min_area_ratio=cd.get("min_area_ratio", 0.10),
                perspective_padding=cd.get("perspective_padding", 10),
                max_skew_angle=cd.get("max_skew_angle", 45.0),
            )

        if en := data.get("enhance"):
            cl = en.get("clahe", {})
            bz = en.get("binarize", {})
            dn = en.get("denoise", {})
            config.enhance = EnhanceConfig(
                enabled=en.get("enabled", True),
                clahe=CLAHEConfig(
                    clip_limit=cl.get("clip_limit", 2.0),
                    tile_grid_size=cl.get("tile_grid_size", [8, 8]),
                ),
                binarize=BinarizeConfig(
                    enabled=bz.get("enabled", False),
                    block_size=bz.get("block_size", 11),
                    c_constant=bz.get("c_constant", 2),
                ),
                denoise=DenoiseConfig(
                    enabled=dn.get("enabled", True),
                    kernel_size=dn.get("kernel_size", 3),
                ),
            )

        if det := data.get("detect"):
            bp = det.get("blank_page", {})
            config.detect = DetectConfig(
                blank_page=BlankPageConfig(
                    enabled=bp.get("enabled", True),
                    white_pixel_ratio_threshold=bp.get("white_pixel_ratio_threshold", 0.99),
                ),
                orientation=OrientationConfig(
                    enabled=det.get("orientation", {}).get("enabled", True),
                    gradient_xy_threshold=det.get("orientation", {}).get("gradient_xy_threshold", 0.85),
                ),
                barcode=BarcodeConfig(
                    enabled=det.get("barcode", {}).get("enabled", True),
                ),
            )

        return config
