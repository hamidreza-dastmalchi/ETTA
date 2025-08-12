from .oxford_pets import OxfordPets
from .eurosat import EuroSAT
from .ucf101 import UCF101
from .sun397 import SUN397
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .fgvc import FGVCAircraft
from .food101 import Food101
from .oxford_flowers import OxfordFlowers
from .stanford_cars import StanfordCars
from .imagenetv2 import ImageNetV2
from .imagenet_a import ImageNetA
from .imagenet_r import ImageNetR
from .imagenet_sketch import ImageNetSketch
from .imagenet import ImageNet
from .imagenet_blur5 import ImageNetBlur5
from .imagenet_imbalance50 import ImageNetImbalance50
from .imagenet_imbalance10 import ImageNetImbalance10
from .imagenet_imbalance25 import ImageNetImbalance25


dataset_list = {
                "oxford_pets": OxfordPets,
                "eurosat": EuroSAT,
                "ucf101": UCF101,
                "sun397": SUN397,
                "caltech101": Caltech101,
                "dtd": DescribableTextures,
                "fgvc": FGVCAircraft,
                "food101": Food101,
                "oxford_flowers": OxfordFlowers,
                "stanford_cars": StanfordCars,
                "imagenet-a": ImageNetA,
                "imagenet-v": ImageNetV2,
                "imagenet-r": ImageNetR,
                "imagenet-s": ImageNetSketch,
                "imagenet-i": ImageNet,
                "imagenet-blur5": ImageNetBlur5,
                "imagenet-imbalance50": ImageNetImbalance50,
                "imagenet-imbalance10": ImageNetImbalance10,
                "imagenet-imbalance25": ImageNetImbalance25,
                }


def build_dataset(dataset, root_path):
    return dataset_list[dataset](root_path)