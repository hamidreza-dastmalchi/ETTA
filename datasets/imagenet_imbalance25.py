import os
from .utils import Datum, DatasetBase, listdir_nohidden

from .imagenet import read_classnames

TO_BE_IGNORED = ["README.txt"]
# template = [
#     "itap of a {}.",
#     "a bad photo of the {}.",
#     "a origami {}.",
#     "a photo of the large {}.",
#     "a {} in a video game.",
#     "art of the {}.",
#     "a photo of the small {}.",
# ]
template = ['a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
    'A real-world image of a {} in an unusual setting.',
    'A difficult-to-recognize photo of a {}.',
    'A low-quality, blurred image of a {}.',
    'A partially obstructed view of a {}.',
    'An image of a {} with challenging lighting conditions.',
    'A real-world photo of a {} from an unusual angle.',
    'A photo of a {} with background clutter.',
    'A close-up view of a {} with minimal context.',
    'A shadowed, hard-to-see {}.',
    'A natural photo of a {} partially out of frame.',
    'An obscured image of a {} blending with the background.',
    'A distant shot of a {} in complex surroundings.',
    'An ambiguous image of a {} where it is hard to distinguish.',
    'A real-world snapshot of a {} in challenging conditions.',
    'A distorted view of a {}.',
    'An image of a {} taken from an uncommon perspective.',
    'A noisy and difficult-to-interpret image of a {}.',
    'A visually complex photo with a {} in the background.',
    'A partially visible photo of a {} with distractions around it.',
    'A natural image of a {} in unexpected surroundings.']


class ImageNetImbalance25(DatasetBase):
    """ImageNet-A(dversarial).

    This dataset is used for testing only.
    """

    dataset_dir = "imagenet-imbalance25"

    def __init__(self, root):
        root = os.path.abspath(os.path.expanduser(root))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.template = template

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = read_classnames(text_file)

        data = self.read_data(classnames)

        super().__init__(test=data) 

    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = listdir_nohidden(image_dir, sort=True)
        folders = [f for f in folders if f not in TO_BE_IGNORED]
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items