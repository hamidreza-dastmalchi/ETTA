import os

from .utils import DatasetBase

from .oxford_pets import OxfordPets


# template = ['a photo of a person doing {}.']
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
    'A video of someone performing {}.',
    'A clip showing a person engaged in {}.',
    'A video showing the sport of {}.',
    'Footage of people participating in {}.',
    'A recording of someone doing {}.',
    'An action video of a person involved in {}.',
    'A clip of an athlete performing {}.',
    'A video showing the activity of {}.',
    'An outdoor scene of someone doing {}.',
    'A video of a person practicing {} indoors.',
    'A video showcasing the sport of {}.',
    'A dynamic video of {} in action.',
    'A person performing {} in a competition.',
    'A training video of someone doing {}.',
    'An example video of {} as an activity.',
    'A close-up video of a person performing {}.',
    'A slow-motion video of {} in action.',
    'An intense video clip of {} being done.',
    'A video showing someone skillfully doing {}.',
    'A video of {} in a gym or studio setting.'
    ]


class UCF101(DatasetBase):

    dataset_dir = 'ucf101'

    def __init__(self, root):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'UCF-101-midframes')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_UCF101.json')

        self.template = template

        test = OxfordPets.read_split(self.split_path, self.image_dir)
    
        super().__init__(test=test)
