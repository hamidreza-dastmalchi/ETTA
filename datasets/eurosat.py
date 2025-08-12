import os

from .utils import DatasetBase
from .oxford_pets import OxfordPets


#template = ['a sattelite photo of {}.']
#template = ['a photo of {}.']
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
    'A satellite image of a {}.',
    'An aerial view of a {} captured by satellite.',
    'A high-resolution satellite photo of a {}.',
    'A satellite view showing a {} from above.',
    'A top-down image of a {} taken from space.',
    'A {} seen from a satellite perspective.',
    'A satellite image displaying a {} in detail.',
    'An overhead view of a {} from satellite.',
    'A satellite capture of a {} with surrounding areas.',
    'A satellite shot of {} terrain.',
    'A satellite image showing a {} at night.',
    'A satellite image of {} during the day.',
    'A {} in satellite imagery with high detail.',
    'A satellite view of {} land formations.',
    'A close-up satellite image of a {} region.',
    'A {} captured by satellite with natural colors.',
    'A grayscale satellite image of a {}.',
    'A satellite view of {} seasonal changes.',
    'A {} shown from a high-altitude satellite view.',
    'A broad satellite view encompassing a {}.',

    # Annual Crop
    "A satellite image of an {} showing vast fields with cultivated crops.",
    "An aerial satellite image of {} fields arranged in neat rows.",
    "A satellite view of {} with varying green and yellow crop patches.",
    "A {} land captured by satellite with irrigation lines visible.",
    "An overhead view of {} farmland during harvest season.",
    "A {} captured by satellite showing seasonal crop rotation.",
    "A top-down space image of {} displaying neatly divided plots.",

    # Forest
    "A dense {} captured in a high-resolution satellite image.",
    "A {} seen from space, displaying a vast canopy of green trees.",
    "A satellite image of a {} revealing deforestation patterns.",
    "An infrared satellite image showing the health of {} vegetation.",
    "A {} in satellite imagery with rivers cutting through.",
    "A satellite capture of a {} with mountainous terrain.",
    "A {} viewed from space showing seasonal foliage changes.",

    # Herbaceous Vegetation
    "A {} covered with patches of grass and low-lying vegetation.",
    "A {} in a satellite image showing diverse plant cover.",
    "A high-resolution satellite image of a {} displaying grasslands.",
    "A {} as seen from above, revealing open green fields.",
    "A {} captured in a multi-spectral image showing plant growth.",
    "A {} in satellite view showing dry vs lush vegetation.",
    "A {} displayed in an aerial photograph with sparse tree coverage.",

    # Highway
    "A satellite view of a {} cutting through the landscape.",
    "A {} captured from space showing intersecting roads.",
    "A {} with multiple lanes visible in an aerial image.",
    "A {} seen in a satellite shot with surrounding traffic.",
    "A {} illuminated in a satellite night view.",
    "An overhead image of a {} weaving through urban areas.",
    "A {} captured in high resolution, showing road infrastructure.",

    # Industrial Buildings
    "A {} area captured in a satellite image with large rooftops.",
    "A {} as seen from space, surrounded by factories and warehouses.",
    "A {} complex with smoke plumes visible in satellite view.",
    "A {} in an overhead space image, displaying shipping containers.",
    "A {} captured by a satellite, showing clusters of buildings.",
    "A {} viewed from above, revealing parking lots and loading docks.",
    "A {} as seen in an urban satellite scan.",

    # Pasture
    "A {} in a satellite image, showing vast grazing land.",
    "A {} with visible cattle paths in aerial photography.",
    "A {} seen from space with fenced grazing sections.",
    "A {} captured in a satellite view, showing green and brown patches.",
    "A {} region displayed in satellite imagery with rolling hills.",
    "A {} in an overhead satellite shot with scattered livestock.",
    "A {} captured with infrared imaging, highlighting vegetation growth.",

    # Permanent Crop
    "A {} in satellite imagery with structured orchard rows.",
    "A {} captured from space, revealing vineyards or groves.",
    "A {} region in an aerial view, showing plantation patterns.",
    "A {} captured in high-resolution satellite photography.",
    "A {} viewed from space, displaying irrigation networks.",
    "A {} seen in an overhead shot, with trees aligned in straight rows.",
    "A {} land in satellite imagery, highlighting long-term cultivation.",

    # Residential Buildings
    "A {} captured in a satellite image showing rooftops.",
    "A {} as seen from space, displaying urban housing density.",
    "A {} region in satellite imagery, showing a structured grid layout.",
    "A {} as seen in an aerial satellite view, with streets and houses.",
    "A {} night-time satellite image revealing city lights.",
    "A {} with high-rise buildings casting long shadows in satellite view.",
    "A {} captured from space, showing suburban and city areas.",

    # River
    "A {} winding through the landscape in a satellite image.",
    "A {} captured in a space-based image, showing water currents.",
    "A {} delta displayed in an aerial satellite scan.",
    "A {} in a satellite shot, with bridges crossing over it.",
    "A {} in a thermal satellite image, revealing water temperatures.",
    "A {} captured in a high-resolution space photograph.",
    "A {} displayed in satellite imagery, surrounded by lush vegetation.",

    # Sea/Lake
    "A {} in a satellite image, showing deep blue waters.",
    "A {} captured from space, revealing coastal erosion patterns.",
    "A {} seen in an overhead satellite view with small islands.",
    "A {} with visible waves and currents in satellite imagery.",
    "A {} captured in an infrared satellite shot, showing water temperature variations.",
    "A {} surrounded by urban development, seen from space.",
    "A {} with fishing boats visible in a high-resolution satellite image."
]








NEW_CLASSNAMES = {
    'AnnualCrop': 'Annual Crop Land',
    'Forest': 'Forest',
    'HerbaceousVegetation': 'Herbaceous Vegetation Land',
    'Highway': 'Highway or Road',
    'Industrial': 'Industrial Buildings',
    'Pasture': 'Pasture Land',
    'PermanentCrop': 'Permanent Crop Land',
    'Residential': 'Residential Buildings',
    'River': 'River',
    'SeaLake': 'Sea or Lake'
}


class EuroSAT(DatasetBase):

    dataset_dir = 'eurosat'

    def __init__(self, root):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, '2750')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_EuroSAT.json')
        
        self.template = template

        test = OxfordPets.read_split(self.split_path, self.image_dir)
        
        super().__init__(test=test)
