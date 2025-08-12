import os
# from transformers import LlamaForCausalLM, LlamaTokenizer
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
from .utils import DatasetBase
from .oxford_pets import OxfordPets
import torch
from huggingface_hub import login
# login()
# template = ['a photo of {}, a type of food.']
# template = ['a photo of a {}.']



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
    'A close-up photo of a plate of {}.',
    'A delicious-looking dish of {}.',
    'A high-quality image of freshly prepared {}.',
    'A beautifully plated {} dish.',
    'A mouth-watering shot of {}.',
    'A meal featuring {} as the main ingredient.',
    'A photo of a {} on a serving plate.',
    'A bowl of freshly made {}.',
    'A close-up of a {} garnished with herbs.',
    'A serving of {} on a rustic table.',
    'An artistic presentation of {}.',
    'A top-down view of a plate of {}.',
    'A close-up of a {} on a cutting board.',
    'A steaming hot {} ready to be eaten.',
    'A restaurant-quality serving of {}.',
    'A picture of homemade {}.',
    'A dish of {} with sides and garnishes.',
    'An image of {} served on fine china.',
    'A freshly cooked plate of {}.',
    'A savory {} presented in a minimalist style.']


# template = [
#     "a photo of a delicious {}.",
#     "a close-up photo of a {}.",
#     "a photo of a {} on a plate.",
#     "a photo of a {} served in a bowl.",
#     "a photo of a {} on a table.",
#     "a photo of a {} with garnish.",
#     "a photo of a {} being prepared.",
#     "a photo of a {} in a restaurant.",
#     "a photo of a {} on a cutting board.",
#     "a photo of a {} with a fork and knife.",
#     "a photo of a {} with a drink next to it.",
#     "a photo of a {} being cooked.",
#     "a photo of a {} fresh out of the oven.",
#     "a photo of a {} with steam rising from it.",
#     "a photo of a {} in a kitchen.",
#     "a photo of a {} in natural light.",
#     "a photo of a {} with vibrant colors.",
#     "a photo of a {} at a picnic.",
#     "a photo of a {} in a takeout container.",
#     "a photo of a {} in a buffet.",
#     "a photo of a {} being served.",
#     "a photo of a {} with sauce on it.",
#     "a photo of a half-eaten {}.",
#     "a photo of a {} with fresh ingredients.",
#     "a photo of a {} with a side dish.",
#     "a photo of a {} in a food truck.",
#     "a photo of a {} being held in someone's hand.",
#     "a photo of a {} on a wooden table.",
#     "a photo of a {} on a fancy plate.",
#     "a photo of a {} at a food festival.",
#     "a close-up photo of a {}'s texture.",
#     "a photo of a {} with chopsticks.",
#     "a photo of a {} with melted cheese.",
#     "a photo of a {} with fresh herbs.",
#     "a photo of a {} in a bowl with soup.",
#     "a photo of a {} on a grill.",
#     "a photo of a {} with a crispy crust.",
#     "a photo of a {} with powdered sugar on top.",
#     "a photo of a {} on a colorful background.",
#     "a photo of a {} in a minimalist setting.",
#     "a photo of a {} with a napkin next to it.",
#     "a photo of a {} at a dinner party.",
#     "a photo of a {} in a fast-food setting.",
#     "a photo of a homemade {}.",
#     "a photo of a {} on a rustic table.",
#     "a photo of a {} with a knife slicing it.",
#     "a photo of a {} being plated by a chef.",
#     "a photo of a {} on a white plate.",
#     "a photo of a {} with a creamy texture.",
#     "a photo of a {} stacked in layers.",
#     "a photo of a {} with fresh fruit.",
#     "a photo of a {} with chocolate on it.",
#     "a photo of a spicy {}.",
#     "a photo of a {} at breakfast.",
#     "a photo of a {} at lunch.",
#     "a photo of a {} at dinner.",
#     "a photo of a street vendor selling {}.",
#     "a photo of a {} in a stylish restaurant.",
#     "a photo of a {} served with a smile.",
#     "a photo of a {} in a fancy setting.",
#     "a photo of a {} in a casual setting.",
#     "a photo of a messy {}.",
#     "a photo of a {} with colorful toppings.",
#     "a photo of a traditional {}.",
#     "a photo of a {} with a modern twist.",
#     "a black and white photo of a {}.",
#     "a painting of a {}.",
#     "a photo of a {} next to a cup of coffee.",
#     "a photo of a {} being shared by people.",
#     "a photo of a {} with a bite taken out.",
#     "a photo of a {} on a picnic blanket.",
#     "a photo of a {} on a clean white background."
# ]

# template = [
#     "an aerial view of a {}.",
#     "a medical image of a {}.",
#     "a {} on a plate.",
#     "a {} in a field.",
#     "a {} in a kitchen.",
#     "a {} in a hospital.",
#     "a satellite photo of a {}.",
#     "a {} in a forest.",
#     "a {} on a table.",
#     "a {} in the sky.",
#     "a {} on a farm.",
#     "a {} being cooked.",
#     "a close-up of a {}.",
#     "a drawing of a {}.",
#     "a {} in a jar.",
#     "a {} in a city.",
#     "a 3D model of a {}.",
#     "a diagram of a {}.",
#     "a blurry {}.",
#     "a microscope image of a {}.",
#     "a {} in water.",
#     "a {} in space.",
#     "a cartoon {}.",
#     "a {} in a lab.",
#     "a photo of a {}.",
#     "a sculpture of a {}.",
#     "a painting of a {}.",
#     "a {} in a forest.",
#     "a slice of a {}.",
#     "a black and white {}.",
#     "a vintage photo of a {}.",
#     "a {} in a machine.",
#     "a zoomed-in {}.",
#     "a {} in the desert.",
#     "a digital rendering of a {}.",
#     "a {} on a fence.",
#     "a {} next to food.",
#     "a pixelated {}.",
#     "a {} on a beach.",
#     "a {} in a factory.",
#     "a product photo of a {}.",
#     "a {} in a museum.",
#     "a realistic {}.",
#     "a messy {}.",
#     "a {} in a box.",
#     "a colorful {}.",
#     "a faded {}.",
#     "a textured {}.",
#     "a futuristic {}.",
#     "a glowing {}.",
#     "a rusty {}.",
#     "a broken {}.",
#     "a {} under a microscope.",
#     "a scratched {}.",
#     "a sleek {}.",
#     "a distorted {}.",
#     "a {} in a bag.",
#     "a plastic {}.",
#     "a wet {}.",
#     "a stained {}.",
#     "a frozen {}.",
#     "a shiny {}.",
#     "a {} in a fridge.",
#     "a blurry outline of a {}.",
#     "a scratched image of a {}.",
#     "a bright {}.",
#     "a reflective {}.",
#     "a {} with vibrant colors.",
#     "a misty view of a {}.",
#     "a {} in a crowded place.",
#     "a {} next to a computer.",
#     "a floating {}.",
#     "a moldy {}.",
#     "a {} on a hill.",
#     "a futuristic view of a {}.",
#     "a shadow of a {}.",
#     "a dirty {}.",
#     "a peaceful {}.",
#     "a surreal {}."
# ]

# template = [
#     'a bad photo of a {}, a type of a food.',
#     'a photo of many {}, a type of a food.',
#     'a sculpture of a {}, a type of a food.',
#     'a photo of the hard to see {}, a type of a food.',
#     'a low resolution photo of the {}, a type of a food.',
#     'a rendering of a {}, a type of a food.',
#     'graffiti of a {}, a type of a food.',
#     'a bad photo of the {}, a type of a food.',
#     'a cropped photo of the {}, a type of a food.',
#     'a tattoo of a {}, a type of a food.',
#     'the embroidered {}, a type of a food.',
#     'a photo of a hard to see {}, a type of a food.',
#     'a bright photo of a {}, a type of a food.',
#     'a photo of a clean {}, a type of a food.',
#     'a photo of a dirty {}, a type of a food.',
#     'a dark photo of the {}, a type of a food.',
#     'a drawing of a {}, a type of a food.',
#     'a photo of my {}, a type of a food.',
#     'the plastic {}, a type of a food.',
#     'a photo of the cool {}, a type of a food.',
#     'a close-up photo of a {}, a type of a food.',
#     'a black and white photo of the {}, a type of a food.',
#     'a painting of the {}, a type of a food.',
#     'a painting of a {}, a type of a food.',
#     'a pixelated photo of the {}, a type of a food.',
#     'a sculpture of the {}, a type of a food.',
#     'a bright photo of the {}, a type of a food.',
#     'a cropped photo of a {}, a type of a food.',
#     'a plastic {}, a type of a food.',
#     'a photo of the dirty {}, a type of a food.',
#     'a jpeg corrupted photo of a {}, a type of a food.',
#     'a blurry photo of the {}, a type of a food.',
#     'a photo of the {}, a type of a food.',
#     'a good photo of the {}, a type of a food.',
#     'a rendering of the {}, a type of a food.',
#     'a {} in a video game, a type of a food.',
#     'a photo of one {}, a type of a food.',
#     'a doodle of a {}, a type of a food.',
#     'a close-up photo of the {}, a type of a food.',
#     'a photo of a {}, a type of a food.',
#     'the origami {}, a type of a food.',
#     'the {} in a video game, a type of a food.',
#     'a sketch of a {}, a type of a food.',
#     'a doodle of the {}, a type of a food.',
#     'a origami {}, a type of a food.',
#     'a low resolution photo of a {}, a type of a food.',
#     'the toy {}, a type of a food.',
#     'a rendition of the {}, a type of a food.',
#     'a photo of the clean {}, a type of a food.',
#     'a photo of a large {}, a type of a food.',
#     'a rendition of a {}, a type of a food.',
#     'a photo of a nice {}, a type of a food.',
#     'a photo of a weird {}, a type of a food.',
#     'a blurry photo of a {}, a type of a food.',
#     'a cartoon {}, a type of a food.',
#     'art of a {}, a type of a food.',
#     'a sketch of the {}, a type of a food.',
#     'a embroidered {}, a type of a food.',
#     'a pixelated photo of a {}, a type of a food.',
#     'itap of the {}, a type of a food.',
#     'a jpeg corrupted photo of the {}, a type of a food.',
#     'a good photo of a {}, a type of a food.',
#     'a plushie {}, a type of a food.',
#     'a photo of the nice {}, a type of a food.',
#     'a photo of the small {}, a type of a food.',
#     'a photo of the weird {}, a type of a food.',
#     'the cartoon {}, a type of a food.',
#     'art of the {}, a type of a food.',
#     'a drawing of the {}, a type of a food.',
#     'a photo of the large {}, a type of a food.',
#     'a black and white photo of a {}, a type of a food.',
#     'the plushie {}, a type of a food.',
#     'a dark photo of a {}, a type of a food.',
#     'itap of a {}, a type of a food.',
#     'graffiti of the {}, a type of a food.',
#     'a toy {}, a type of a food.',
#     'itap of my {}, a type of a food.',
#     'a photo of a cool {}, a type of a food.',
#     'a photo of a small {}, a type of a food.',
#     'a tattoo of the {}, a type of a food.'
# ]



def infer_image_type(class_names, model, tokenizer):
    prompt = f"""Given the following class identifiers: {', '.join(map(str, class_names[:10]))}, 
    what single type of images do these most likely represent? Provide a single word for image category. Be specific and choose the most probable category.

    Examples of image types:
    - food dishes
    - flowering plants
    - medical scans
    - vehicles
    - microscopic organisms
    - celestial bodies
    - aerial photographs
    - domestic pets
    - satellite imagery
    - architectural structures
    - marine life
    - electronic devices
    - ancient artifacts
    - geological formations
    - sports equipment

    Based on the given class identifiers, the single most likely image type is:"""
    
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, temperature=0.7)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the image type from the response
    image_type = response.split("image type is:")[-1].strip().split(".")[0].lower()
    
    return image_type

# # Load LLaMA model and tokenizer
# model_name = "meta-llama/Llama-2-7b-chat-hf"  # Choose an appropriate LLaMA model
# # tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token=True)
# # model = LlamaForCausalLM.from_pretrained(model_name, use_auth_token=True)
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2LMHeadModel.from_pretrained('gpt2')

# class Food101(DatasetBase):
#     def __init__(self, root, dataset_dir, split_path):
#         self.dataset_dir = os.path.join(root, dataset_dir)
#         self.image_dir = os.path.join(self.dataset_dir, 'images')
#         self.split_path = os.path.join(self.dataset_dir, split_path)
        
#         test = OxfordPets.read_split(self.split_path, self.image_dir)
        
#         # Get class names from the test set
#         class_names = list(set([item[1] for item in test]))
        
#         # Infer image type using LLaMA
#         image_type = infer_image_type(class_names, model, tokenizer)
        
#         # Generate new templates
#         self.template = [t + f', a type of {image_type}.' for t in template]

#         super().__init__(test=test)

import json
class Food101(DatasetBase):

    dataset_dir = 'food-101'

    def __init__(self, root):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_Food101.json')
        
        self.template = template

        test = OxfordPets.read_split(self.split_path, self.image_dir)

        # class_ids = list(set([item[1] for item in test]))

        with open(self.split_path, 'r') as f:
            split_data = json.load(f)
        
        # Extract class IDs from the split data
        class_ids = set()
        for item in split_data['test']:
            class_id = item[2]  # Assuming the class ID is the second item in each entry
            class_ids.add(class_id)
        
        class_names = list(class_ids)


        # image_type = infer_image_type(class_names, model, tokenizer)

        # image_type = image_type.split("\n")[0][2:]
        
        # # Generate new templates
        # self.template = [t + f', a type of {image_type}.' for t in template]


        super().__init__(test=test)