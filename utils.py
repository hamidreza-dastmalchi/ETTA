import os
import yaml
import torch
import math
import numpy as np
import clip
from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader, AugMixAugmenter
import torchvision.transforms as transforms
from PIL import Image
import json
from sklearn.cluster import KMeans

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def get_entropy(loss, clip_weights):
    max_entropy = math.log2(clip_weights.size(1))
    return float(loss / max_entropy)


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model, json_file_path=None):
    """
    Create CLIP classifier weights by encoding text prompts for each class.
    
    This function converts class names into feature vectors that CLIP can use for classification.
    It supports two modes: class-specific prompts (CuPL) and general template prompts.
    
    Args:
        classnames (list): List of class names for the dataset
        template (list): List of prompt templates (e.g., ["a photo of a {}"])
        clip_model: Pre-trained CLIP model for text encoding
        json_file_path (str, optional): Path to JSON file containing class-specific descriptions
        
    Returns:
        torch.Tensor: CLIP classifier weights with shape (num_classes, num_prompts, feature_dim)
        
    Note:
        - When json_file_path is provided, combines detailed descriptions with template prompts
        - When json_file_path is None, uses only template prompts
        - All embeddings are L2-normalized for fair comparison
    """
    with torch.no_grad():  # No gradient computation needed for inference
        # Initialize storage for class embeddings
        clip_weights = []
        clip_weights_template = []
        description_lengths = []
        
        # =============================================================================
        # MODE 1: WITH CLASS-SPECIFIC PROMPTS (CuPL)
        # =============================================================================
        if json_file_path is not None:
            print(f"Loading class-specific prompts from: {json_file_path}")
            
            # Load detailed class descriptions from JSON file
            with open(json_file_path, 'r') as f:
                class_descriptions = json.load(f)
            
            # First pass: collect description lengths to find minimum
                for classname in classnames:
                # Replace underscores with spaces for better text processing
                classname_clean = classname.replace('_', ' ')
                
                # Get detailed description for this class
                class_description = class_descriptions[classname_clean]
                description_lengths.append(len(class_description))
            
            # Find minimum description length for consistent processing
            min_length = min(description_lengths)
            print(f"Using minimum description length: {min_length}")
            
            # Second pass: encode descriptions and create embeddings
            with open(json_file_path, 'r') as f:
                class_descriptions = json.load(f)
                
                for classname in classnames:
                    # Clean classname for text processing
                    classname_clean = classname.replace('_', ' ')
                    
                    # Get detailed description for this class
                    class_description = class_descriptions[classname_clean]
                    
                    # =============================================================================
                    # ENCODE DETAILED CLASS DESCRIPTIONS
                    # =============================================================================
                    # Tokenize and encode the detailed description
                    texts = clip.tokenize(class_description).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                    
                    # Normalize embeddings for cosine similarity
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    
                    # =============================================================================
                    # ENCODE TEMPLATE PROMPTS
                    # =============================================================================
                    # Create template prompts (e.g., "a photo of a {classname}")
                    template_texts = [t.format(classname_clean) for t in template]
                    template_texts = clip.tokenize(template_texts).cuda()
                    
                    # Encode template prompts
                    class_embeddings_template = clip_model.encode_text(template_texts)
                    
                    # Normalize template embeddings
                    class_embeddings_template /= class_embeddings_template.norm(dim=-1, keepdim=True)
                    
                    # =============================================================================
                    # COMBINE DETAILED AND TEMPLATE EMBEDDINGS
                    # =============================================================================
                    # Take first min_length detailed embeddings and combine with template embeddings
                    # This ensures consistent dimensionality across all classes
                    combined_embeddings = torch.cat(
                        (class_embeddings[:min_length], class_embeddings_template), 
                        dim=0
                    )
                    
                    # Add batch dimension and store
                    clip_weights.append(combined_embeddings.unsqueeze(0))
                    
        # =============================================================================
        # MODE 2: TEMPLATE-ONLY PROMPTS (FALLBACK)
        # =============================================================================
        else:
            print("Using template-only prompts (no class-specific descriptions)")
            
            for classname in classnames:
                # Clean classname for text processing
                classname_clean = classname.replace('_', ' ')
                
                # =============================================================================
                # CREATE AND ENCODE TEMPLATE PROMPTS
                # =============================================================================
                # Format template prompts with current classname
                template_texts = [t.format(classname_clean) for t in template]
                
                # Tokenize and encode the template prompts
                template_texts = clip.tokenize(template_texts).cuda()
                class_embeddings = clip_model.encode_text(template_texts)
                
                # Normalize embeddings for cosine similarity
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                
                # Add batch dimension and store
                clip_weights.append(class_embeddings.unsqueeze(0))

        # =============================================================================
        # FINAL PROCESSING AND OUTPUT
        # =============================================================================
        # Concatenate all class embeddings into a single tensor
        # Shape: (num_classes, num_prompts, feature_dim)
        clip_weights = torch.cat(clip_weights, dim=0).cuda()

        print(f"Final CLIP weights shape: {clip_weights.shape}")
        print(f"Number of classes: {clip_weights.shape[0]}")
        print(f"Prompts per class: {clip_weights.shape[1]}")
        print(f"Feature dimension: {clip_weights.shape[2]}")
        
        return clip_weights



def get_clip_logits(images, clip_model, clip_weights, alpha):
    """
    Compute CLIP classification logits with adaptive prompt filtering.
    
    This function computes similarity scores between image features and class weights,
    using an adaptive filtering mechanism that selects only the most relevant prompts
    for each class based on the alpha parameter.
    
    Args:
        images: Input images (single tensor or list of augmented images)
        clip_model: Pre-trained CLIP model for feature extraction
        clip_weights: Class embeddings from clip_classifier (tensor or list)
        alpha (float): Fraction of top prompts to use (e.g., 0.2 = top 20%)
        
    Returns:
        tuple: (image_features, clip_logits, pred, top_clip_weights)
            - image_features: Final image features (single or averaged)
            - clip_logits: Classification logits
            - pred: Predicted class index
            - top_clip_weights: Filtered class weights used for prediction
            
    Note:
        - Supports both single images and augmented image sets
        - Automatically filters prompts based on image-class similarity
        - Uses confidence-based selection when multiple images are available
    """
    with torch.no_grad():  # No gradient computation needed for inference
        # =============================================================================
        # INPUT PROCESSING AND IMAGE FEATURE EXTRACTION
        # =============================================================================
        # Handle both single images and lists of augmented images
        if isinstance(images, list):
            images = torch.cat(images, dim=0).cuda()
        else:
            images = images.cuda()

        # Extract features from all images using CLIP
        images_all_features = clip_model.encode_image(images)
        
        # Normalize features for cosine similarity computation
        images_all_features /= images_all_features.norm(dim=-1, keepdim=True)
        
        # Use first image for main processing (will be updated if multiple images available)
        image_features = images_all_features[0].unsqueeze(0)
        
        # =============================================================================
        # COMPUTE INITIAL LOGITS AND FILTER PROMPTS
        # =============================================================================
        # clip_weights is always a tensor, so we can process directly
        # Compute initial logits between image features and all class weights
        clip_logits = torch.matmul(clip_weights, image_features.T).squeeze(-1)
        
        # Sort logits in descending order (highest similarity first)
        # This helps identify the most relevant prompts for each class
        sorted_logits, sorted_indices = torch.sort(clip_logits, dim=-1, descending=True)
        
        # Select top alpha% of prompts (e.g., alpha=0.2 means top 20%)
        num_top_prompts = int(sorted_logits.size(-1) * alpha)
        top_indices = sorted_indices[:, :num_top_prompts]
        
        # Extract top weights for each class based on similarity scores
            top_clip_weights = []
            for i in range(clip_weights.size(0)):
            # Get the top prompts for class i
                top_clip_weight = clip_weights[i, top_indices[i,:], :].unsqueeze(0)
                top_clip_weights.append(top_clip_weight)

        # Combine filtered weights from all classes
        top_clip_weights = torch.cat(top_clip_weights, dim=0)

        # Average the filtered weights and normalize
            top_clip_weights = top_clip_weights.mean(dim=1)
        top_clip_weights = top_clip_weights / top_clip_weights.norm(dim=1, keepdim=True)
        
        # Compute final logits using filtered weights
        clip_logits = image_features @ top_clip_weights.T

        # =============================================================================
        # MULTI-IMAGE PROCESSING (ENSEMBLE PREDICTION)
        # =============================================================================
        if images_all_features.size(0) > 1:
            # Multiple augmented images available - use ensemble approach            
            # Compute logits for all augmented images
            clip_logits = images_all_features @ top_clip_weights.T
            
            # Scale logits for better numerical stability
            clip_logits_scaled = 100 * clip_logits
            
            # Compute entropy for each prediction (lower entropy = higher confidence)
            batch_entropy = softmax_entropy(clip_logits_scaled)
            
            # Select most confident predictions (lowest entropy)
            # Use top 10% most confident predictions
            confidence_threshold = 0.1
            selected_idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * confidence_threshold)]
            output = clip_logits[selected_idx].clone()
            
            # Average selected features and predictions for final output
            image_features = images_all_features[selected_idx].mean(0).unsqueeze(0)
            clip_logits = output.mean(0).unsqueeze(0)

            # Compute additional metrics for analysis
            loss = avg_entropy(100 * output)
            prob_map = output.softmax(1).mean(0).unsqueeze(0)
            pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
            
        else:
            # Single image - standard processing            
            # Scale logits for consistency
            clip_logits_scaled = 100 * clip_logits
            
            # Compute entropy and probability map
            loss = softmax_entropy(clip_logits_scaled)
            prob_map = clip_logits_scaled.softmax(1)
            pred = int(clip_logits_scaled.topk(1, 1, True, True)[1].t()[0])
        
        return image_features, clip_logits, pred, top_clip_weights


def get_ood_preprocess():
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
    base_transform = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224)])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    aug_preprocess = AugMixAugmenter(base_transform, preprocess, n_views=63, augmix=True)

    return aug_preprocess


def get_config_file(config_path, dataset_name):
    if dataset_name == "I":
        config_name = "imagenet.yaml"
    elif dataset_name in ["A", "V", "R", "S"]:
        config_name = f"imagenet_{dataset_name.lower()}.yaml"
    else:
        config_name = f"{dataset_name}.yaml"
    
    config_file = os.path.join(config_path, config_name)
    
    with open(config_file, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.SafeLoader)

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The configuration file {config_file} was not found.")

    return cfg


def build_test_data_loader(dataset_name, root_path, preprocess, aug_preprocess):
    if dataset_name == 'M':
        dataset = ImageNet(root_path, preprocess)
        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=1, num_workers=8, shuffle=True)
    
    elif dataset_name in ['A','V','R','S', 'I']:
        if aug_preprocess:
            preprocess = get_ood_preprocess()
        dataset = build_dataset(f"imagenet-{dataset_name.lower()}", root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)

    elif dataset_name in ['caltech101','dtd','eurosat','fgvc','food101','oxford_flowers','oxford_pets','stanford_cars','sun397','ucf101']:
        if aug_preprocess:
            preprocess = get_ood_preprocess()
        dataset = build_dataset(dataset_name, root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)
    
    else:
        raise "Dataset is not from the chosen list"
    
    return test_loader, dataset.classnames, dataset.template



