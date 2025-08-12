"""
ETTA (Efficient Test-Time Adaptation) for CLIP Models

This module implements a test-time adaptation strategy that improves CLIP model performance
by dynamically updating the model during inference using a cache-based learning approach.

Key Features:
- Cache-based knowledge accumulation during testing
- Entropy-based fusion of cached and original predictions
- Support for multiple CLIP backbones (ViT-B/16, RN50)
- Multi-dataset processing capability
"""

import random
import argparse
from tqdm import tqdm
import torch
import clip
from utils import *


def get_arguments():
    """
    Parse command line arguments for the test-time adaptation.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="ETTA: Efficient Test-Time Adaptation for CLIP")
    
    # Add the config file
    parser.add_argument('--config', dest='config', type=str, default="configs",
                       help="Config file for the test-time adaptation.")
    
    # Dataset configuration
    parser.add_argument('--datasets', dest='datasets', type=str, default="fgvc")
    
    parser.add_argument('--data-root', dest='data_root', type=str, default='D:\TDA\DATA',
                       help='Path to the datasets directory.')
    
    # Model configuration
    parser.add_argument('--backbone', dest='backbone', default="RN50", type=str,
                       choices=['RN50', 'ViT-B/16'], 
                       help='CLIP model backbone to use: RN50 or ViT-B/16.')
    
    # TTA approach
    parser.add_argument('--tta-approach', dest='tta_approach', default="ETTA", type=str,
                       choices=['ETTA', 'ETTA_PLUS'], 
                       help='TTA approach to use: ETTA or ETTA_PLUS.')
    
    args = parser.parse_args()
    return args


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def select_confident_samples(logits, top):
    """
    Select the most confident samples based on entropy.
    
    Args:
        logits (torch.Tensor): Model output logits
        top (float): Fraction of most confident samples to select
        
    Returns:
        tuple: (selected_logits, selected_indices)
    """
    # Calculate entropy for each sample (lower entropy = higher confidence)
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    
    # Select top-k most confident samples (lowest entropy)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx


def avg_entropy(outputs):
    """
    Calculate average entropy across multiple outputs.
    
    Args:
        outputs (torch.Tensor): Model outputs
        
    Returns:
        torch.Tensor: Average entropy
    """
    # Convert to logits
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) 
    
    # Calculate average logits across samples
    avg_logits = logits.logsumexp(dim=0) - torch.log(torch.tensor(logits.shape[0], dtype=logits.dtype))
    
    # Clamp to avoid numerical issues
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    
    # Calculate entropy from average logits
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


# =============================================================================
# CACHE MANAGEMENT FUNCTIONS
# =============================================================================

def update_cache(cache, pred, image_features, contextless_weight):
    """
    Update the cache with new image features for a predicted class.
    
    Args:
        cache (dict): Cache storing learned features for each class
        pred (int): Predicted class index
        image_features (torch.Tensor): Features extracted from the current image
        contextless_weight (torch.Tensor): Weight vector for the predicted class
    """
    # Get current cache state for the predicted class
    current_sum = cache[pred]["sum"]
    current_weight = cache[pred]["weight"]
    
    # Calculate exponential term for weight update
    exp_term = torch.exp(contextless_weight @ image_features.T)
    
    # Update sum and weight using exponential moving average
    new_sum = current_sum + exp_term
    new_weight = (current_weight * current_sum + exp_term * image_features) / new_sum
    
    # Store updated values back to cache
    cache[pred]["sum"] = new_sum
    cache[pred]["weight"] = new_weight


def compute_cache_logits(image_features, cache):
    """
    Compute logits using cached knowledge.
    
    Args:
        image_features (torch.Tensor): Features of the current image
        cache (dict): Cache containing learned features for each class
        
    Returns:
        torch.Tensor: Logits computed from cached knowledge
    """
    # Collect all cached weights
    cached_weights = []
    for key in cache.keys():
        cached_weight = cache[key]["weight"]
        cached_weights.append(cached_weight)
    
    # Concatenate all weights and normalize
    cached_weights = torch.cat(cached_weights, dim=0)
    cached_weights = cached_weights / (cached_weights.norm(dim=1, keepdim=True) + 1e-6)
    
    # Compute logits using cached knowledge
    cache_logits = image_features @ cached_weights.T
    return cache_logits


# =============================================================================
# MAIN TEST-TIME ADAPTATION FUNCTION
# =============================================================================

def run_test_tda(loader, clip_model, clip_weights, classnames, dataset_name, config):
    """
    Run Test-Time Adaptation (TTA) on the given dataset.
    
    This function implements the core TTA algorithm that:
    1. Processes test images sequentially
    2. Updates cache with learned knowledge
    3. Combines cached and original predictions
    4. Tracks performance improvements
    
    Args:
        loader: DataLoader for test images
        clip_model: Pre-trained CLIP model
        clip_weights: CLIP classifier weights
        classnames: List of class names
        dataset_name: Name of the dataset being processed
        args: Command line arguments
        
    Returns:
        float: Final accuracy achieved by the adapted model
    """
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    # Initialize tracking variables
    clip_accuracies = []      # Original CLIP performance
    cache_accuracies = []     # Cache-enhanced performance
    cache = {}                # Knowledge cache for each class
    
    # Initialize cache for each class
    if isinstance(clip_weights, torch.Tensor):
        # Handle tensor format
        for k in range(clip_weights.size(0)):
            cache[k] = {
                "sum": 0, 
                "weight": torch.zeros_like(clip_weights[0][0]).unsqueeze(0)
            }
    else:
        # Handle list format
        for k in range(len(clip_weights)):
            cache[k] = {
                "sum": 0, 
                "weight": torch.zeros_like(clip_weights[0][0][0]).unsqueeze(0)
            }
    
    # =============================================================================
    # MAIN TESTING LOOP
    # =============================================================================
    for i, (images, target) in enumerate(tqdm(loader, desc='Processing test images: ')):
        
        # Extract features and get initial predictions (no gradient computation)
        with torch.no_grad():
            image_features, clip_logits, pred, clip_top_weights = get_clip_logits(
                images, clip_model, clip_weights, alpha=float(config['alpha'])
            )
        
        # =============================================================================
        # COMPUTE MEAN CLIP WEIGHTS
        # =============================================================================
        if isinstance(clip_weights, torch.Tensor):
            # Average across ensemble dimension
            clip_weights_mean = clip_weights.mean(dim=1)
            clip_weights_mean = clip_weights_mean / clip_weights_mean.norm(dim=1, keepdim=True)
        else:
            # Average across list of weight tensors
            clip_weights_mean = []
            for k in range(len(clip_weights)):
                clip_weights_mean.append(clip_weights[k].mean(dim=1))
            clip_weights_mean = torch.cat(clip_weights_mean, dim=0)
            clip_weights_mean = clip_weights_mean / clip_weights_mean.norm(dim=1, keepdim=True)
        
        # Compute logits using mean weights
        clip_logits_mean = image_features @ clip_weights_mean.T
        
        # Move target to GPU
        target = target.cuda()
        
        # =============================================================================
        # UPDATE CACHE AND COMPUTE ENHANCED PREDICTIONS
        # =============================================================================
        # Update cache with new knowledge from current image
        update_cache(cache, pred, image_features, clip_top_weights[pred].unsqueeze(0))
        
        # Compute logits using cached knowledge
        cache_logits = compute_cache_logits(image_features, cache)
        
        # =============================================================================
        # ENTROPY-BASED FUSION OF PREDICTIONS
        # =============================================================================
        # Calculate entropy for cache-based predictions
        cache_prob = (100 * cache_logits).softmax(dim=1) + 1e-6
        cache_entropy = -(cache_prob * cache_prob.log()).sum(dim=1)
        
        # Calculate entropy for original CLIP predictions
        clip_prob = (100 * clip_logits).softmax(dim=1) + 1e-6
        clip_entropy = -(clip_prob * clip_prob.log()).sum(dim=1)
        
        # Fuse predictions based on entropy (lower entropy = higher confidence)
        # More confident predictions get higher weight
        merged_logit = (clip_entropy / (clip_entropy + cache_entropy)) * cache_logits + \
                      (cache_entropy / (clip_entropy + cache_entropy)) * clip_logits + \
                      0 * clip_logits_mean  # Mean weights not used in current implementation
        
        # =============================================================================
        # EVALUATE PERFORMANCE
        # =============================================================================
        # Calculate accuracy for both approaches
        acc_cache = cls_acc(merged_logit, target)    # Cache-enhanced accuracy
        acc_clip = cls_acc(clip_logits, target)      # Original CLIP accuracy
        
        # Store accuracies for tracking
        clip_accuracies.append(acc_clip)
        cache_accuracies.append(acc_cache)
        
        # Report progress every 200 images
        if i % 200 == 199:
            print(f"Progress - Cache accuracy: {sum(cache_accuracies)/len(cache_accuracies):.4f}")
            print(f"Progress - CLIP accuracy: {sum(clip_accuracies)/len(clip_accuracies):.4f}")
    
    # =============================================================================
    # FINAL RESULTS
    # =============================================================================
    final_cache_acc = sum(cache_accuracies) / len(cache_accuracies)
    final_clip_acc = sum(clip_accuracies) / len(clip_accuracies)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS FOR {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"Original CLIP Accuracy: {final_clip_acc:.4f}")
    print(f"Cache-Enhanced Accuracy: {final_cache_acc:.4f}")
    print(f"Improvement: {final_cache_acc - final_clip_acc:.4f}")
    print(f"{'='*60}\n")
    
    return final_cache_acc


# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def main():
    """
    Main function that orchestrates the entire ETTA process.
    
    This function:
    1. Parses command line arguments
    2. Loads and initializes the CLIP model
    3. Processes each specified dataset
    4. Runs test-time adaptation
    5. Reports results
    """
    # Parse command line arguments
    args = get_arguments()

    # read the config file
    configs = get_config_file(args.config, args.datasets)
    if args.backbone == "ViT-B/16":
        config = configs["VIT"][args.tta_approach]
    else:
        config = configs["RESNET"][args.tta_approach]


    # Set random seeds for reproducibility
    random.seed(1)
    torch.manual_seed(1)
    
    print(f"Initializing ETTA with backbone: {args.backbone}")
    print(f"Alpha parameter: {float(config['alpha'])}")
    print(f"CSP parameter: {config['csp']}")
    print(f"Augmentation parameter: {config['augmen']}")
    print(f"Data root: {args.data_root}")
    
    # Load CLIP model
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()
    print("CLIP model loaded successfully!")
    
    # Process each specified dataset
    datasets = args.datasets.split('/')
    print(f"Processing {len(datasets)} dataset(s): {datasets}")
    
    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*50}")
        
        # Build data loader for current dataset
        test_loader, classnames, template = build_test_data_loader(
            dataset_name, args.data_root, preprocess, config['augmen']
        )
        
        # Determine dataset type for CLIP weights
        if dataset_name in ["I", "A", "V", "R", "S"]:
            dataset_name2 = "imagenet"  # ImageNet variants
        else:
            dataset_name2 = dataset_name  # Custom datasets
        
        # Load CLIP classifier weights with CuPL prompts
        print(f"Loading CLIP weights with CuPL prompts for {dataset_name2}...")
        
        # determine whether to use CSP or GP
        if config['csp']:
            json_file_path = f"./datasets/class_descriptions/CuPL_prompts_{dataset_name2}.json"
        else:
            json_file_path = None

        clip_weights = clip_classifier(
            classnames, template, clip_model, 
            json_file_path=json_file_path
        )
        
        # Run test-time adaptation
        print("Starting Test-Time Adaptation...")
        final_accuracy = run_test_tda(
            test_loader, clip_model, clip_weights, 
            classnames, dataset_name, config
        )
        
        print(f"Dataset {dataset_name} completed with final accuracy: {final_accuracy:.4f}")
    
    print("\nAll datasets processed successfully!")


if __name__ == "__main__":
    main()