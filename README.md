# ETTA: Efficient Test-Time Adaptation for Vision-Language Models

Official implementation of **ETTA: Efficient Test-Time Adaptation for Vision-Language Models via Dynamic Embedding Updates**.

## About

ETTA is a novel test-time adaptation approach that improves CLIP model performance through two key innovations:

1. **Adaptive Ensembling Module**: Intelligently filters and selects the most relevant prompts for each test sample
2. **Recursive Update Mechanism**: Dynamically updates class embeddings based on incoming test-time samples

The method adapts to test data without requiring retraining, making it efficient and practical for real-world deployment.



![Block Diagram](assets/diagram_gif.gif)


## Key Features

- **Adaptive Prompt Filtering**: Dynamically selects top-α% most relevant prompts per class based on image-class similarity
- **Recursive Cache Updates**: Continuously updates class representations using exponential moving average of test samples
- **Cache-based Knowledge Accumulation**: Maintains and refines knowledge during testing
- **Entropy-based Fusion**: Intelligently combines cached and original predictions
- **Support for Multiple CLIP Backbones**: Compatible with ViT-B/16 and RN50
- **Multi-dataset Processing**: Handles various datasets including ImageNet variants
- **Class-Specific Prompts (CSP) vs General Prompts (GP)**: Flexible prompt selection strategies


## Requirements

- Python 3.7
- PyTorch 1.12.1
- CUDA 11.3
- CLIP model
- Additional packages listed in `requirements.txt`

## Installation

Follow these steps to set up a conda environment and ensure all necessary packages are installed:

```bash
git clone https://github.com/hamidreza-dastmalchi/ETTA.git 
cd ETTA

conda create -n tda python=3.7
conda activate tda

# The results are produced with PyTorch 1.12.1 and CUDA 11.3
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

pip install -r requirements.txt
```

## Dataset Setup

To set up all required datasets, kindly refer to the guidance in `DATASETS.md`, which incorporates steps for two benchmarks.

## Configuration

The configuration for ETTA is in `configs/dataset.yaml` and can be tailored within the provided file to meet the needs of various datasets.

### Prompt Strategies

ETTA supports two prompt strategies that can be configured:

#### Class-Specific Prompts (CSP) - ETTA+
- Uses detailed, domain-specific descriptions for each class
- Loads prompts from JSON files (e.g., `CuPL_prompts_imagenet.json`)
- Combines detailed descriptions with template prompts
- Generally provides better performance through richer class representations
- Enable with: `--use-csp True`

#### General Prompts (GP) - ETTA
- Uses only template-based prompts (e.g., "a photo of a {classname}")
- Simpler approach with fewer computational requirements
- Good baseline performance
- Enable with: `--use-csp False`

### Hyperparameters

The main hyperparameter in ETTA is **alpha**, which controls the adaptive ensembling:

- **alpha = 0.2**: Uses top 20% of most relevant prompts per class
- **alpha = 0.1**: Uses top 10% of most relevant prompts (more focused)
- **alpha = 0.5**: Uses top 50% of prompts (more robust but potentially noisy)

The alpha parameter determines the trade-off between prompt focus and robustness in the adaptive prompt filtering mechanism.

## Running ETTA

### Basic Usage

```bash
python ETTA.py --datasets caltech101 --backbone ViT-B/16 --alpha 0.2
```

### Command Line Arguments

- `--datasets`: Datasets to process (e.g., "caltech101", "I/A/V/R/S" for ImageNet variants)
- `--data-root`: Path to datasets directory (default: "D:\TDA\DATA")
- `--backbone`: CLIP model backbone ("ViT-B/16" or "RN50")
- `--alpha`: Alpha value for adaptive prompt filtering (default: 0.2)
- `--use-csp`: Use class-specific prompts instead of general prompts (default: True)

### Examples

```bash
# Run on Caltech101 with ViT-B/16 backbone
python ETTA.py --datasets caltech101 --backbone ViT-B/16 --alpha 0.2

# Run on ImageNet variants with RN50 backbone
python ETTA.py --datasets I/A/V/R/S --backbone RN50 --alpha 0.1

# Run on multiple datasets with custom data path
python ETTA.py --datasets caltech101/dtd --data-root /path/to/data --alpha 0.3
```

### Supported Datasets

- **ImageNet Variants**: I (ImageNet), A (ImageNet-A), V (ImageNet-V), R (ImageNet-R), S (ImageNet-S)
- **Standard Datasets**: caltech101, dtd, eurosat, fgvc, food101, oxford_flowers, oxford_pets, stanford_cars, sun397, ucf101

## How It Works

1. **Initialization**: Loads CLIP model and creates empty knowledge caches for each class
2. **Adaptive Prompt Filtering**: For each test image, selects top-α% most relevant prompts per class
3. **Feature Extraction**: Extracts image features using CLIP encoder
4. **Recursive Cache Updates**: Updates class embeddings using exponential moving average of test samples
5. **Entropy-based Fusion**: Combines cached and original predictions using confidence weighting
6. **Continuous Adaptation**: The model becomes progressively better at recognizing patterns in the test data

## Performance

ETTA typically shows significant improvements over baseline CLIP models, especially in domain adaptation scenarios. The combination of adaptive ensembling and recursive updates often outperforms original CLIP predictions by:

- Leveraging learned knowledge from the test data
- Adaptively selecting the most relevant prompts for each sample
- Continuously refining class representations during testing

## Citation

If you find this work useful, please cite:

```bibtex
@article{dastmalchi2024etta,
  title={ETTA: Efficient Test-Time Adaptation for Vision-Language Models via Dynamic Embedding Updates},
  author={Dastmalchi, Hamidreza and others},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.

---

**Note**: This implementation is based on the official ETTA paper and provides a complete, production-ready codebase for test-time adaptation of vision-language models with adaptive ensembling and recursive update capabilities. 
