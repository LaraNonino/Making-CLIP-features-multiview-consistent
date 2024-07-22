# Making CLIP Features Multiview Consistent

This repository contains the code for the semestral project "Making CLIP Features Multiview Consistent" by Lara Nonino, supervised by Barath Daniel and Engelmann Francis at ETH Zurich.

## Project Overview

Enhancing the multiview consistency of CLIP (Contrastive Language-Image Pretraining) using SigLIP to improve applications in object recognition, scene understanding, and augmented reality.

## Repository Structure

- `dataset_scripts`: Scripts to create the custom dataset.
- `study_scripts`: Scripts for initial analysis of embeddings.
- `fine-tune_scripts`: Scripts to fine-tune SigLIP.

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/LaraNonino/Making-CLIP-features-multiview-consistent.git
   cd Making-CLIP-features-multiview-consistent
   ```

2. Create a virtual environment and install dependencies:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

## Usage

### Dataset Creation

1. Navigate to the dataset_scripts directory:
   cd dataset_scripts

2. According to the dataset you want to create, run the corresponding jupyter notebbok

### Initial Analysis

1. Navigate to the study_scripts directory:
   cd ../study_scripts

2. Run the jupyter notebooks

### Fine-tuning SigLIP

1. Navigate to the fine-tune_scripts directory:
   cd ../fine-tune_scripts

2. Run the .sh script

## Results and Analysis

Results of the fine-tuning process, including the improved multiview consistency of embeddings, can be found in the results directory. Detailed analyses and visualizations demonstrate the enhancements achieved.

## Future Work

Future work involves:
- Unfreezing internal layers for further fine-tuning to improve alignment and consistency between visual and textual representations.
- Enhancing text embeddings by attaching a projection head to the pretrained text encoder and fine-tuning it with an additional dataset.
- Using a combined loss approach to maintain the relationship between image and text embeddings.

## Contact

For any questions or further information, please contact Lara Nonino at [lnonino@student.ethz.ch].
    