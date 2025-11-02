# ORION: An Occlusion-Robust Invariant Retrieval Framework

> **Project Status:** Phase 2 (Core Development)
>
> **Baseline results established (01 Nov 2025).**

[**Live Demo (Coming Soon)**]() | [**Model Weights (Coming Soon)**]() | [**Project Report (Coming Soon)**]()

---

## 1. The Problem

State-of-the-art vision-language models like **CLIP** perform exceptionally well on clean benchmark datasets but fail significantly in real-world scenarios where objects are partially occluded.
This project demonstrates a practical method to mitigate that performance degradation.

📉 Based on our initial benchmark, a **50% occlusion** causes the model's retrieval performance to **drop by over 76%**.

## 2. The Solution: ORION

**ORION (An Occlusion-Robust Invariant Open-Vocabulary Network)** is a fine-tuning strategy that uses a novel **completion-consistency loss** to teach the CLIP vision encoder to be robust to missing information.

The training objective teaches CLIP to map multiple views of an image (clean, occluded) to the same point in the embedding space, making it **occlusion-invariant**.

### 🧩 Architecture Overview

The diagram below shows ORION’s fine-tuning pipeline.
The model minimizes both:
-   **Consistency Loss:** forces occluded and clean embeddings to match
-   **Retrieval Loss:** ensures correct caption alignment

```mermaid
graph TD;

    subgraph "1. Data Pipeline"
        A[Clean Image] --> B(Add Occlusion);
        B --> C[Occluded Image];
        A --> D(Processor);
        C --> D(Processor);
    end

    subgraph "2. ORION Fine-Tuning"
        D --> E[CLIP Vision Encoder];
        E --> F[Occluded Embedding];
        A --> G[CLIP Vision Encoder];
        G --> H[Original Embedding];
        H -.-> I(Consistency Loss);
        F --> I(Consistency Loss);
    end

    subgraph "3. Standard CLIP Loss"
        F --> J(Retrieval Loss);
        K[Text Caption] --> L[CLIP Text Encoder];
        L --> M[Text Embedding];
        M --> J(Retrieval Loss);
    end

    subgraph "4. Combined Loss"
        I --> N(Total Loss);
        J --> N(Total Loss);
        N --> O(Backward Pass);
        O -- Updates --> E;
    end

## 3. Performance: Baseline vs. ORION
This is the core result of the project, demonstrating the effectiveness of the orion_loss. The baseline was established using the yerevann/coco-karpathy test split on 1,500 samples with a 50% occlusion patch.

Metric                              Baseline CLIP    ORION (Fine-Tuned)
Clean Image Accuracy (Top-1)        48.47 %          TBD %
Occluded Image Accuracy (Top-1)     11.73 %          TBD %

## 4. Visual Examples
Here is a visual example of the data the model is trained on. The goal is for both images to produce a similar embedding.

## 5. Tech Stack
This project leverages a modern, end-to-end MLOps stack:

Model: PyTorch, Transformers, Hugging Face

Data: Hugging Face datasets

Deployment: Gradio & Hugging Face Spaces

Version Control: Git & GitHub


## 6. How to Reproduce (Getting Started)
Clone the repository:
Bash
git clone [https://github.com/SupreethReddy25/Orion.git](https://github.com/SupreethReddy25/Orion.git)
cd Orion

Create a virtual environment and install dependencies:
Bash
# Create a virtual environment
python -m venv venv
# Activate it (Windows PowerShell)
.\venv\Scripts\Activate
# Install all required libraries
pip install -r requirements.txt

Run the baseline evaluation (to verify your setup):
Bash
# This will run on your GPU and download the dataset.
# It will take ~20-25 minutes as it downloads 1,500 images.
python scripts/run_baseline.py

Expected Output:

==============================
--- BASELINE RESULTS (Top-1 Retrieval) ---
Clean Image Accuracy:   48.47%
Occluded Image Accuracy: 11.73%
==============================
Results saved to baseline_results.txt


## 7. Project Roadmap
[ ] Core Development: Implement the OrionDataset and orion_loss function.

[ ] Fine-Tuning: Run the training loop to fine-tune ORION.

[ ] Publish Model: Train the final model and publish the orion-clip weights to the Hugging Face Hub.

[ ] Deploy Demo: Deploy the final Gradio app to a public Hugging Face Space.

## 8. References & Citations
CLIP: Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. arXiv:2103.00020.
Dataset: yerevann/coco-karpathy. The MS-COCO dataset with Karpathy test splits. https://cocodataset.org/