# Text-to-Image Diffusion from Scratch

This project is an implementation of a Text-to-Image Diffusion Model, leveraging advanced machine learning techniques and models such as Variational Autoencoders (VAE), U-Net, and Contrastive Language-Image Pre-Training (CLIP). The goal of this project is to generate high-quality images from textual descriptions and optionally from input images with added noise.

## Introduction

The Text-to-Image Diffusion Model is designed to create images based on textual descriptions using a combination of cutting-edge neural network architectures. The model can also refine images by removing noise, producing clear and visually appealing outputs. This project was implemented from scratch, without relying on any external libraries, showcasing a deep understanding of the underlying algorithms and techniques. The project demonstrates the integration of multiple complex models to achieve a coherent and functional text-to-image pipeline.

## Sample Outputs

<table>
  <tr>
    <td><img src="https://i.ibb.co/hXthGk0/img1.jpg" alt="Image 1" style="width: 100%;"/></td>
    <td><img src="https://i.ibb.co/VjhfnSC/img2.jpg" alt="Image 2" style="width: 100%;"/></td>
  </tr>
  <tr>
    <td><img src="https://i.ibb.co/grz5JMh/img3.jpg" alt="Image 3" style="width: 100%;"/></td>
    <td><img src="https://i.ibb.co/4gFFvyx/img4.jpg" alt="Image 4" style="width: 100%;"/></td>
  </tr>
</table>

## Architecture

<img src="https://i.ibb.co/MhmTs43/architecture.jpg" alt="Image 1" style="width: 100%;"/>

The architecture of this project consists of several key components that work together to transform text into images:

1. CLIP Model: Converts text descriptions into embeddings.
2. VAE Encoder: Encodes images into a latent space representation.
3. U-Net: Predicts the noise present in the image and denoises it.
4. VAE Decoder: Reconstructs images from the latent space.
5. DDPM Sampler: Iteratively refines the image by removing predicted noise.

The interplay between these components allows the model to understand textual descriptions and generate corresponding images.

## Components

### 1. Attention Mechanisms (`attention.py`)

- **Self-Attention**: Calculates the relevance of each token with respect to other tokens within the same block, used within both text and image embeddings.
- **Cross-Attention**: Calculates the relevance between tokens of text and pixels of images, crucial for aligning text descriptions with image features.

### 2. CLIP (`clip.py`)

- **Purpose**: Generates text embeddings from descriptions.
- **Contribution**: Translates textual information into a format that the model can use to condition the image generation process.

### 3. Denoising Diffusion Probabilistic Models (DDPM) (`ddpm.py`)

- **Purpose**: Refines images by removing noise at each time step.
- **Contribution**: Ensures that the generated image is progressively improved and denoised to match the input description.

### 4. Variational Autoencoder (VAE) Encoder and Decoder

- **Encoder (`encoder.py`)**: Reduces the dimensionality of the image, capturing essential features in a latent space.
- **Decoder (`decoder.py`)**: Reconstructs the image from the latent representation.
- **Contribution**: VAE helps in representing images in a compressed form, enabling efficient manipulation and reconstruction.

### 5. U-Net (`diffusion.py`)

- **Purpose**: Predicts the noise in images and denoises them.
- **Contribution**: Acts as the core network that processes the image at different scales, ensuring detailed and high-quality outputs.

### 6. Model Loading and Conversion

- **Model Loader (`model_loader.py`)**: Loads pretrained models and their weights.
- **Model Converter (`model_converter.py`)**: Converts weights from different formats to be compatible with the current implementation.
- **Contribution**: Facilitates the use of pretrained models, making the system more robust and efficient.

### 7. Pipeline (`pipeline.py`)

- **Purpose**: Integrates all components to form a complete text-to-image generation system.
- **Contribution**: Orchestrates the process from text input to final image output, ensuring all components work seamlessly together.

## Conclusion

This project demonstrates a sophisticated approach to generating images from textual descriptions by integrating multiple advanced neural network components, all implemented from scratch. By building this system without any external libraries, it showcases a comprehensive understanding of the algorithms and techniques involved. The use of VAE, U-Net, and CLIP models, along with the DDPM sampling technique, results in a powerful and flexible text-to-image generation system. The modular design of the components allows for easy adaptation and extension, paving the way for further research and development in the field of generative models.
