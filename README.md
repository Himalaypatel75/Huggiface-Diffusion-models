# Diffusion Model - Conditional Image Generation

This repository contains a diffusion model for conditional image generation using the Hugging Face Diffusers library. The example is based on the official Hugging Face documentation and demonstrates how to generate images conditioned on text prompts.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Model Details](#model-details)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Diffusion models are a class of generative models that can produce high-quality images by iteratively refining a noise vector according to a learned distribution. This repository implements a diffusion model that can generate images based on text prompts, allowing for controllable image generation.

## Features

- **Conditional Image Generation**: Generate images conditioned on text prompts.
- **Pre-trained Models**: Use state-of-the-art pre-trained models for image generation.
- **Custom Prompts**: Input custom text prompts to generate unique images.
- **Easy-to-Use API**: Simple API for integration into your own projects.

## Installation

To install the necessary dependencies, clone the repository and install the required Python packages:

```bash
git clone https://github.com/yourusername/diffusion-model.git
cd diffusion-model
pip install -r requirements.txt
```

Ensure that you have Python 3.8 or higher installed.

## Usage

To generate an image conditioned on a text prompt, use the following script:

```python
import torch
from diffusers import StableDiffusionPipeline

# Load the pre-trained model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Generate an image based on a prompt
prompt = "A futuristic cityscape with flying cars"
image = pipe(prompt).images[0]

# Save the image
image.save("generated_image.png")
```

### Running on CPU

If you do not have a GPU, you can run the model on a CPU by removing the `.to("cuda")` line:

```python
pipe = pipe.to("cpu")
```

### Customization

You can customize the text prompt and other parameters in the script to generate different images.

## Examples

Here are some example prompts and the corresponding generated images:

1. **Prompt**: "A serene beach at sunset"
   ![Serene Beach](example_images/beach.png)

2. **Prompt**: "A futuristic cityscape with flying cars"
   ![Futuristic Cityscape](example_images/cityscape.png)

3. **Prompt**: "A dragon flying over a mountain"
   ![Dragon](example_images/dragon.png)

## Model Details

This repository uses the [Stable Diffusion](https://huggingface.co/docs/diffusers/using-diffusers/conditional_image_generation) model provided by Hugging Face. Stable Diffusion is a powerful generative model capable of creating high-quality images from text descriptions.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or new features to propose.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
