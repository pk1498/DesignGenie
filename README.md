# Design Genie - Virtual Interior Design Tool

## Project Overview
The Virtual Interior Design Tool leverages AI to analyze room photos and recommend furniture, decor, and color schemes that match the room’s layout and style. It combines advanced deep learning techniques with user-friendly visualization methods to create a seamless virtual design experience.

## Features
- Detect existing furniture and decor items in user-uploaded room images.
- Generate design recommendations based on room layout and style.
- Visualize recommendations using 2D overlays and image inpainting.

## Installation
Follow these steps to set up the project on your local machine:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/VirtualInteriorDesignTool.git
   cd VirtualInteriorDesignTool

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
   
3. Download the necessary datasets:

   - [MIT Places365]()
   - [SUN Dataset]()
   - Furniture and decor images from sites like IKEA, Wayfair, and Pinterest.

4. Set up directories

    ```bash
    mkdir -p data/raw data/processed data/annotations data/furniture_decor
    mkdir -p models/object_detection models/inpainting models/depth_estimation
    mkdir -p results/visualizations results/logs results/metrics

## Usage

1. Prepare Data:

    - Preprocess datasets using scripts in src/data_preprocessing/.

2. Train Models:

    - Train and fine-tune object detection models with src/object_detection/.

3. Run Recommendations:

   - Generate design suggestions using src/recommendation/.
   
4. Visualize:

    - Visualize recommendations using src/visualization/.

## Contributing
We welcome contributions! Please fork the repository and create a pull request with your changes.

## License
This project is licensed under the MIT License.

## Acknowledgments
We are grateful for the open datasets and pre-trained models from:

- MIT Places365
- SUN Dataset
- YOLOv5
- Stable Diffusion