# GradCAM
GradCAM functions to produce activation map from a given layer of the feature extractor.

NOTE: only compatible with a marugoto MIL model (https://github.com/KatherLab/marugoto.git)


## Installation & Usage
1. Create Python virtual environment (```python -m venv <env_name>```)
2. Activate the environment (```. ./<env_name>```)
3. Install the repo using:
   ```
   python -m pip install git+https://github.com/brooksja/GradCAM.git
   ```
5. In python, add the following to your imports:
   ```Python
   from GradCAM.GradCAM import GCAM
   ```
6. Use:
   ```Python
   GCAM(
       img_path,
       MILmodel_path,
       extractor,
       layer,
       transform,
       outpath
     )
   ```

Inputs:
  - img_path - path to the image to be analysed.
  - MILmodel_path - path to the marugoto checkpoint .pkl object.
  - extractor - feature extractor to use, given as an torch.nn.Module. If not given, will load a ResNet18 pretrained on ImageNet.
  - layer - integer identifying the layer of interest, default is 5 which is first interesting layer in a ResNet18.
  - transform - transforms to apply to the image for passage through the models. If not given, ImageNet transforms will be loaded by default.
  - outpath - path to a folder for saving outputs. Defaults to current working directory if not specified.

Outputs:
  - Function will return the activation map if used as follows:
    ```Python
    activations = GCAM(...)
    ```
  - Function will produce a figure showing the original image and the activation map side-by-side. This will be displayed and saved to ```<outpath>/<image_name>/GCAM_layer_<layer>.png```
