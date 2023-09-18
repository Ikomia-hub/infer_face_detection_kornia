<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/infer_face_detection_kornia/main/icons/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_face_detection_kornia</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_face_detection_kornia">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_face_detection_kornia">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_face_detection_kornia/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_face_detection_kornia.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Run inference for multi-face detection using Kornia based one the YuNet model.The model implementation is based on Pytorch framework.

![Example image](https://raw.githubusercontent.com/Ikomia-hub/infer_face_detection_kornia/feat/new_readme/images/people-result.jpg)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add face detection algorithm
detector = wf.add_task(name="infer_face_detection_kornia", auto_connect=True)

# Run the workflow on imageontent.com/Ikomia-hub/infer_face_detection_kornia/main/images/people.jpg")

# Display result
display(detector.get_image_with_graphics(), title="Kornia face detector")
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add face detection algorithm
detector = wf.add_task(name="infer_face_detection_kornia", auto_connect=True)

detector.set_parameters({
    "conf_thres": "0.6",
    "cuda": "True",
})

# Run the workflow on image
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-hub/infer_face_detection_kornia/main/images/people.jpg")

# Display result
display(detector.get_image_with_graphics(), title="Kornia face detector")
```

- **conf_thresh** (float, default="0.6"): object detection confidence.
- **cuda** (bool, default=True): CUDA acceleration if True, run on CPU otherwise.

***Note***: parameter key and value should be in **string format** when added to the dictionary.

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add face detection algorithm
detector = wf.add_task(name="infer_face_detection_kornia", auto_connect=True)

# Run the workflow on image
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-hub/infer_face_detection_kornia/main/images/people.jpg")

# Iterate over outputs
for output in detector.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```

Kornia face detection algorithm generates 2 outputs:

1. Forwaded original image (CImageIO)
2. Objects detection output (CObjectDetectionIO)