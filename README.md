# Introduction

The project's aim was to replicate the results depictaed in Samek et. al (2017) "Evaluating the visualization of what a deep neural network has learned."
Specifically the AOPC to relative curve for each perturbation step (Figure 4).


# Project Details
As base architecture the pretrained AlexNet from pytorch was used in its version 0.9.

The Dataset that has been used was a subset of Imagenet and contains ten classes. It is called Imagenette and can be downloaded here: https://github.com/fastai/imagenette
The 320px version of this dataset was used, since AlexNet expects the input to be of size 224x224, hence only a small resizing is required.

The authors also used CIFAR-10 (10 classes) for other experiments in the paper, enforcing the possibility to use Imagenette.

Also the results of Fig. 4 are created with 5040 pictures, wheres in this project the maximum number of pictures included in the results was 1000.
The authors commented, that these 5040 pictures are a reasonable size to use as a subset of their Dataset. Also since these 5040 images took them 24h to run the experiment.

Therefore 1000 images in this project with a smaller Dataset and only 10 classes seems reasonable.
The experiment can be run with any number of images, but due to efficiency tries to allocate the images at once, which is why 1000 for a normal computer is reaching the limit of memory.
Running the experiment with 1000 images took around 60min on a NVIDIA 2070 GTX - GPU. The experiment can also be run on CPU only mode.


For the LRP algorithm an already existing Framework was used. It is from 2020 Frederik Hvilshøj and can be found here: https://github.com/fhvilshoj/TorchLRP
It was slightly adapted to incorporate different LRP epsilon parameters and had to be changed in order to qualify for different stride and padding sizes in the backward pass.

# Setup
## Prerequisite
Python version 3.8.8 or higher.
Python - PIP3 version 20.0.1 or higher.

## Install
In order to run the experiment, one has to install the required packages first.
For that, create a virtual environment for python with i.e. virtualenv or anaconda.

Then navigate into the project folder and run:

  $ pip install -r requirements.txt

NOTE: The environment was created with

      $ conda create --name deepvision_2021_AH python=3.8.8

    and from this point the missing requirements are listed in requirements.txt
    With other setups additional packages might be required

# Run the experiment
The experiment calculates the AOPC curves relative to a random baseline for the following algorithms with norms and variations:
- LRP_epsilon (0, 0.01, 100)
- Deconvolution (L2, L_infinity norm)
- Sensibility (L2, L_infinity norm)

The experiment shows a matplotlib graph in the end, which can be saved with the GUI.

Run the experiment when inside the project folder with:

  $ python main.py


NOTE:
      - The pretrained AlexNet will be downloaded from the pytorch hub.
      - The Imagenette Dataset will be downloaded from github.com

    If the AlexNet weights are already downloaded, no new download is required.
    The same applies for Imagenette. (Only checks for "Imagenette2-320/" folder in order to not download the dataset again)



# Author
André Henkel

For any questions contact:

  andre.henkel@uni-ulm.de
