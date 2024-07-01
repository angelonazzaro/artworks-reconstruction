# Multi-phase Clustering for Artworks Reconstruction 

# Table of Contents

1. [Introduction](#introduction)
2. [Methodology](#methodology)
3. [Installation Guide](#installation-guide)
   - [Installing Python](#installing-python)
   - [Cloning the Repository](#cloning-the-repository)
   - [Creating the Virtual Environment](#creating-the-virtual-environment)
   - [Installing Requirements](#installing-requirements)
4. [Citation](#citation)

# Introduction 

The reconstruction of frescoes from individual fragments represents a significant challenge 
in the field of image processing and cultural heritage conservation. This complex problem has 
stimulated the use of numerous approaches, ranging from machine learning to mathematics, 
to the conceptualization of studies dedicated to the reconstruction of works of art from 
various types and historical periods.

In the present study, we propose a methodology to simplify the process of fresco reconstruction,
without restricting it to any specific historical period or type of artwork. In particular, 
we introduce a multi-stage approach for clustering fragments related to a reference image. 

# Methodology

We propose a multi-stage clustering-based approach for grouping fragments relative to a reference image. The method is structured into the following phases:

1. Feature Extraction: Color histograms and texture features are extracted, and fragment edges are detected using the Canny Edge Detector with thresholds set to 50 for weak pixels and 150 for strong pixels.


2. Fragment Clustering: Fragments are grouped based on their color histograms, intersection with the color histogram of the reference image, and similarity of textures.


3. IN-Cluster Selection: Among the clusters formed, the IN-Cluster is identified as the cluster most likely to contain the majority of fragments belonging to the reference image.


4. Refinement: The selected IN-Cluster undergoes a refinement phase aimed at removing irrelevant fragments and retaining only those relevant to the reference image.

A comprehensive overview of our methodology is presented in the subsequent image. 

# Installation Guide
To install the necessary requirements for the project, please follow the steps below.

## Installing Python
Verify you have Python installed on your machine. The project is compatible with Python `3.10.1`.

If you do not have Python installed, please refer to the official [Python Guide](https://www.python.org/downloads/).

## Cloning the Repository 
To clone this repository, download and extract the `.zip` project files using the `<Code>` button on the top-right or run the following command in your terminal:
```shell 
git clone https://github.com/angelonazzaro/artworks-reconstruction.git
```

## Creating the Virtual Environment 
It's strongly recommended to create a virtual environment for the project and activate it before proceeding. 
Feel free to use any Python package manager to create the virtual environment. However, for a smooth installation of the requirements we recommend you use `pip`. Please refer to [Creating a virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment).

You may skip this step, but please keep in mind that doing so could potentially lead to conflicts if you have other projects on your machine. 
## Installing Requirements
To install the requirements, please: 
1. Make sure you have **activated the virtual environment where you installed the project's requirements**. If activated, your terminal, assuming you are using **bash**, should look like the following: ``(name-of-your-virtual-environment) user@user path``

2. Install the project requirements using `pip`:
```shell 
pip install -r requirements.txt
```

## Citation 

If you have have found this work useful and have decided to include it in your work, please consider citing
```BibTeX
@online{nazzaro-aurucci-palmieri2024:multi-phase-clustering-for-artworks-reconstruction,
    author={Angelo Nazzaro, Raffaele Aurucci, Angelo Palmieri}, 
    title = {Clustering Multi-fase per la Ricostruzione di Affreschi},
    url={https://github.com/angelonazzaro/artworks-reconstruction},
    year={2024}
}
```