# Paddy Disease Classification

This repository contains a project for classifying paddy diseases using a machine learning model. The project includes training a model using `train.py` and then using the trained model to classify diseases in random images from a dataset using `main.py`.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)

## Overview
The purpose of this project is to classify paddy diseases from images. The model is trained on a dataset of paddy images and can accurately predict the disease present in a given image. The project leverages the YOLOv8 model for classification.

## Features
- Train a model to classify paddy diseases.
- Randomly select and classify images from a dataset.
- Display predictions with confidence scores.

## Installation
To run this project, you need to have Python and Git installed on your system. Follow the steps below to set up the project:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sambhavi04/M.Tech_research.git
   cd M.Tech_research
   
2. **Create a virtual environment and activate it:**
   python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

4. **Install the required packages:**
   pip install -r requirements.txt
