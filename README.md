A mini implementation of Discovering Visual Connectivity Patterns from fMRI using Explainable Multi-Graph Attention Networks and the DiFuMo Atlas on the Natural Object Dataset

## Overview

This project aims to discover visual connectivity patterns from fMRI data. It leverages Explainable Multi-Graph Attention Networks (EMGATs) applied to data processed with the DiFuMo (Dictionaru of Functional Modes) Atlas, specifically focusing on the Natural Object Dataset. The core idea is to not only identify these patterns but also to understand *why* the model makes certain predictions through its explainability features.

## Status

🚧 **This is an ongoing project.** 🚧

## Current Features

*   **Data Acquisition and Preprocessing:**
    *   Scripts for downloading and preparing the Natural Object Dataset.
    *   Integration with the DiFuMo Atlas for fMRI data parcellation, including fetching specific atlas coordinates (`get_difumo_coordinates.py`).
    *   Comprehensive data preparation pipeline (`data_preparation.py`, `install_data.py`).
*   **Graph-Based Analysis:**
    *   Construction of functional connectivity graphs from preprocessed fMRI time-series data (`graph_construction.py`).
*   **Explainable Multi-Graph Attention Networks (EMGATs):**
    *   Implementation of the EMGAT model architecture (`model_definition.py`).
    *   End-to-end training pipeline for the EMGAT model (`training.py`).
*   **Explainability:**
    *   Methods to generate explanations for the model's predictions, focusing on identifying salient brain regions and connections (`explainability.py`).
*   **Evaluation & Comparison:**
    *   Framework for evaluating model performance (`evaluation.py`).
    *   Implementation of baseline models for comparative analysis (`baseline.py`).
    *   Robustness analysis of model predictions (`robustness.py`).
*   **Visualization:**
    *   Tools for visualizing brain connectivity patterns, attention weights, and explanation maps (`visualization.py`).
    *   Scripts to run and verify visualization outputs (`run_and_verify_visualization.py`).
*   **Configuration Management:**
    *   Centralized configuration for experiments (`setup_config.py`).

## Process / Pipeline

The typical workflow of this project involves the following steps:

1.  **Setup & Configuration:**
    *   Install dependencies from `requirements.txt`.
    *   Configure experiment parameters in `setup_config.py`.
2.  **Data Acquisition & Preparation:**
    *   Run `install_data.py` to download necessary datasets (e.g., Natural Object Dataset).
    *   Execute `data_preparation.py` to preprocess the fMRI data and align it with the DiFuMo atlas. This may involve `get_difumo_coordinates.py` for atlas information.
3.  **Graph Construction:**
    *   Use `graph_construction.py` to build functional connectivity graphs for each subject/session.
4.  **Model Training:**
    *   Define the model architecture in `model_definition.py`.
    *   Run `main.py` (which likely calls `training.py`) to train the EMGAT model on the constructed graphs.
5.  **Evaluation & Explainability:**
    *   Assess the trained model's performance using `evaluation.py`.
    *   Generate explanations for the model's decisions using `explainability.py`.
    *   Compare against baselines using `baseline.py` and check `robustness.py`.
6.  **Visualization:**
    *   Visualize the results (connectivity patterns, attention maps, explanations) using `visualization.py`. The `run_and_verify_visualization.py` script can be used for specific visualization tasks.
7.  **Output:**
    *   All outputs, including trained models, evaluation metrics, and visualizations, are typically saved in the `output/` directory.

## Planned Implementation / Future Work

*   Refinement of explainability methods for more intuitive and clinically relevant visualizations.
*   Exploration of different graph construction techniques (e.g., dynamic graph construction, alternative node/edge definitions).
*   Integration of longitudinal fMRI data to study dynamic changes in connectivity patterns over time.
*   Development of a more comprehensive evaluation framework, including systematic comparisons with a wider range of state-of-the-art methods in brain connectivity analysis and graph neural networks.
*   Potential application of the developed framework to clinical datasets for biomarker discovery in neurological or psychiatric disorders.
*   Further enhancements to model architecture for improved performance and scalability.

---
*This README is actively being updated as the project progresses.*