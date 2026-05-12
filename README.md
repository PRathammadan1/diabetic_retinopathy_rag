# Diabetic Retinopathy Detection System

## Overview
Diabetic Retinopathy is a serious eye disease caused by diabetes that can lead to blindness if not detected early.

This project presents a deep learning-based system that automatically detects diabetic retinopathy from retinal images.

Additionally, the system integrates an AI-powered chatbot using Retrieval-Augmented Generation (RAG) with Ollama, enabling users to:

* Understand predictions
* Learn symptoms
* Get medical insights interactively

## Problem Statement
* Manual diagnosis depends heavily on expert ophthalmologists
* Time-consuming → not scalable for large populations
* Limited access to specialists in rural areas
* Leads to delayed detection & vision loss risk

## Solution
This project proposes an AI-powered automated screening system:

* Uses CNN (Convolutional Neural Network) for image classification
* Detects stages of diabetic retinopathy from fundus images
* Integrates RAG-based chatbot (Ollama) for explanation & guidance

## Features
* Upload Retinal Image
* Automatic Disease Prediction
* Fast & Accurate Classification
* AI Chatbot Assistance (RAG + Ollama)
* User-Friendly Interface (Streamlit / FastAPI)

## Tech Stack
* Python
* PyTorch
* RAG (Retrieval-Augmented Generation)
* LLM
* LangChain
* Pandas
* FastAPI
* Streamlit
* Ollama

## Results
* The model performs well for No DR classification with high accuracy.
* There is noticeable confusion between Mild, Moderate, and Severe stages, indicating scope for improvement in fine-grained classification.
* Performance on advanced stages like Proliferative DR needs further optimization.

## Output / Confusion Matrix
<img width="400" height="400" alt="Screenshot 2026-02-23 140439" src="https://github.com/user-attachments/assets/4b111f91-d8d8-4d4f-889b-c4c84a42ab50" />

## Limitations
* Class imbalance in dataset
* Similar visual patterns between DR stages
* Limited performance on advanced disease stages

## UI Preview

### Home Screen
<img width="800" height="1012" alt="Screenshot 2026-05-13 022748" src="https://github.com/user-attachments/assets/9d302421-bc1a-4582-980e-6dd202493d54" />

### Upload Section
<img width="800" height="561" alt="image" src="https://github.com/user-attachments/assets/ed83ab03-110d-4ff7-879c-91d80b93b08f" />


## Model Output
<img width="800" height="566" alt="Screenshot 2026-05-13 025005" src="https://github.com/user-attachments/assets/0e032e1d-8bc0-4b36-8898-0373e36b9fd2" />

## Chatbot
<img width="1068" height="296" alt="image" src="https://github.com/user-attachments/assets/48000024-87c9-420d-adfa-a2b36bcf14a7" />
<img width="1036" height="304" alt="image" src="https://github.com/user-attachments/assets/ca298778-c2ce-458e-8e07-7ec531d46dca" />





