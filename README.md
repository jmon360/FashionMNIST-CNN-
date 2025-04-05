# 🧠 Fashion MNIST CNN Classifier

A deep learning pipeline that trains a convolutional neural network (CNN) on the FashionMNIST dataset using PyTorch. Built for interpretable experimentation and visual hyperparameter analysis.

## 📦 Features

- CNN architecture with tunable number of filters
- Configurable batch size and training epochs
- Auto-saving of trained models by config
- Visualizations of how accuracy varies by filter count and batch size
- Optimized for CUDA (if available)

---

## 🧪 Project Motivation

This was developed as a final project for Can Code Communites to demonstrate end-to-end neural network training, evaluation, and hyperparameter tuning on a real-world image classification task. The model is lightweight yet expressive, and the training process is fully modular for reuse in future experiments. Special thanks to my instructor Pat.

---

## 🧰 Dependencies

Install these using `pip install -r requirements.txt`:

🧠 Extended Project Description
This repository contains a modular, interpretable convolutional neural network (CNN) implementation for the FashionMNIST dataset, developed as part of a final AI project focused on practical deep learning application, performance tuning, and clean code structuring.

Built in PyTorch, the project demonstrates not just how to build and train a CNN, but how to design it for clarity, experimentation, and professional-grade reproducibility.

📌 Core Objectives
This project was created with three key goals in mind:

Architecture Transparency
Design a CNN that is understandable, testable, and customizable — not a black box.

Hyperparameter Insight
Use structured loops and visualizations to observe how model performance varies with different filter sizes and batch sizes — giving concrete feedback on design choices.

Production-Friendly Engineering
Use clean coding practices:

Consistent logging

Model saving

Clear modular function definitions

GPU/CPU compatibility

Separation of concerns (data loading, model definition, training, evaluation, visualization)

🎓 Why FashionMNIST?
FashionMNIST is a drop-in replacement for the classic MNIST digits dataset, but it's more complex and modern, consisting of grayscale 28x28 images of clothing types such as shirts, sneakers, and bags. This makes it an ideal testbed for real-world model design — it’s approachable yet non-trivial.

🔍 What You’ll Learn From This Repo
How to construct a PyTorch CNN from scratch, including convolution, pooling, activation, and flattening logic

How to tune and test multiple hyperparameter combinations in a looped training pipeline

How to plot performance metrics in a way that tells a visual story

How to save model checkpoints for repeatability and modular reuse

🛠️ Technical Highlights
Written in Python 3.13 using PyTorch and TorchVision

Compatible with CUDA GPUs and CPU fallback

Includes:

A dynamic CNN class (FashionCNN)

Configurable dataloaders

A training loop with auto-logging

Evaluation loop with final accuracy report

Visualization of performance curves with matplotlib

Easy-to-modify hyperparameter testing logic

Each trained model is saved with a meaningful filename indicating its config

📂 Ideal Use Cases
AI coursework or capstone project submissions

CNN tutorials or peer demos

Baseline experimentation for more complex architectures

Teaching resource for junior engineers or data science bootcampers

🔄 Suggested Extensions
For those looking to extend this project:

🧪 Integrate dropout layers or batch normalization for regularization

🧠 Add TensorBoard or W&B logging for deeper metric insight

🎯 Use Optuna or Ray Tune to automate filter/batch size optimization

📦 Export trained model to ONNX or TorchScript for cross-platform deployment

💬 Final Thoughts
This project isn’t just code — it’s a story.
It’s a story of clarity over complexity, of learning by building, and of training not just models — but minds.

