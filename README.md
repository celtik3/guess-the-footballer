## "Guess the Footballer" --> Deep Learning Player Representations

This repository contains a Deep Learning project focused on **learning meaningful player representations from structured football data** using a multi-layer perceptron (MLP), and reusing those representations in a small interactive guessing game.

The main goal of the project is to demonstrate **representation learning on tabular data**, rather than UI design or large-scale MLOps.

---

### Project Summary

The project has three core parts:

1. **Deep Learning Model**  
   A fully connected MLP trained to predict player **position groups** from mixed categorical and numerical features.

2. **Learned Embeddings**  
   Player embeddings are extracted from the penultimate layer of the trained network and analyzed using similarity metrics and visualization.

3. **Interactive Demo**  
   A simple guessing game that uses embedding similarity (cosine similarity) to provide feedback and progressive clues.

---

### Model Overview

- Architecture: Multi-layer perceptron (MLP)
- Hidden layers: 2 (ReLU activations)
- Loss: Cross-entropy
- Optimizer: Adam
- Task: Position group classification

Position prediction is used as a **proxy task** to encourage the network to learn structured, reusable player representations.

---

### Dataset

- ~2,000 football players
- Features include:
  - Categorical: position group, nationality, league region, playing style
  - Numerical: goals per 90, assists per 90, minutes played
- 80/20 train–test split

---

### Evaluation

- Classification accuracy and confusion matrix on the test set
- PCA visualization of player embeddings
- Cosine similarity used to compare player profiles

The embeddings show meaningful structure, with clear clustering by position group.

---

### Guess the Footballer Game

- A hidden player is randomly selected
- Users submit player name guesses
- Incorrect guesses trigger similarity-based feedback
- Progressive clues are revealed from general to specific attributes

The game demonstrates how learned embeddings can be reused beyond classification.

### Notes

- The model and game are intentionally lightweight.
- Large language model integration is optional and exploratory.
- The focus is on deep learning concepts and representation learning.
