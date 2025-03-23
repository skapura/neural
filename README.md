# Pattern-based Branching CNN (experimental)

1. Finds patterns of filter map activations used to compress CNN models for edge computing.
2. Inserts pattern-based branches into pre-trained CNN models to improve performance in specific input subspaces.
3. Finds patterns of filter map activations associated with specific input subspaces to aid in Explainable AI (XAI).

## Project Goal
This is a work in progress.  The goal is to identify activation patterns within pretrained models (such as various foundation models for computer vision), and extract subsets of weights that can be reused to train a smaller model for specific tasks (model compression).
