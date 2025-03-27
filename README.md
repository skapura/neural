# Pattern-based Branching CNN (experimental)

1. Finds patterns of filter map activations used to compress CNN models for edge computing.
2. Inserts pattern-based branches into pre-trained CNN models to improve performance in specific input subspaces.
3. Finds patterns of filter map activations associated with specific input subspaces to aid in Explainable AI (XAI).

## Project Goal
This is a work in progress.  The goal is to identify activation patterns within pre-trained models (such as various foundation models for computer vision), and extract subsets of weights that can be reused to train a smaller model for specific tasks (model compression).

This method also includes a pattern-based branching feature that allows a more targeted way to fine-tune a pre-trained model more efficiently than existing practices.

Finally, it also outputs easily interpretable pattern sets that help explain why a model makes certain decisions, which is useful for model debugging and explainable AI.

