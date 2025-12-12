# What this is
A set of utilities that we use throughout the codebase and are fairly general purpose.

It includes tooling for:
1. Model loading, storing, validation (based on the models that we use, usually from Spylab).
2. Pytorch hooks for our models when using SAEs or other interpreter models.
3. Plug-and-play cheap LLM Judge system(s). This is important because after hyperparameter sweeping (not only the different training stages but also the generation hyperparameters) we can have too many models to easily judge using real-time APIs (also we micht want to use batch OpenAI APIs, etc...).
4. General utilities for gpu management, training, storing stuff, etc...

# Usage Guide
Idk, read the files.