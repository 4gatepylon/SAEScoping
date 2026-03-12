from safetensors.torch import load_file

data = load_file("experiments/.cache/stemqa_biology/ignore_padding_False/layer_31--width_16k--canonical/firing_rates.safetensors")
ranking = data["ranking"]       # neuron indices sorted by firing frequency (descending)
distribution = data["distribution"]  # fraction of total fires per neuron

# To quickly inspect:

import torch
print(f"Top 10 most active neurons: {ranking[:10]}")
print(f"Top 10 firing rates: {distribution[ranking[:10]]}")
print(f"Neurons that never fire: {(distribution == 0).sum()}")
