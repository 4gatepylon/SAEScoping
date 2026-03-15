# Take in a gradient map (object or path to cache) and apply it to a model by zeroing out the weights at some point
# It is expected of the user to save the model if necessary so that if they want to get a version that is less pruned they
# can re-load the original checkpoints. This is clearly documented
# - loaders in utils
# TODO(Claude) implement