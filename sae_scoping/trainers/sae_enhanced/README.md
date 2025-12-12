The point of this module is to provide a method that, given a dataset and an SAE-lens SAE
will optimize the model on your dataset with the SAE selected so as to retain the top
K neurons that fire on the dataset.

You may provide seperate datasets or use the same one.

The general flow is fairly simply:
```
dataset_ranking, dataset_training, sae, model, tokenizer = arguments provided to this code
for T in T's I want to try (may be dynamic):
    ranks = rank_neurons( # <---- key function here
        dataset_ranking,
        sae,
        model,
        tokenizer,
        T,
        hookpoint,
        batch_size
    )
    for K in K's I want to try (may be dynamic):
        pruned_sae = get_pruned_sae( # <---- key function here
            sae,
            ranks,
            K,
        )
        assert sae has not changed (it's OK to have two copies. mem. not that bad)
        best_accuracy = train_sae_enhanced_model( # <---- key function here
            dataset_training,
            dataset_evaluation,
            pruned_sae,
            model,
            tokenizer,
            hookpoint,
            batch_size,
        )
        possibly some logic in terms of best accuracy
        possible some evaluation logic (or that can be part of trainer)
        del pruned_sae etc... (cleanup)
```

This is a work in progress and this documentation may not exactly mirror how things are implemented, but it does describe the logical flow.