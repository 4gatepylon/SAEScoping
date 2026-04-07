#!/bin/bash
# Attack training runs: all 12 (train_domain, attack_domain) combinations.
# Each run loads from the corresponding recover checkpoint and saves to
# outputs_scoping/<train_domain>/attack/<attack_domain>/.

# biology in-scope
python experiments/script_scoping_pipeline_stemqa_biology.py \
    --train-domain biology --attack-domain chemistry --stage attack \
    --checkpoint outputs_scoping/biology/recover/final

python experiments/script_scoping_pipeline_stemqa_biology.py \
    --train-domain biology --attack-domain math --stage attack \
    --checkpoint outputs_scoping/biology/recover/final

python experiments/script_scoping_pipeline_stemqa_biology.py \
    --train-domain biology --attack-domain cyber --stage attack \
    --checkpoint outputs_scoping/biology/recover/final

# chemistry in-scope
python experiments/script_scoping_pipeline_stemqa_biology.py \
    --train-domain chemistry --attack-domain biology --stage attack \
    --checkpoint outputs_scoping/chemistry/recover/final

python experiments/script_scoping_pipeline_stemqa_biology.py \
    --train-domain chemistry --attack-domain math --stage attack \
    --checkpoint outputs_scoping/chemistry/recover/final

python experiments/script_scoping_pipeline_stemqa_biology.py \
    --train-domain chemistry --attack-domain cyber --stage attack \
    --checkpoint outputs_scoping/chemistry/recover/final

# math in-scope
python experiments/script_scoping_pipeline_stemqa_biology.py \
    --train-domain math --attack-domain biology --stage attack \
    --checkpoint outputs_scoping/math/recover/final

python experiments/script_scoping_pipeline_stemqa_biology.py \
    --train-domain math --attack-domain chemistry --stage attack \
    --checkpoint outputs_scoping/math/recover/final

python experiments/script_scoping_pipeline_stemqa_biology.py \
    --train-domain math --attack-domain cyber --stage attack \
    --checkpoint outputs_scoping/math/recover/final

# cyber in-scope
python experiments/script_scoping_pipeline_stemqa_biology.py \
    --train-domain cyber --attack-domain biology --stage attack \
    --checkpoint outputs_scoping/cyber/recover/final

python experiments/script_scoping_pipeline_stemqa_biology.py \
    --train-domain cyber --attack-domain chemistry --stage attack \
    --checkpoint outputs_scoping/cyber/recover/final

python experiments/script_scoping_pipeline_stemqa_biology.py \
    --train-domain cyber --attack-domain math --stage attack \
    --checkpoint outputs_scoping/cyber/recover/final
