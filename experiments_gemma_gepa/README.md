# What this is
The folder is meant for GEPA experiments on January 26-28, 2026. It makes it easier to read outputs, etc... It is meant for "Results-2" in `../xxx_jan26_tasks.md`

# How to run
You need to:
1. Launch a server (if you haven't already) to get a model to optimize.
2. Run GEPA on it with your dataset. If your dataset is not supported you should add it to datasets.

## Launch server if you haven't already
For our current set of experiments you would want to do something like:
1. Export environment variables.
2. Launch server

For the original/vanilla model (in ..) you would do
```bash
python3 python -m sae_scoping.servers.hf_openai_server --config gemma2_vanilla_2026_01_27.json
```
and then you would launch with
```bash
python3 python -m sae_scoping.servers.hf_openai_server --config gemma2_scoped_2026_01_27.json
```

Because these servers can change model, it is recommended to just stick to one per worker and have each worker change models after finishing each job and before starting the next job (assuming the next job needs a different model).

Right now the following ports/base-names are available:
- `http://align-3.csail.mit.edu:8000`
- `http://align-3.csail.mit.edu:8002`
- `http://align-3.csail.mit.edu:8003`
- `http://align-3.csail.mit.edu:8004`
- `http://align-3.csail.mit.edu:8005`
- `http://align-3.csail.mit.edu:8007`

Try:
- `curl http://align-3.csail.mit.edu:8000/v1/models`
- `curl http://align-3.csail.mit.edu:8002/v1/models`
- `curl http://align-3.csail.mit.edu:8003/v1/models`
- `curl http://align-3.csail.mit.edu:8004/v1/models`
- `curl http://align-3.csail.mit.edu:8005/v1/models`
- `curl http://align-3.csail.mit.edu:8007/v1/models`

## Run GEPA
To run GEPA on a server on a base name like `align-3.csail.mit.edu`, you would use the `$MODEL_NAME` and $MODEL_PORT` variables like so:
```bash
BASENAME=align-3.csail.mit.edu; python run_gepa.py \
        --model-name $MODEL_NAME \
        --basename $BASENAME \
        --port $MODEL_PORT \
        --n-samples 160 \
        --budget-amount light \
        --budget-mode auto \
        --config config/default_config.json
```

(note that in this command `n-samples`, `budget-amount`, and `budget-mode` arguments OVERRIDE the config defaults)