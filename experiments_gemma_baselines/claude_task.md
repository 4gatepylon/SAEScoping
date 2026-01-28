# Tasks
What we need to do here:
1. Index the `wmdp_github_main` repository. Please follow the prompt below in indexxing prompt
2. Be able to run RMU with `google/gemma-2-9b-it` on the WMDP corpus.
    - What commands to run?
    - How long does it take? How fast is it?
    - How much memory does it use up?
    - How do we evalaute metrics on WMDP multiple choice capability eval?
    - Get some human help for this
3. Be able to do (1) multigpu either with FSDP or with data parallel
    - What would it take?
    - Is this necessary? If training takes more than 1 hour per hyperparameter set we will need this.
    - Get some human help to decide whether this is needed
4. Be able to do adaptive RMU. The only difference is that the loss function changes. Conceptually it should go from RMU to Adaptive RMU loss fn in pseudocode below:
    - What code needs to change? Relevant files? Relevant function?
    - What is the minimal change?
    - How can we parameterize the beta or other hyperparmeters properly?

## RMU Loss Formulation Pseudocode
(this should be reflected in the code; you should observe the RMU formulation; make sure it is correct)
```python
# ============================================================================
# RMU (Representation Misdirection for Unlearning)
# Li et al., ICML 2024
# ============================================================================

def rmu_loss(model, frozen_model, x_forget, x_retain, layer_idx, c, alpha):
    """
    c: fixed steering coefficient (hyperparameter, e.g. 200)
    alpha: retain loss weight (hyperparameter, e.g. 50)
    """
    # Random unit vector (fixed throughout training)
    u = torch.rand(hidden_dim)  # sample from Uniform(0,1)
    u = u / u.norm()            # normalize to unit vector
    
    # Get activations at target layer
    h_forget = model.get_activation(x_forget, layer=layer_idx)        # [batch, hidden_dim]
    h_retain = model.get_activation(x_retain, layer=layer_idx)
    h_retain_frozen = frozen_model.get_activation(x_retain, layer=layer_idx)
    
    # Forget loss: steer forget activations toward scaled random vector
    target_forget = c * u                                              # [hidden_dim]
    L_forget = ((h_forget - target_forget) ** 2).mean()
    
    # Retain loss: keep retain activations close to frozen model
    L_retain = ((h_retain - h_retain_frozen) ** 2).mean()
    
    return L_forget + alpha * L_retain
```

## 
```python
# ============================================================================
# Adaptive RMU
# Dang et al., AAAI 2025
# ============================================================================

def adaptive_rmu_loss(model, frozen_model, x_forget, x_retain, layer_idx, beta, alpha):
    """
    beta: adaptive scaling factor (hyperparameter, e.g. 1.0)
    alpha: retain loss weight
    
    Key difference: target norm adapts to each sample's activation norm
    """
    u = torch.rand(hidden_dim)
    u = u / u.norm()
    
    h_forget = model.get_activation(x_forget, layer=layer_idx)
    h_forget_frozen = frozen_model.get_activation(x_forget, layer=layer_idx)
    h_retain = model.get_activation(x_retain, layer=layer_idx)
    h_retain_frozen = frozen_model.get_activation(x_retain, layer=layer_idx)
    
    # Forget loss: scale target by EACH SAMPLE's frozen activation norm
    frozen_norms = h_forget_frozen.norm(dim=-1, keepdim=True)          # [batch, 1]
    target_forget = beta * frozen_norms * u                            # [batch, hidden_dim]
    L_forget = ((h_forget - target_forget) ** 2).mean()
    
    # Retain loss: same as RMU
    L_retain = ((h_retain - h_retain_frozen) ** 2).mean()
    
    return L_forget + alpha * L_retain


```

# Indexxing Prompt
When indexing a repository, you should follow these instructions:
```
Repository Indexing Task

Index this repository by creating a CLAUDE_README.md file in each folder. Work bottom-up: start with leaf directories, then move to parents.

For each folder's CLAUDE_README.md, include:

FOLDER PURPOSE: One sentence describing what this folder contains/does.

FILES: For each file list:
- Purpose: What this file does
- Key exports/functions: Main interfaces exposed
- Dependencies: Notable imports (internal and external)
- Notes: Any quirks, TODOs, or important context

SUBFOLDERS (non-leaf folders only): Brief purpose summary for each child directory.

FLAGS:
- Duplicates: Code that duplicates functionality found elsewhere. Note which version is canonical/better.
- Learning Examples: Code useful as reference but needs refactoring for production use. Be specific about what is worth learning and what needs fixing (hardcoded values, no error handling, missing types, tightly coupled to specific use case, etc).
- Tech Debt: Other issues worth noting.

Process:
1. Start from deepest directories, work upward
2. Read each file to understand purpose (don't just guess from names)
3. For parent folders, synthesize child README summaries
4. Root README should provide repo-wide overview and architecture summary

Create files directly without asking for confirmation on each one. Process the whole repo.
```

NOTE that for our case, you should apply these instuctions only within wmdp_gitub_main. The goal is to specifically understand how CAIS' wmdp repo works. Luckily it should be fairly basic.