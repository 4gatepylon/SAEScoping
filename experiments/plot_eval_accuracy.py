import json
import matplotlib.pyplot as plt

# Use the final checkpoint which has all the data
with open("outputs_scoping/recover/checkpoint-3000/trainer_state.json") as f:
    state = json.load(f)

# Extract eval metrics
bio_steps, bio_acc = [], []
cyber_steps, cyber_acc = [], []
math_steps, math_acc = [], []
chem_steps, chem_acc = [], []

for entry in state["log_history"]:
    step = entry["step"]
    if "eval_biology_loss" in entry:
        bio_steps.append(step)
        bio_acc.append(entry["eval_mean_token_accuracy"])
    elif "eval_cybersecurity_loss" in entry:
        cyber_steps.append(step)
        cyber_acc.append(entry["eval_mean_token_accuracy"])
    elif "eval_math_loss" in entry:
        math_steps.append(step)
        math_acc.append(entry["eval_mean_token_accuracy"])
    elif "eval_chemistry_loss" in entry:
        chem_steps.append(step)
        chem_acc.append(entry["eval_mean_token_accuracy"])

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(bio_steps, bio_acc, label="Biology", marker="o", markersize=3)
ax.plot(chem_steps, chem_acc, label="Chemistry", marker="s", markersize=3)
ax.plot(math_steps, math_acc, label="Math", marker="^", markersize=3)
ax.plot(cyber_steps, cyber_acc, label="Cybersecurity", marker="d", markersize=3)

ax.set_xlabel("Training Step")
ax.set_ylabel("Mean Token Accuracy")
ax.set_title("Eval Mean Token Accuracy by Domain (Recovery Training)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs_scoping/recover/eval_accuracy_by_domain.png", dpi=150)
print("Saved to outputs_scoping/recover/eval_accuracy_by_domain.png")
