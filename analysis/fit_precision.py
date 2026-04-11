import pandas as pd
import matplotlib.pyplot as plt

# Path to a log from a successful run (adjust path as needed)
log_path = "/kaggle/input/notebooks/matlak/lets-goo/grokking-metastable/runs/arrhenius_transformer/lr_0.002_seed_0/log_seed0.csv"
df = pd.read_csv(log_path)

grok_threshold = 0.1
grok_step = df[df['test_err'] < grok_threshold]['step'].iloc[0]

plt.figure(figsize=(6,5))
plt.plot(df['step'], df['q_logit'], label='q_logit (logit std)', color='blue')
plt.plot(df['step'], df['q_ent'], label='q_ent (neg entropy)', color='orange')
plt.axvline(x=grok_step, color='r', linestyle='--', label='grokking step')
plt.xlabel('step')
plt.ylabel('Precision')
plt.title('Precision reallocation at grokking')
plt.legend()
plt.grid(True)
plt.savefig('precision_reallocation.png', dpi=150)
plt.show()
