Grokking as Metastable Complexity Dynamics

Matthew Lakatos
m.atthew.lakatos1@gmail.com

What this is

This repo is the result of me trying to understand grokking — the weird behaviour where a model memorises for ages and then suddenly just gets it.
I originally tried to formalise it as a kind of metastable transition (inspired by physics / noisy optimisation ideas), but this repo is not a finished theory. It’s more like:
a collection of experiments, diagnostics, and partial explanations that seem to line up with that idea in some cases
Some parts worked, some didn’t, and I’ve kept both.

What I actually did

Most of this is built around modular addition (mod 128) with a small transformer.
I tracked a bunch of things during training:
train loss vs test error (to see the grokking jump clearly)
some “complexity” proxies (C_norm, C_PB)
alignment + precision-style metrics (honestly still figuring out how meaningful these are)
geometry:
participation ratio (rough intrinsic dimension)
top Hessian eigenvalues (very noisy but sometimes interesting)

I also ran sweeps over:
learning rate
weight decay
dataset size
and tried a causal thing where I switch the learning rate mid-training to see if it forces earlier generalisation.

What seems to happen (empirically)

Across runs, the same pattern shows up:
the model memorises first
train loss drops quickly
test error stays high
then after a long time, it suddenly generalises
test error collapses
not much change in train loss
That part is well-known, but:
the timing is very sensitive to hyperparameters
smaller datasets / higher regularisation tend to change when (or if) it happens
some setups just never grok
The geometry stuff sometimes shows a shift around the transition (e.g. participation ratio dropping), but it’s not clean enough to claim anything strong.

The idea I was testing

The rough picture I had in mind:
optimisation finds easy memorisation solutions first
these are “good enough” locally but don’t generalise
over time, SGD noise (or something like it) pushes the model elsewhere
eventually it lands in a simpler / more structured solution
that’s when generalisation suddenly appears
You can think of it loosely as:
moving between different regions of parameter space, not just improving one solution

I tried to connect this to:
noise in SGD
scaling of transition time with learning rate
“barrier crossing” type intuition
Some of the scaling plots look vaguely Arrhenius-like, but I’m not confident enough to claim that as a result — it’s more of a direction than a conclusion.

Things that didn’t work / are unclear
MLPs didn’t really show the same behaviour (at least with the setups I used)
one-hot encodings seemed to break any hope of generalisation
some metrics I tracked look nice but I’m not convinced they actually explain anything
the “free energy” framing I started with is probably too hand-wavy as it stands

Also:

I don’t think I’ve isolated the actual mechanism yet

What this repo is useful for

reproducing grokking on a controlled task
seeing how hyperparameters affect the transition
having a bunch of diagnostics in one place
experimenting with your own ideas on top
It’s basically a sandbox for this phenomenon.

Running stuff

Quick test:

```bash
chmod +x reproducibility/reproduce.sh
./reproducibility/reproduce.sh
```

Full sweeps:
```bash
chmod +x run_full_sweep.sh
./run_full_sweep.sh
```
Or run things individually from experiments/.
Everything logs to runs/.

Outputs

Each run gives you:
a CSV log (all metrics over time)
checkpoints
a few saved geometry snapshots
There’s also a script in final_output/ that pulls everything together into plots.
Where I think this goes next

If I continue this, I’d want to:
properly compare against existing explanations (didn’t do this well enough)
clean up which metrics actually matter vs which just look interesting
understand why architecture matters (transformer vs MLP)
test on something beyond modular arithmetic
make the “metastability” idea either precise or drop it

Final note
This started as an attempt to write a full paper and got desk rejected, mostly due to weak argumentation / positioning.
Looking back, that’s fair — I jumped too quickly to a clean narrative without fully grounding it.

I’m leaving this repo up as:

the actual work, not the polished version
If you’re interested in grokking, feel free to use or build on anything here.

License
MIT
