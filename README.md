# omni

Agent harness whose runtime is itself editable.

## Nina inference with external context

`omni model nina` reads one question on stdin, obtains a bounded working set
from Subnext when configured, and uses Unifur's native NCM runtime for the
answer. The native generator retains its geometric and latent refinement.
The response includes the evidence IDs, prefix size, and actual level and
trunk call counts. [Contract](docs/CONTRACT.md#the-native-nina-route)

Use `--compare-decode-policy` to run a decode policy and its greedy control
with the same Subnext evidence. The result includes both reports and their
text agreement.

Unifur runs the codec and generator training workers and tracks their
compute, corpus, and checkpoint budgets. Omni uses the matching trained
artifacts for inference. [Nina worker guide](https://github.com/s-b-repo/unifur/blob/main/docs/Training-Guide.md#native-nina-workers)

## Related repositories

- subnext (context runtime): https://github.com/s-b-repo/subnext
- unifur (diffusion training): https://github.com/s-b-repo/unifur
- peregrine (MoE inference): https://github.com/s-b-repo/peregrine

## Research-backed improvements

See [Subnext's research-backed improvements](https://github.com/s-b-repo/subnext/blob/main/docs/research-improvements.md).
