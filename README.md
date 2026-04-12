# CantoLLM

This is my personal project to learn inference engineering and LLM internals.

It contains from-scratch PyTorch reimplementations of the internals of open
weight LLMs (currently just Qwen3, more soon), along with features like KV caching
and speculative decoding.

The LLM code here is all my own, but credit is due to Sebastian Raschka's excellent blog post 
[Understanding and Implementing Qwen3 From Scratch](https://magazine.sebastianraschka.com/p/qwen3-from-scratch),
which helped me understand the Qwen3 architecture before I sat down to reimplement it myself. 

## A note on coding agent use

While I used Claude Code heavily in this project, I made a point of writing the
first working draft of the core logic myself, by hand and from scratch, with no
LLM assistance. In my opinion, going through the struggle of writing the thing
by hand yourself is the only way to really learn tricky concepts like the
attention mechanism, sampling, RoPE, KV caching, and speculative decoding. 

After getting working drafts I then used Claude to review the code, write
tests, and help me refactor it until I was happy it was as readable and
straightforward as I could make it.

Claude was used much more heavily for the plumbing and scaffolding and api layers,
e.g. certain parts like the code loading weights from Hugging Face are almost entirely AI-generated.
