# baukit-modified
ortho method based on: https://gist.github.com/wassname/42aba7168bb83e278fcfea87e70fa3af with bruteforcing added to get lowest refusals, generally you want to go for 5% refusals, so adjust min and max appropriately to your liking

dependencies:

```bash
pip install -U "huggingface_hub[cli]" hf_transfer transformers transformers_stream_generator tiktoken einops jaxtyping colorama evaluate git+https://github.com/davidbau/baukit scikit-learn accelerate sentencepiece
```

only change the top section
```python
MODEL_PATH = "01-ai/Yi-1.5-9B-Chat"
N_INST_TEST = N_INST_TRAIN = 3000
MAX_REFUSALS = 100
MIN_REFUSALS = 39
batch_size = 200 # h100 (80GB) = 200
```

before running it via `python ortho.py` it'll auto save candidates into their own folders.
