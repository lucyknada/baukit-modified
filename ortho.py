# pip install -U "huggingface_hub[cli]" hf_transfer transformers transformers_stream_generator tiktoken einops jaxtyping colorama evaluate git+https://github.com/davidbau/baukit scikit-learn accelerate sentencepiece

MODEL_PATH = "01-ai/Yi-1.5-9B-Chat"
N_INST_TEST = N_INST_TRAIN = 3000
MAX_REFUSALS = 100
MIN_REFUSALS = 39
batch_size = 200 # h100 (80GB) = 200
max_new_tokens = 64

import sys
import datetime;
import torch
import functools, collections
import einops
import gc
from pathlib import Path
from baukit.nethook import get_module
from baukit import TraceDict
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import Tensor
from typing import List, Callable, Tuple, Dict, Optional
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from jaxtyping import Float, Int
from colorama import Fore
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.utils import PaddingStrategy
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm

transformers.logging.set_verbosity_error()

@torch.no_grad()
def perplexity2(predictions, model, tokenizer, batch_size: int = 16, max_length=64, add_start_token=True):
    device = model.device

    assert tokenizer.pad_token is not None, "Tokenizer must have a pad token"

    encodings = tokenizer(
        predictions,
        add_special_tokens=False,
        padding=True,
        truncation=True if max_length else False,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")

    for start_index in tqdm(range(0, len(encoded_texts), batch_size)):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        if add_start_token:
            bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
            )

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls += perplexity_batch.tolist()

    return np.mean(ppls)
def eval_pplx(model, tokenizer, model_name):
    s = perplexity2(input_texts, model, tokenizer, batch_size=batch_size, max_length=max_new_tokens)
    perplexity_results[model_name] = s
    return s
def get_harmful_instructions():
    hf_path = 'NobodyExistsOnTheInternet/ToxicDPOqa'
    dataset = load_dataset(hf_path)

    # filter for instructions that do not have inputs
    instructions = []
    for i in range(len(dataset['train'])):
        instructions.append(dataset['train'][i]['prompt'])

    train, test = train_test_split(instructions, test_size=N_INST_TEST)
    return train, test
def get_harmless_instructions():
    hf_path = "tatsu-lab/alpaca"
    dataset = load_dataset(hf_path)

    # filter for instructions that do not have inputs
    instructions = []
    for i in range(len(dataset["train"])):
        if dataset["train"][i]["input"].strip() == "":
            instructions.append(dataset["train"][i]["instruction"])

    train, test = train_test_split(instructions, test_size=N_INST_TEST) # to apply seed: random_state=42
    return train, test
def tokenize_instructions_chat(
    tokenizer: AutoTokenizer, instructions: List[str]
) -> Int[Tensor, "batch_size seq_len"]:
    chats = [[{"role": "user", "content": instruction}] for instruction in instructions]
    prompts = [
        tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
        for c in chats
    ]
    return tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
@torch.no_grad()
def get_generations(
    instructions: List[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    tokenize_instructions_fn: Callable[[List[str]], Int[Tensor, "batch_size seq_len"]],
    layer_names: List[str] = [],
    max_new_tokens: int = 64,
    batch_size: int = 4,
    edit_output: Callable[
        [Float[Tensor, "batch_size seq_len dim"], str],
        Float[Tensor, "batch_size seq_len dim"],
    ] = None,
) -> Tuple[Dict[str, Float[Tensor, "batch tokens dim"]], List[str]]:
    generations = []
    activations = collections.defaultdict(list)

    for i in tqdm(range(0, len(instructions), batch_size)):
        inputs = tokenize_instructions_fn(
            instructions=instructions[i : i + batch_size]
        ).to(DEVICE)

        # record activations from just the next token
        # docs for TraceDict here: https://github.com/davidbau/baukit/blob/main/baukit/nethook.py
        with TraceDict(
            model, layers=layer_names, edit_output=edit_output,
        ) as ret:
            model(**inputs)

        for layer_name in layer_names:
            act = ret[layer_name].output[0].cpu()
            activations[layer_name].append(act)

        generation = model.generate(**inputs, max_new_tokens=max_new_tokens)
        t = inputs.input_ids.shape[1]
        generation = generation[:, t:]
        generations.extend(generation)

    pos = -1  # just the last token
    activations = {
        k: torch.concatenate([vv[:, pos] for vv in v], dim=0).cpu()
        for k, v in activations.items()
    }
    generations = tokenizer.batch_decode(generations, skip_special_tokens=True)

    return activations, generations
def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()
@torch.no_grad()
def direction_ablation_hook(
    output: Float[Tensor, "... d_act"],
    layer: str,
    inputs,
    directions: Dict[str, Float[Tensor, "d_act"]],
):
    ln = read2edit_layer_map[layer]
    direction = directions[ln].to(output.device)
    proj = (
        einops.einsum(
            output, direction.view(-1, 1), "... d_act, d_act single -> ... single"
        )
        * direction
    )
    return output - proj
def get_orthogonalized_matrix(
    matrix: Float[Tensor, "... d_model"], vec: Float[Tensor, "d_model"]
) -> Float[Tensor, "... d_model"]:
    proj = (
        einops.einsum(
            matrix, vec.view(-1, 1), "... d_model, d_model single -> ... single"
        )
        * vec
    )
    return matrix - proj

torch.set_grad_enabled(False)
MODEL_PATH = MODEL_PATH.lower()
verbose = False
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
).eval()
DEVICE = model.device
layers = list(range(2, len(model.model.layers)))
layers_to_read = [f"model.layers.{l}" for l in layers]
perplexity_results = {}
input_texts = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]
input_texts = [s for s in input_texts[:1000] if s!='']

def bruteforce():
    harmful_inst_train, harmful_inst_test = get_harmful_instructions()
    harmless_inst_train, harmless_inst_test = get_harmless_instructions()
    tokenize_instructions_fn = functools.partial(
        tokenize_instructions_chat, tokenizer=tokenizer
    )
    activations, generations = get_generations(
        instructions=harmful_inst_train[: batch_size * 2],
        model=model,
        tokenizer=tokenizer,
        layer_names=layers_to_read,
        tokenize_instructions_fn=tokenize_instructions_fn,
        max_new_tokens=6,
        batch_size=batch_size,
    )
    harmless_cache, harmless_generation = get_generations(
        instructions=harmless_inst_train[:N_INST_TRAIN],
        model=model,
        tokenizer=tokenizer,
        layer_names=layers_to_read,
        tokenize_instructions_fn=tokenize_instructions_fn,
        max_new_tokens=1,
        batch_size=batch_size,
    )
    clear_mem()
    harmful_cache, harmfull_generation = get_generations(
        instructions=harmful_inst_train[:N_INST_TRAIN],
        model=model,
        tokenizer=tokenizer,
        layer_names=layers_to_read,
        tokenize_instructions_fn=tokenize_instructions_fn,
        max_new_tokens=1,
        batch_size=batch_size,
    )
    refusal_directions = {
        ln: harmful_cache[ln].mean(0) - harmless_cache[ln].mean(0) for ln in layers_to_read
    }
    refusal_directions = {k: v / v.norm() for k, v in refusal_directions.items()}
    refusal_directions = {k: v.to(DEVICE) for k, v in refusal_directions.items()}
    read2edit_layer_map = {
        f"model.layers.{l}.self_attn.o_proj": f"model.layers.{l}" for l in layers[1:]
    }
    read2edit_layer_map.update(
        {f"model.layers.{l}.mlp.down_proj": f"model.layers.{l}" for l in layers[1:]}
    )
    layers_to_edit = list(read2edit_layer_map.keys())
    clear_mem()
    edit_output = functools.partial(direction_ablation_hook, directions=refusal_directions)
    clear_mem()
    for key in layers_to_edit:
        m = get_module(model, key)
        ln = read2edit_layer_map[key]
        refusal_dir = refusal_directions[ln].to(m.weight.device)
        if "mlp" in key:
            m.weight.data = get_orthogonalized_matrix(m.weight.T, refusal_dir).T
        else:
            m.weight.data = get_orthogonalized_matrix(m.weight, refusal_dir)
    clear_mem()
    _, orthogonalized_generations = get_generations(
        instructions=harmful_inst_test[:N_INST_TRAIN],
        model=model,
        tokenizer=tokenizer,
        tokenize_instructions_fn=tokenize_instructions_fn,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    failures = 0
    for i in range(N_INST_TEST):
        if any(s in orthogonalized_generations[i] for s in ["with?", "something else", "I cannot", "I can't"]):
            failures += 1

    print(Fore.RED + f"FAILURES: {failures} out of {N_INST_TEST}" + Fore.RESET)

    if(failures > MAX_REFUSALS):
        return bruteforce()
    
    if(failures < MIN_REFUSALS):
        return bruteforce()

    model_name = Path(MODEL_PATH).stem.lower()
    f = f"./{model_name}-ortho-baukit-{failures}fail-{N_INST_TEST}total-bf16-{int(datetime.datetime.utcnow().timestamp())}"
    print(f"saving to {f}")
    model.save_pretrained(f)
    tokenizer.pad_token = "<|eot_id|>"
    tokenizer.pad_token_id = 128009
    tokenizer.set_truncation_and_padding(PaddingStrategy.DO_NOT_PAD, TruncationStrategy.DO_NOT_TRUNCATE, None, 0, None)
    tokenizer.save_pretrained(f)

# keep 20 candidates
for i in range (20):
  bruteforce()
