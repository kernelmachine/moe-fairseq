from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import torch.nn.functional as F
import sys
from evaluate import load
from datasets import load_dataset
import pandas as pd
from tqdm.auto import tqdm
import numpy as np

os.environ['TRANSFORMERS_CACHE']='/gscratch/zlab/sg01/'

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def ppl(path, prompts):
    perplexity = load("perplexity", module_type='metric')
    return perplexity.compute(predictions=prompts, model_id=path,device='gpu')

def log_probs_with_ppl(path, prompt):
    model = AutoModelForCausalLM.from_pretrained(path, cache_dir="/gscratch/zlab/sg01/transformers_cache/")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, cache_dir="/gscratch/zlab/sg01/transformers_cache/")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    logits = outputs.logits
    arg_probs, _ = F.softmax(logits, dim=-1).max(-1)
    print("argmax probility:", arg_probs[0].cpu().detach().numpy())
    log_probs, tokens = F.log_softmax(logits, dim=-1).max(-1)
    print("argmax log probability:", log_probs[0].cpu().detach().numpy())
    sent = tokenizer.decode(tokens.squeeze().cpu().detach().numpy(), skip_special_tokens=False)
    print("argmax tokens:", sent)
    xentropy_loss = outputs[0]
    print("cross entropy loss:", xentropy_loss.item())
    ppl = torch.exp(xentropy_loss).item()
    print("ppl:", ppl)


if __name__ == "__main__":

#    input_texts = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"][:10] # doctest: +SKIP
 #   input_texts = [[s for s in input_texts if s!=''][0]]
   # import pdb; pdb.set_trace()
    os.environ['HF_HOME'] = '/gscratch/zlab/sg01'
    # input_texts = ["In its most general sense, the term 'world' refers to the totality of entities, to the whole of reality or to everything that is."]
    df = pd.read_json('/gscratch/zlab/sg01/data/c4/valid_c4_small/C4_small/00000/C4.jsonl', lines=True)
    input_texts = df.text.head(200).tolist()

    batches = batch(input_texts, n=4)

    for model_id in ["opt-350m"]:
        ppls = []
        print(20 * "=" + model_id + 20 * "=")
        model_path = os.path.join("facebook", model_id)
        pbar = tqdm(batches)
        for batch in pbar:
            import pdb; pdb.set_trace()
            ppls.append(ppl(model_path, batch)['mean_perplexity'])
            pbar.set_description(f"ppl: {np.mean(ppls)}")
