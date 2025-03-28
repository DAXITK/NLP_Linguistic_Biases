import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
import nltk

# Download necessary NLTK data
nltk.download('punkt')

# Cache the BLEU metric so it's loaded only once
bleu_metric = evaluate.load("bleu")

def load_model_and_tokenizer(model_name="gpt2"):
    """Load GPT model and tokenizer with FP16 support if GPU is available."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device

def compute_bleu(reference, candidate):
    """
    Compute BLEU score between reference and candidate texts using NLTK's sentence_bleu 
    with smoothing.
    """
    smoothie = SmoothingFunction().method1
    score = sentence_bleu([reference.split()], candidate.split(), smoothing_function=smoothie)
    return score


def compute_log_likelihood(text, model, tokenizer, device):
    """Computes log likelihood for a given text using the model."""
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return -loss.item() * input_ids.size(1)

def compute_bias_factor(text, tokenizer):
    """Calculates a simple tokenization error rate as a bias adjustment factor."""
    tokens = tokenizer.tokenize(text)
    if not tokens:
        return 0.0
    # Count tokens that do not start with the space marker (assuming GPT's tokenizer uses "Ġ")
    split_errors = sum(1 for token in tokens[1:] if not token.startswith("Ġ"))
    return split_errors / len(tokens)

def batch_log_likelihood(texts, model, tokenizer, device, batch_size=8):
    """Compute log-likelihood for a batch of texts."""
    ll_list = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Batch Log-Likelihood"):
        batch = texts[i:i+batch_size]
        # Tokenize batch with padding and truncation (adjust max_length as needed)
        inputs = tokenizer(
            batch, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        # Calculate per-sequence loss (adjust for padding)
        losses = -outputs.loss * torch.sum(inputs["attention_mask"], dim=1)
        ll_list.extend(losses.tolist())
    return ll_list

def batch_generate(prompts, model, tokenizer, device, batch_size=8, max_new_tokens=50):
    """Generate outputs for a batch of prompts."""
    outputs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating outputs"):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(
            batch, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return outputs

def batch_bertscore(references, hypotheses, lang="hi", batch_size=16):
    """Compute BERTScore in batches."""
    scores = []
    for i in tqdm(range(0, len(references), batch_size), desc="BERTScore"):
        batch_refs = references[i:i+batch_size]
        batch_hyps = hypotheses[i:i+batch_size]
        _, _, F1 = bert_score(batch_hyps, batch_refs, lang=lang)
        scores.extend(F1.tolist())
    return scores
