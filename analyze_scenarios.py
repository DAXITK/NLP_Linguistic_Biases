import json
import pandas as pd
from tqdm import tqdm
from utils import (
    load_model_and_tokenizer,
    batch_log_likelihood,
    batch_generate,
    batch_bertscore,
    compute_bias_factor
)

def analyze_scenarios(scenarios, model, tokenizer, device, batch_size=8):
    results = []
    
    # Define mapping from English category names to Hindi keys
    mapping = {
        "Hotel": "होटल",
        "Restaurant": "रेस्तरां"
    }
    
    # Build lists of English texts and aligned Hindi references by iterating over each category and level
    en_texts = []
    hi_refs = []
    for category in scenarios["English"]:
        hindi_category = mapping.get(category)
        if hindi_category is None:
            raise KeyError(f"Mapping for category {category} not found in Hindi scenarios.")
        for level in scenarios["English"][category]:
            texts = scenarios["English"][category][level]
            references = scenarios["Hindi"][hindi_category][level]
            if len(texts) != len(references):
                raise ValueError(f"Mismatch in number of texts and references for category {category} level {level}")
            en_texts.extend(texts)
            hi_refs.extend(references)
    
    # Generate Hindi outputs from English texts
    hi_outputs = batch_generate(en_texts, model, tokenizer, device, batch_size)
    
    # Batch compute all metrics (log-likelihood and BERTScore)
    en_lls = batch_log_likelihood(en_texts, model, tokenizer, device, batch_size)
    hi_lls = batch_log_likelihood(hi_refs, model, tokenizer, device, batch_size)
    bert_scores = batch_bertscore(hi_refs, hi_outputs, lang="hi", batch_size=batch_size)
    
    # Compile results with progress tracking
    for idx, (en_text, hi_output, hi_ref) in enumerate(zip(en_texts, hi_outputs, hi_refs)):
        results.append({
            "en_text": en_text,
            "hi_output": hi_output,
            "hi_reference": hi_ref,
            "en_ll": en_lls[idx],
            "hi_ll": hi_lls[idx],
            "bert_f1": bert_scores[idx],
            "bias_score": hi_lls[idx] - en_lls[idx]  # Example bias score: difference in log-likelihoods
        })
    
    return pd.DataFrame(results)

def main():
    model, tokenizer, device = load_model_and_tokenizer("gpt2")
    with open("/content/NLP_Linguistic_Biases/scenarios.json", "r", encoding="utf-8") as f:
        scenarios = json.load(f)
    
    results = analyze_scenarios(scenarios, model, tokenizer, device, batch_size=8)
    results.to_csv("results.csv", index=False)
    print(f"Saved results for {len(results)} samples")

if __name__ == "__main__":
    main()
