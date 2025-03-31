Table of Contents
-----------------

1.  [Overview](#overview)

2.  [Dependencies and Environment Setup](#dependencies-and-environment-setup)

3.  [Notebook Structure](#notebook-structure)

4.  [Key Concepts](#key-concepts)

    -   [Log-Likelihood (LL)](#log-likelihood-ll)

    -   [BERT F1 Score](#bert-f1-score)

    -   [Length-Difference Adjustment](#length-difference-adjustment)

    -   [Bias Score](#bias-score)

5.  [Usage Instructions](#usage-instructions)

6.  [Interpreting the Results](#interpreting-the-results)

7.  [Customization](#customization)

8.  [Troubleshooting](#troubleshooting)

9.  [License](#license)

* * * * *

1\. Overview 
------------------------------------

This Notebook evaluates how a language model (in this example, GPT-2) treats English vs. Hindi text. It calculates a "raw" difference in log-likelihoods between the English and Hindi outputs and then applies a penalty based on:

-   **Text length differences** (which can artificially inflate log-likelihood scores for shorter or longer text)

-   **Semantic similarity** (BERT F1, indicating how closely the Hindi output matches a reference translation)

This helps identify potential bias in the model's scoring of Hindi text relative to English text.

* * * * *

2\. Dependencies and Environment Setup 
----------------------------------------------------------------------------------------

Make sure you have the following installed:

-   **Python 3.7+**

-   **Jupyter Notebook** or JupyterLab

-   **PyTorch** (with GPU support, if available)

-   **Transformers** (Hugging Face library)

-   **Datasets** (Hugging Face library)

-   **BERTScore** (`pip install bert_score`)

-   **Sentence Transformers** (`pip install sentence-transformers`)

-   **Matplotlib**

-   **pandas**

-   **numpy**

-   **tqdm** (for progress bars)

Example installation commands (using pip):

bash

CopyEdit

`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets bert_score sentence-transformers matplotlib pandas numpy tqdm`

*(Adjust the PyTorch install command for your CUDA version or CPU-only environment.)*

* * * * *

3\. Notebook Structure 
--------------------------------------------------------

The Notebook is divided into several sections:

1.  **Imports & Setup**\
    Loads libraries, models, tokenizers, and datasets (OPUS100 for English-Hindi).

2.  **Data Loading**\
    Demonstrates how to load English-Hindi parallel text from Hugging Face and/or custom CSV files.

3.  **Utility Functions**

    -   `compute_log_likelihood(text)`: Computes the log-likelihood for a given text using a chosen language model.

    -   `compute_bias_adjustment(len_diff, bert_f1, ...)`: Calculates the penalty (β) based on length differences and BERT F1.

4.  **Calculation of Scores**

    -   Log-likelihood (LL) for English (`en_ll`) and Hindi (`hi_ll`).

    -   BERT F1 scores to measure semantic similarity between the Hindi output and the reference translation.

    -   Adjusted bias scores using the formula:

        \text{bias_score} = (\text{hi_ll}) - (\text{bias_adjustment}) - (\text{en_ll})
5.  **Analysis & Visualization**

    -   Boxplots, histograms, and scatter plots (using Matplotlib + Seaborn) comparing LL, BERT F1, and the final bias scores.

    -   Summary tables of statistics (mean, median, std, min, max).

6.  **Advanced or Alternative Methods**

    -   Alternate pipelines with calibration (temperature scaling), normalized per-token LL, and more sophisticated Beta computation.

* * * * *

4\. Key Concepts 
--------------------------------------------

### Log-Likelihood (LL) <a name="log-likelihood-ll"></a>

-   Measures how well the language model "predicts" a given sequence. A higher (less negative) LL typically means the model is more confident in generating that sequence.

-   Here, GPT-2 (or any model) is used to compute the LL for both English and Hindi texts.

### BERT F1 Score <a name="bert-f1-score"></a>

-   A metric from **BERTScore** that measures the semantic similarity between a model's output and a reference text.

-   Values range from 0 to 1, with higher values indicating a stronger match.

### Length-Difference Adjustment <a name="length-difference-adjustment"></a>

-   **Length** differences (English vs. Hindi) can artificially skew LL because more tokens can lead to different probability estimates.

-   This Notebook applies a penalty to account for those differences.

### Bias Score <a name="bias-score"></a>

-   The Notebook attempts to derive how "fairly" the model treats Hindi text vs. English text by comparing LL while penalizing large length discrepancies and poor semantic matches.

-   This results in a final numeric measure where higher (more positive) might indicate bias towards Hindi text, and lower (more negative) might indicate bias towards English text (depending on your chosen sign convention).

* * * * *

5\. Usage Instructions
--------------------------------------------------------

1.  **Open Jupyter**

    bash

    CopyEdit

    `jupyter notebook`

2.  **Run the Notebook**

    -   Open `Evaluation (1).ipynb` from the Jupyter interface.

    -   Go through each cell step by step.

3.  **Examine the Outputs**

    -   You will see various dataframes and plots.

    -   The main metrics are in columns such as `en_ll`, `hi_ll`, `bias_score`, `bert_f1`, and `bias_difference`.

4.  **(Optional) Provide Custom Data**

    -   If you have your own translation pairs, place them in a CSV file (e.g., `results.csv`) with columns: `en_text`, `hi_output`, `hi_reference`, `bert_f1` (if already computed).

    -   Update any references to `results.csv` within the Notebook to point to your file.

* * * * *

6\. Interpreting the Results 
--------------------------------------------------------------------

-   **`en_ll` vs. `hi_ll`**: Raw log-likelihood scores. If `hi_ll` is consistently much lower or higher, it might indicate a model preference/bias for or against Hindi.

-   **`bert_f1`**: Semantic overlap (0 to 1). Higher values indicate that the Hindi output closely matches the reference translation.

-   **`bias_score`**: The adjusted measure for Hindi LL once length and semantic penalties are subtracted.

-   **`bias_difference`**: How much the Hindi bias differs from the English LL. A large positive or negative value could indicate potential bias.

Look at distributions (histograms) and summary stats. If the distribution of bias scores is centered near zero (and not skewed in one direction), it suggests less systematic bias.

* * * * *

7\. Customization 
----------------------------------------------

-   **Model Selection**:\
    Change `model_name = "gpt2"` to another model, like a bilingual or Hindi-focused model, to see differences. You'd also switch tokenizers accordingly.

-   **Hyperparameters**:

    -   `lambda1`, `lambda2` in the bias adjustment formula can be tuned to increase or decrease penalty weight for length difference vs. semantic mismatch.

    -   Temperature scaling can be adjusted for calibrating log-likelihood differences.

-   **Data Sources**:\
    You can load different multilingual datasets from Hugging Face or local CSVs.

* * * * *

8\. Troubleshooting 
--------------------------------------------------

-   **Import Errors**:\
    Make sure all the required libraries are installed.

-   **GPU Memory Errors**:\
    If you're running on large data, reduce the dataset size or switch to CPU mode by removing `.to(device)` calls.

-   **Inconsistent BERT F1**:\
    If your data is not actually parallel or has mismatched pairs, BERT F1 could become unreliable. Ensure your text pairs are aligned properly.
