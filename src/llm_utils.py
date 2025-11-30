import json
import ast
import numpy as np
import re
import time
import os
from openai import AzureOpenAI
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score

def parse_llm_labels(x):
    """
    Normalize GPT / human labels into a clean Python list of strings.
    Handles messy cases: None, JSON, Python lists, dicts, bad formatting.
    """
    # 1) None / NaN
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []

    # 2) Already a Python list
    if isinstance(x, list):
        return [str(i).strip().lower() for i in x if i]

    # 3) Dict with list inside
    if isinstance(x, dict):
        for k, v in x.items():
            if isinstance(v, list):
                return [str(i).strip().lower() for i in v if i]
        return []

    # 4) String: try JSON, ast.literal_eval
    if isinstance(x, str):
        s = x.strip()

        # 4a) JSON
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(i).strip().lower() for i in parsed if i]
            if isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        return [str(i).strip().lower() for i in v if i]
        except Exception:
            pass

        # 4b) ast.literal_eval
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(i).strip().lower() for i in parsed if i]
        except Exception:
            pass

    # 5) Fallback: nothing usable
    return []

# The prompt template is now moved inside get_prompt_template() to avoid duplication
def get_prompt_template():
    return """
    You are an expert annotator for clinical text.

    Task:
    Given a patient note (premise), identify which of the following categories apply:
    {categories}

    Rules:
    0) Identify all applicable categories from the list above that apply to the patient note.
    Return ONLY a JSON array of applicable categories, with NO explanation.
    1) UMBRELLA DRUG RULE: If opioids, marijuana, or cocaine are present, ALSO include "drug_use".
    Example: "Admits heroin use" -> ["opioids", "drug_use"]
    2) EMPLOYMENT:
    - Label "employment" ONLY if the premise asserts unemployment, job loss, or work-related problems.
    - Do NOT label if the person is working, retired, a student, or a homemaker. 
    3) HOUSING:
    - Label only if unstable (homeless, shelter, unsafe housing).
    - Do NOT label for pets, family members, or neutral mentions of home/living situation.
    4) SUBSTANCE USE:
    - Include category if use is stated (current or past).
    - If "quit", treat as past use and still include category.
        (e.g."She had quit smoking cigarettes some 25 years ago and is a nondrinker." ->['smoking'])
    - Negations cancel it (e.g., "denies alcohol use" -> []).
    5) FOOD & TRANSPORTATION:
    - Label only if lack/barrier exists ("no access to food", "no transportation").
    - Do NOT label if they have food or transportation.
    6) NEGATIONS:
    - If "never", "denies", "none", "no history" or any negative words or phrases are stated -> do NOT include that category.
        Example: "No history of alcohole and drug use" -> []
    7) AMBIGUITY:
    - If speculative ("may use", "possibly unemployed") -> return [].

    {examples}
    Premise: "{text}"
    Answer:
    """
def run_llm_annotation(df, categories, examples, deployment_name, sleep_time=1):
    
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    # Get the prompt template
    prompt_template = get_prompt_template()
    annotations = []
    for text in df["premise"]:
        filled_prompt = prompt_template.format(
            categories=", ".join(categories),
            examples=examples,
            text=text
        )
        try:
            resp = client.chat.completions.create(
                model=deployment_name,
                messages=[{"role": "user", "content": filled_prompt}],
                temperature=0
            )
            raw_output = resp.choices[0].message.content.strip()
            match = re.search(r"\[.*\]", raw_output, re.S)
            if match:
                raw_output = match.group(0)
            annotations.append(raw_output)
        except Exception as e:
            print(f"Error on: {text[:50]}... -> {e}")
            annotations.append("[]")
        time.sleep(sleep_time)
    return annotations
     
def convert_gold_labels_to_list(gold_label_value):
    """
    Convert your gold label format to a clean list.
    Handles: "(smoking,)", "()", "(smoking, alcohol)", etc.
    
    Args:
        gold_label_value: Your gold label (tuple, string, or list)
    
    Returns:
        List of category strings, e.g., ["smoking", "alcohol"]
    """
    # If it's already a tuple or list
    if isinstance(gold_label_value, (tuple, list)):
        return [str(cat).strip() for cat in gold_label_value if cat]
    
    # If it's a string like "(smoking,)" or "(smoking, alcohol)"
    if isinstance(gold_label_value, str):
        try:
            # Use eval to convert string to tuple
            parsed = eval(gold_label_value)
            if isinstance(parsed, (tuple, list)):
                return [str(cat).strip() for cat in parsed if cat]
        except:
            pass
    
    # Empty case
    return []

def calculate_disagreement_score(gold_labels, predicted_labels, categories):
    """
    Calculate how many categories the model got wrong.
    
    Args:
        gold_labels: List like ["smoking", "alcohol"]
        predicted_labels: List like ["smoking", "employment"]
        categories: All 10 categories
    
    Returns:
        disagreement_score: Number of wrong categories (0 to 10)
        per_category_errors: Dict showing which categories were wrong
    """
    # Convert to binary vectors for comparison
    gold_vec = [1 if cat in gold_labels else 0 for cat in categories]
    pred_vec = [1 if cat in predicted_labels else 0 for cat in categories]
    
    # Count disagreements
    disagreement_score = sum(g != p for g, p in zip(gold_vec, pred_vec))
    
    # Track which categories had errors
    per_category_errors = {}
    for i, cat in enumerate(categories):
        if gold_vec[i] != pred_vec[i]:
            if gold_vec[i] == 1 and pred_vec[i] == 0:
                per_category_errors[cat] = "FN (missed)"
            else:
                per_category_errors[cat] = "FP (false alarm)"
    
    return disagreement_score, per_category_errors

def run_zero_shot_and_score(df, categories, deployment_name="gpt-35-turbo", sample_size=10):
    """
    Run 0-shot annotation and calculate disagreement scores.
    
    Args:
        df: DataFrame with 'premise' and 'gold_labels' columns
        categories: List of 10 SDoH categories
        deployment_name: Your Azure deployment name
        sample_size: Number of samples to run (default 100, like the paper)
    
    Returns:
        DataFrame with columns: premise, gold_labels, predicted_labels, 
                                disagreement_score, errors
    """
    # Sample from your data (random)
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42).copy()
    
    print(f"Running 0-shot on {len(sample_df)} samples...")
    
    # Run 0-shot annotation using YOUR existing function
    predictions = run_llm_annotation(
        df=sample_df,
        categories=categories,
        examples="",  # No examples for 0-shot
        deployment_name=deployment_name,
        sleep_time=1
    )
    
    # Parse predictions
    sample_df['predicted_labels_raw'] = predictions
    sample_df['predicted_labels'] = sample_df['predicted_labels_raw'].apply(parse_llm_labels)
    
    # Calculate disagreement scores
    scores = []
    errors = []
    
    for idx, row in sample_df.iterrows():
        gold = parse_llm_labels(row['gold_labels'])
        pred = row['predicted_labels']
        
        score, error_dict = calculate_disagreement_score(gold, pred, categories)
        scores.append(score)
        errors.append(error_dict)
    
    sample_df['disagreement_score'] = scores
    sample_df['errors'] = errors
    
    print(f"\nCompleted! Disagreement score stats:")
    print(sample_df['disagreement_score'].describe())
    
    return sample_df

def select_hard_examples_automatically(results_df, top_n=2):
    """
    Automatically select the hardest examples based on disagreement score.
    
    Args:
        results_df: DataFrame from run_zero_shot_and_score()
        top_n: Number of hard examples to select (default: 2)
    
    Returns:
        List of hard example dictionaries
    """
    # Filter to only errors (disagreement_score > 0)
    errors_df = results_df[results_df['disagreement_score'] > 0].copy()
    
    if len(errors_df) == 0:
        print("No errors found! Model is too good for hard example selection.")
        return []
    
    # Sort by disagreement score (highest first)
    errors_df = errors_df.sort_values('disagreement_score', ascending=False)
    
    # Select top N hardest examples
    hard_examples = []
    for idx, row in errors_df.head(top_n).iterrows():
        hard_examples.append({
            'premise': row['premise'],
            'gold_labels': row['gold_labels'],
            'predicted_labels': row['predicted_labels'],
            'disagreement_score': row['disagreement_score'],
            'errors': row['errors']
        })
    
    return hard_examples

# Auto-Generate Explanations
def generate_explanation_for_hard_example(premise, gold_labels, predicted_labels, 
                                          categories, deployment_name="gpt-35-turbo"):
    """
    Automatically generate explanation for why a prediction was wrong.
    
    Args:
        premise: The text
        gold_labels: Correct labels (list)
        predicted_labels: What model predicted (list)
        categories: All 10 categories
        deployment_name: Your Azure deployment
    
    Returns:
        explanation: String explaining the error and correct reasoning
    """
    from openai import AzureOpenAI
    import os
    
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    # Format labels for prompt
    gold_str = str(gold_labels) if gold_labels else "[]"
    pred_str = str(predicted_labels) if predicted_labels else "[]"
    
    # Create explanation prompt
    explanation_prompt = f"""You are an expert at explaining clinical text annotation errors.

A model incorrectly classified this text:

Text: "{premise}"

Model predicted: {pred_str}
Correct answer: {gold_str}

Your task: Write a clear, concise explanation (2-3 sentences) that:
1. Identifies WHY the model was wrong
2. Points out the key phrase or context that determines the correct answer
3. Provides the rule the model should follow

Focus on teaching the model to avoid this mistake in the future.

Explanation:"""
    
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": explanation_prompt}],
            temperature=0.3,  # Slightly creative but consistent
            max_tokens=150
        )
        
        explanation = response.choices[0].message.content.strip()
        return explanation
        
    except Exception as e:
        print(f"Error generating explanation: {e}")
        return f"The correct label is {gold_str} because the text clearly indicates this."
    
def create_2shot_hard_prompt(hard_examples, categories, premise_to_classify):
    """
    Create a 2-shot prompt with hard examples and explanations.
    
    Args:
        hard_examples: List of 2 hard example dicts (with explanations)
        categories: Your 10 categories
        premise_to_classify: New text to classify
    
    Returns:
        Full prompt string ready for GPT
    """
    # Start with your base prompt template
    prompt = f"""You are an expert annotator for clinical text.

Task:
Given a patient note (premise), identify which of the following categories apply:
{', '.join(categories)}

Rules:
0) Identify all applicable categories from the list above that apply to the patient note.
Return ONLY a JSON array of applicable categories, with NO explanation.
1) UMBRELLA DRUG RULE: If opioids, marijuana, or cocaine are present, ALSO include "drug_use".
2) EMPLOYMENT: Label ONLY if unemployed, job loss, or work problems. Do NOT label if working, retired, student, or homemaker.
3) HOUSING: Label only if unstable (homeless, shelter, unsafe). Do NOT label for neutral mentions.
4) SUBSTANCE USE: Include if use is stated (current or past). Negations cancel it.
5) FOOD & TRANSPORTATION: Label only if lack/barrier exists.
6) NEGATIONS: If "never", "denies", "none", "no history" or any negative words → do NOT include that category.
7) AMBIGUITY: If speculative ("may use", "possibly") → return [].

Here are two challenging examples to guide you:

Example 1:
Premise: "{hard_examples[0]['premise']}"
Answer: {parse_llm_labels(hard_examples[0]['gold_labels'])}
Explanation: {hard_examples[0]['explanation']}

Example 2:
Premise: "{hard_examples[1]['premise']}"
Answer: {parse_llm_labels(hard_examples[1]['gold_labels'])}
Explanation: {hard_examples[1]['explanation']}

Now classify this text:
Premise: "{premise_to_classify}"
Answer:"""
    
    return prompt    

# Select easy examples automatically
def select_easy_examples_automatically(results_df):
    """
    Select easy examples: 1 positive (TP) + 1 negative (TN)
    where model got it right (disagreement_score = 0)
    """
    # Filter perfect predictions
    perfect_df = results_df[results_df['disagreement_score'] == 0].copy()
    
    # Separate into positive and negative
    positive_df = perfect_df[perfect_df['gold_labels'].apply(
        lambda x: len(parse_llm_labels(x)) > 0
    )]
    negative_df = perfect_df[perfect_df['gold_labels'].apply(
        lambda x: len(parse_llm_labels(x)) == 0
    )]
    
    # Select one from each (median length for diversity)
    if len(positive_df) > 0:
        pos_lengths = positive_df['premise'].str.len()
        pos_idx = (pos_lengths - pos_lengths.median()).abs().idxmin()
        easy_positive = positive_df.loc[pos_idx].to_dict()
    else:
        easy_positive = None
    
    if len(negative_df) > 0:
        neg_lengths = negative_df['premise'].str.len()
        neg_idx = (neg_lengths - neg_lengths.median()).abs().idxmin()
        easy_negative = negative_df.loc[neg_idx].to_dict()
    else:
        easy_negative = None
    
    return easy_positive, easy_negative

# Generate explanations for easy examples
def generate_explanation_for_easy_example(premise, gold_labels, predicted_labels, 
                                          categories, deployment_name="gpt-35-turbo"):
    """
    Automatically generate explanation for why a prediction was CORRECT.
    This is for easy examples where model got it right.
    
    Args:
        premise: The text
        gold_labels: Correct labels (list)
        predicted_labels: What model predicted (list) - should match gold!
        categories: All 10 categories
        deployment_name: Your Azure deployment
    
    Returns:
        explanation: String explaining why this is correct
    """
    from openai import AzureOpenAI
    import os
    
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    # Format labels for prompt
    label_str = str(gold_labels) if gold_labels else "[]"
    
    # Create explanation prompt
    explanation_prompt = f"""You are an expert at explaining clinical text annotations.

A model correctly classified this text:

Text: "{premise}"

Correct answer: {label_str}

Your task: Write a clear, concise explanation (2-3 sentences) that:
1. Identifies the key phrase or context that determines this classification
2. Explains WHY this answer is correct
3. Provides the reasoning a model should use for similar cases

Explanation:"""
    
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": explanation_prompt}],
            temperature=0.3,
            max_tokens=150
        )
        
        explanation = response.choices[0].message.content.strip()
        return explanation
        
    except Exception as e:
        print(f"Error generating explanation: {e}")
        return f"The correct label is {label_str} based on the text content."

def calculate_multilabel_metrics(gold_labels, predicted_labels, categories):
    """
    Calculate comprehensive metrics for multi-label classification.
    
    Args:
        gold_labels: List of lists, e.g., [[], ['smoking'], ['smoking', 'alcohol']]
        predicted_labels: List of lists (same format)
        categories: List of 10 category names
    
    Returns:
        Dictionary with all metrics
    """
    # Convert to binary matrices (using your existing function concept)
    n_samples = len(gold_labels)
    n_categories = len(categories)
    
    # Create binary matrices
    Y_true = np.zeros((n_samples, n_categories), dtype=int)
    Y_pred = np.zeros((n_samples, n_categories), dtype=int)
    
    for i in range(n_samples):
        for j, cat in enumerate(categories):
            if cat in gold_labels[i]:
                Y_true[i, j] = 1
            if cat in predicted_labels[i]:
                Y_pred[i, j] = 1
    
    # Calculate metrics
    metrics = {}
    
    # 1. Micro-averaged metrics (treats all category predictions equally)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        Y_true, Y_pred, average='micro', zero_division=0
    )
    metrics['precision_micro'] = precision_micro
    metrics['recall_micro'] = recall_micro
    metrics['f1_micro'] = f1_micro
    
    # 2. Macro-averaged metrics (averages across categories)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        Y_true, Y_pred, average='macro', zero_division=0
    )
    metrics['precision_macro'] = precision_macro
    metrics['recall_macro'] = recall_macro
    metrics['f1_macro'] = f1_macro
    
    # 3. Exact match accuracy (whole prediction matches gold)
    exact_matches = sum(
        set(gold) == set(pred) 
        for gold, pred in zip(gold_labels, predicted_labels)
    )
    metrics['exact_match_accuracy'] = exact_matches / n_samples
    
    # 4. Hamming loss (fraction of wrong labels)
    hamming_loss = np.mean(Y_true != Y_pred)
    metrics['hamming_loss'] = hamming_loss
    
    # 5. Cohen's Kappa (agreement measure)
    # Flatten for kappa calculation
    Y_true_flat = Y_true.flatten()
    Y_pred_flat = Y_pred.flatten()
    kappa = cohen_kappa_score(Y_true_flat, Y_pred_flat)
    metrics['cohen_kappa'] = kappa
    
    return metrics
def get_strengthened_prompt_template():
    return """
    You are an expert annotator for clinical text.

    Task:
    Given a patient note (premise), identify which of the following categories apply:
    {categories}

    Rules:
    0) Identify all applicable categories from the list above that apply to the patient note.
    Return ONLY a JSON array of applicable categories, with NO explanation.
    
    1) UMBRELLA DRUG RULE:
    - If opioids, marijuana, or cocaine are present, ALSO include "drug_use".
    - CRITICAL: Do NOT add "drug_use" for smoking or alcohol!
    Examples: "heroin" -> ["opioids", "drug_use"], "smokes" -> ["smoking"]
    
    2) EMPLOYMENT:
    - Label ONLY if unemployed, job loss, or work problems.
    - Do NOT label if working, retired, student, or homemaker.
    
    3) HOUSING:
    - Label only if unstable (homeless, shelter, unsafe).
    
    4) SUBSTANCE USE - CRITICAL RULES (READ CAREFULLY):
    a) ANY mention of use = include category, even if:
       - "Occasional" use (e.g., "occasionally drinks" -> ["alcohol"])
       - "Infrequent" use (e.g., "infrequent alcohol" -> ["alcohol"])
       - "Social" use (e.g., "social drinker" -> ["alcohol"])
       - Past use (e.g., "former smoker" -> ["smoking"])
       - "Quit" (e.g., "quit 10 years ago" -> ["smoking"])
    
    b) Medical abbreviations count as use:
       - "40 pack-year tobacco" = smoking history -> ["smoking"]
       - "ETOH" = alcohol -> (only if positive mention)
    
    c) Third-party reports COUNT (even if patient denies):
       - "Patient denies but son reports history" -> Include the category
       - Family/doctor observations override patient denial
    
    d) Mixed negations: Look carefully for positive mentions:
       - "No alcohol, occasional marijuana, no drugs" -> ["marijuana", "drug_use"]
       - Don't let one negation cancel everything
    
    5) NEGATIONS - BUT READ RULE 4 FIRST:
    - "Never", "denies", "no history" -> do NOT include IF truly absent
    - But occasional/past/quit use still counts (see Rule 4a)
    
    6) FOOD & TRANSPORTATION:
    - Label only if lack/barrier exists.
    
    7) AMBIGUITY:
    - If speculative ("may use") -> return [].

    {examples}
    Premise: "{text}"
    Answer:
    """ 