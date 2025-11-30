import pandas as pd
from datasets import load_dataset, DownloadConfig

# Mapping hypotheses → broader SDoH categories
HYPOTHESIS_TO_CATEGORY = {
    # Employment / Occupation
    "The person is employed.": "employment",
    "The person is employed part time.": "employment",
    "The person is a homemaker.": "employment",
    "The person is not employed.": "employment",
    "The person is retired due to age or preference.": "employment",
    "The person is retired due to disability.": "employment",
    "The person is retired due to an unknown reason.": "employment",
    "The person is a student.": "employment",

    # Housing
    "The person lives in their own or their family's home.": "housing",
    "The person lives in a housing facility.": "housing",

    # Transportation
    "The person has access to transportation.": "transportation",

    # Food security
    "The person is able to obtain food on a consistent basis.": "food",

    # Smoking / Tobacco
    "The person is currently a smoker.": "smoking",
    "The person was a smoker in the past.": "smoking",
    "The person wasn't a smoker in the past.": "smoking",
    "The person is currently not a smoker.": "smoking",

    # Alcohol use
    "The person currently drinks alcohol.": "alcohol",
    "The person drank alcohol in the past.": "alcohol",
    "The person currently does not drink alcohol.": "alcohol",
    "The person did not drink alcohol in the past.": "alcohol",

    # Drug use: Opioids
    "The person uses opioids.": "opioids",
    "The person used opioids in the past.": "opioids",
    "The person did not use opioids in the past.": "opioids",
    "The person does not use opioids.": "opioids",

    # Drug use: Marijuana
    "The person uses marijuana.": "marijuana",
    "The person used marijuana in the past.": "marijuana",
    "The person did not use marijuana in the past.": "marijuana",
    "The person does not use marijuana.": "marijuana",

    # Drug use: Cocaine
    "The person uses cocaine.": "cocaine",
    "The person used cocaine in the past.": "cocaine",
    "The person did not use cocaine in the past.": "cocaine",
    "The person does not use cocaine.": "cocaine",

    # General Drug use
    "The person is a drug user.": "drug_use",
    "The person was a drug user in the past.": "drug_use",
    "The person wasn't a drug user in the past.": "drug_use",
    "The person is not a drug user.": "drug_use",
}

def load_sdoh_dataset():
    """
    Load SDOH-NLI dataset, pivot, and collapse into 10-category multi-label dataframe.
    Returns: pandas.DataFrame with columns: premise + 10 categories
    """
    download_config = DownloadConfig(proxies=None, user_agent="my-user-agent")
    ds = load_dataset("tasksource/SDOH-NLI", download_config=download_config)
    df = ds['train'].to_pandas()

    # Pivot to wide format
    pivot_df = df.pivot_table(
        index="premise",
        columns="hypothesis",
        values="label",
        aggfunc="first"
    ).reset_index()

    # Collapse to categories
    final_df = pd.DataFrame()
    final_df["premise"] = pivot_df["premise"]

    categories = set(HYPOTHESIS_TO_CATEGORY.values())
    for cat in categories:
        cols = [h for h, c in HYPOTHESIS_TO_CATEGORY.items() if c == cat and h in pivot_df.columns]
        final_df[cat] = pivot_df[cols].max(axis=1)

    return final_df
if __name__ == "__main__":
    df = load_sdoh_dataset()
    print(df.head())
    print("Final dataset shape:", df.shape) # Expect (n_samples, 11)