# Automated SDoH Classification with LLMs
Extracting Social Determinants of Health from Clinical Text using Few-Shot Learning: 
Using few-shot learning for multi-label classification across 10 SDoH categories with class imbalance on the data source--SDoH-NIL(HuggingFace)
This project focuses on classifying Social Determinants of Health (SDoH) using Large Language Models (LLMs). It is organized as a series of JupyterNotebooks for step-by-step execution rather than a single entry point via main.py.

**🎯 The Problem**
Social Determinants of Health (SDoH)—factors like housing stability, substance use, and employment status—profoundly impact patient outcomes. However, this critical information is often buried in unstructured clinical notes, making it difficult for healthcare systems to identify at-risk populations and intervene early.

**The Challenge**: Extract 10 different SDoH categories from clinical text with high accuracy, despite:
1. Severe class imbalance (some categories appear in <5% of samples)
2. Limited labeled training data (570 samples)
3. Nuanced clinical language requiring domain understanding

**🚀 Our Approach**
Rather than immediately jumping to traditional ML methods, we developed a systematic pipeline combining automated example selection with state-of-the-art LLMs:
*Phase 1: Automated Few-Shot Learning*

1. Hard Example Mining: Automatically identified the 2 most challenging cases by measuring disagreement between GPT-3.5 predictions and gold labels
2. Balanced Example Selection: Added 1 positive (multi-label) and 1 negative (no labels) example for robust 2-shot learning
3. Prompt Engineering: Tested 5 configurations (0-shot, 2-shot easy, 2-shot with explanations, 2-shot hard)
4. Model Upgrade: Strengthened prompt with critical rules for substance use classification and deployed GPT-4o-mini

*Phase 2: Traditional ML Exploration*
Attempted to distill LLM knowledge into a lightweight XGBoost model for cost reduction and faster inference.

### Final Performance

| Model | F1 Score | Notes |
|-------|----------|-------|
| **GPT-4o-mini (direct)** | **0.940** | ✅ **Winner** - Only 10 errors on 200 test samples |
| GPT-3.5 Turbo (baseline) | 0.845 | 2-shot easy configuration |
| XGBoost (320 noisy labels) | 0.564 | Failed - insufficient data |
| XGBoost (150 gold labels) | 0.480 | Failed - worse with less data |
| Confidence-based Hybrid | 0.659 | Failed - couldn't compensate |

**Key Finding:** Direct LLM inference substantially outperforms model distillation when working with limited training data.
**Per-Category Performance (GPT-4o-mini)**
The model achieves strong performance across all 10 categories:
- Smoking: F1 = 0.95
- Alcohol: F1 = 0.93
- Drug Use: F1 = 0.94
- Employment: F1 = 0.91
- Housing: F1 = 0.96
- Food Insecurity: F1 = 0.97
- Transportation: F1 = 0.94
- Opioids: F1 = 0.95
- Cocaine: F1 = 0.93
- Marijuana: F1 = 0.92

**🧪 What We Learned (The XGBoost Story)**
We rigorously tested whether traditional ML could match LLM performance at lower cost. It couldn't—and here's why:
**Experiment 1: Train on LLM-Generated Labels**
- Used GPT-4o-mini to label 320 additional samples
- Extracted TF-IDF features (1000 max features, 1-2 ngrams)
- Trained XGBoost MultiOutputClassifier
- Result: F1 = 0.564 ❌

**Problem:** The LLM-generated training labels had ~10% error rate, introducing noise that degraded XGBoost performance.
**Experiment 2: Train on Clean Gold Labels**

- Used 150 gold-labeled samples for training
- Eliminated label noise entirely
- Result: F1 = 0.480 ❌ (Even worse!)

**Problem:** Insufficient training data for 10 categories, especially rare ones (6 categories had <5 training examples).
**Experiment 3: Hybrid Approaches**
Tried combining XGBoost for high-confidence predictions with GPT-4o-mini fallback:

- Simple hybrid: F1 = 0.667 ❌
- Confidence-based hybrid: F1 = 0.659 ❌

**Problem:** XGBoost only showed confidence >0.5 on 24-27 samples for 2/10 categories. For most categories, max confidence was <0.1.
**Root Cause Analysis**
Comparing to successful LLM→XGBoost distillation papers (e.g., arXiv 2407.17126):

- Their setup: 1000s of training samples, better class balance, cleaner labels
- Our setup: 320 samples, 6 categories with <5 examples, 10-12% label noise

**Conclusion:** Our approach was methodologically sound; XGBoost simply requires substantially more training data (likely 1000+ samples per category) to compete with LLMs on this task.

## 🛠️ Technical Stack
- LLMs: Azure OpenAI (GPT-3.5-turbo, GPT-4o-mini)
- ML Framework: scikit-learn, XGBoost
- NLP: TF-IDF vectorization
- Data Source: SDoH-NIL dataset (Hugging Face)
- Languages: Python 3.9+
## 📁 Project Structure
```
sdoh-llm-classification/
├── README.md
├── requirements.txt
├── src/
│   ├── sdoh_data.py          # Data loading and preprocessing
│   ├── llm_util.py            # LLM inference utilities
│   └── evaluation.py          # Metrics and analysis
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_sample_selection.ipynb      # Automated hard/easy example selection
│   ├── 03_prompt_engineering.ipynb    # Prompt optimization experiments
│   └── 04_XGBoostImplementation.ipynb # Traditional ML attempts
└── results/
    ├── metrics/
    └── visualizations/
```
## Getting Started
1. Environment Setup
        git clone https://github.com/yaolsenarch/SDoH-LLMClassification.git
        cd SDoH-LLMClassification
        - Create virtual environment (Python 3.9+ recommended)
        python -m venv .venv
        source .venv/bin/activate  # On Windows: .venv\Scripts\activate
        - Install dependencies
        pip install -r requirements.txt
2. Azure OpenAI Configuration
   Set environment variables for Azure OpenAI access:
        export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
        export AZURE_OPENAI_API_KEY="your-api-key"
        export AZURE_OPENAI_DEPLOYMENT="your-deployment-name"
        export AZURE_OPENAI_API_VERSION="your-api-version"
3. SSL Certificate Setup (Corporate Environments)
   If working behind a corporate firewall:
        export REQUESTS_CA_BUNDLE=/path/to/cacert.pem
        export SSL_CERT_FILE=/path/to/cacert.pem
        export HF_HUB_CA_CERTS=/path/to/cacert.pem
4. Run Notebooks Sequentially
   Execute notebooks in order:
        01_data_exploration.ipynb - Understand the dataset and class distribution
        02_sample_selection.ipynb - Automated few-shot example selection
        03_prompt_engineering.ipynb - Test prompt variations and select winner
        04_XGBoostImplementation.ipynb - Explore traditional ML approaches

## 💡 Key Insights
        - Automated Example Selection Works: No manual curation needed—algorithmic selection of hard examples outperformed hand-picked examples
        - LLMs Excel with Limited Data: When training data is scarce (<500 samples), direct LLM inference beats distillation
        - Label Quality Matters More Than Quantity: 150 clean labels < 320 noisy labels for XGBoost, but both insufficient
        - Confidence Scores Reveal Model Limitations: XGBoost's low confidence on 8/10 categories signaled fundamental learning failure
        - Document Failures: Rigorous documentation of failed experiments demonstrates scientific maturity and saves others time
## 📈 Future Work        
        - Scale Data Collection: Aim for 1000+ labeled samples per category to enable effective XGBoost distillation
        - Active Learning: Use LLM uncertainty to prioritize manual labeling efforts
        - Cloud Deployment: Migrate to Azure Functions for production serving
        - Cost Optimization: Explore smaller models (GPT-3.5) once sufficient training data enables fine-tuning
## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
📧 Contact

Author: Yuan-Yuan Olsen
Email: yuanyuan.a.olsen@healthpartners.com 
Project Link: https://github.com/yaolsenarch/SDoH-LLMClassification