# JUDGEBERT-X: Explainable Multilingual Legal Meaning Preservation Using JUDGEBERT

**JUDGEBERT-X** is an explainable cross-lingual Natural Language Processing (NLP) framework designed to evaluate Legal Meaning Preservation (LMP) between English legal clauses and their Hindi translations. 

Evaluating legal translation differs fundamentally from general machine translation evaluation. Generic semantic similarity metrics (such as BLEU, ROUGE, or unconstrained cosine similarity) often report high alignment when most vocabulary overlaps, even if critical legal operatives—such as liability direction, statutory conditions, or exception carve-outs—have been reversed or stripped. JUDGEBERT-X bridges this gap by combining multilingual dense representations with a deterministic neuro-symbolic legal invariant engine.

---

## 1. Research Motivation

In legal contracts (such as residential lease agreements) and statutory policies (such as health and motor insurance), small linguistic perturbations cause complete reversals of enforceability:

- **Legal Obligations vs. Discretionary Permissions:** Confusing mandatory duties (*shall*, *must*, *करना होगा*) with permissive authorizations (*may*, *can*, *सकता है*).
- **Prohibitions and Negations:** Reversing negative restrictions (*shall not sublet* vs. *उप-किराये पर दे सकता है*).
- **Parties and Role Inversion:** Swapping the obligor and obligee (*Landlord* $\leftrightarrow$ *Tenant*), reversing who owes payment or liability.
- **Amounts, Rates, and Timelines:** Subtle numerical shifts (*Rs. 5,000* $\rightarrow$ *Rs. 50,000* or *7 days* $\rightarrow$ *30 days*).
- **Exceptions and Carve-Outs:** Stripping operative exemptions introduced by *unless*, *except for*, or *को छोड़कर*.
- **Conditions and Provisos:** Omitting prerequisite terms introduced by *provided that*, *subject to*, or *बशर्ते*.
- **Coverage Scope Inversion:** Flipping policy coverage from inclusion to exclusion (*including* $\rightarrow$ *को छोड़कर*).

---

## 2. Attribution & Reference Research

This project is an independent research implementation inspired by the conceptual formulation of **JUDGEBERT**:
- **Reference Repository:** [GRAAL-Research/JUDGEBERT](https://github.com/GRAAL-Research/JUDGEBERT)
- **Scope & Lineage:** While the original JUDGEBERT established transformer regression foundations for monolingual legal evaluation, **JUDGEBERT-X** is an independent extension specifically engineered for **cross-lingual English-to-Hindi legal pairs**, incorporating Devanagari Unicode normalization, cross-lingual bilingual projection, and bilingual legal invariant verification.

---

## 3. System Architecture & Pipeline

JUDGEBERT-X executes a 9-stage evaluation pipeline on every English-Hindi pair:

```
[1. English Legal Sentence]       [2. Hindi Candidate Translation]
              │                                   │
              ▼                                   ▼
[Devanagari Normalization & Unicode Preprocessing (NFC, Danda, Hindi Numerals)]
              │                                   │
              ├───────────────────────────────────┤
              ▼                                   ▼
 [3. Lexical Baseline]               [4. Semantic Bi-Encoder Baseline]
 (Cross-Lingual TF-IDF + Lexicon)    (paraphrase-multilingual-MiniLM-L12-v2)
              │                                   │
              └─────────────────┬─────────────────┘
                                ▼
         [5. Extraction of Legal Invariants Across Languages]
         (Polarity, Modality, Parties, Numbers, Provisos, Exceptions)
                                │
                                ▼
         [6. Neuro-Symbolic Invariant Verification & Discrepancy Detection]
         (Rule-based constraint checks across extracted invariant pairs)
                                │
                                ▼
         [7. Final Predicted LMP Score Calculation (1.0 to 10.0 scale)]
                                │
                                ▼
         [8. Categorical Legal Preservation Decision]
         (PRESERVED, PARTIALLY PRESERVED, NOT PRESERVED)
                                │
                                ▼
         [9. Structured Explainability Report & Error Rationale]
```

---

## 4. Models and Technologies

All components run offline and are verified against standard Python ML libraries:

- **Multilingual Bi-Encoder:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dimensional multilingual embeddings, active fallback).
- **Target Backbone Architecture:** `FacebookAI/xlm-roberta-base` Siamese cross-encoder utilizing $[u; v; |u-v|]$ (2304-dimensional representation with regression head).
- **Lexical Baseline:** Cross-lingual character and word n-gram TF-IDF projector using bilingual legal vocabulary mapping.
- **Deep Learning Framework:** PyTorch (`torch>=2.0.0`).
- **Transformers Library:** Hugging Face `transformers` and `sentence-transformers`.
- **Machine Learning & Analytics:** `scikit-learn`, `pandas`, `numpy`, `scipy`.
- **Text Processing:** Python standard `unicodedata` and regular expressions optimized for Devanagari matras and combining characters.

---

## 5. Repository Structure

```text
nlp_m2/
│
├── data/
│   ├── raw/                           # Original domain datasets (Insurance & Tenancy)
│   │   ├── english_insurance.csv
│   │   ├── english_tenancy.xlsx
│   │   ├── hindi_insurance.json
│   │   └── hindi_tenancy.jsonl
│   └── english_hindi/                 # 160 aligned English-Hindi legal sentence pairs
│       ├── aligned_dataset.csv        # Balanced dataset (80 Tenancy, 80 Insurance)
│       ├── annotated.csv              # Human annotation placeholder (strict zero-fabrication)
│       ├── annotation_template.csv    # Annotation template for legal practitioners
│       └── splits/                    # Leakage-safe partitions (Train: 112, Val: 24, Test: 24)
│
├── models/
│   └── tfidf/
│       └── crosslingual_tfidf.pkl     # Fitted cross-lingual TF-IDF baseline model
│
├── src/
│   ├── __init__.py                    # Package exports
│   ├── preprocessing.py               # Devanagari numeral, danda, and NFC normalization
│   ├── data_loader.py                 # Split loader with clause-overlap assertion checks
│   ├── tfidf_model.py                 # Cross-Lingual TF-IDF Baseline model
│   ├── sentence_transformer_model.py  # Multilingual Sentence Transformer baseline
│   ├── judgebert_model.py             # JUDGEBERT-X [u; v; |u-v|] regression architecture
│   ├── training.py                    # Training pipeline with Zero-Fabrication enforcement
│   ├── evaluation.py                  # Test partition evaluator across baselines
│   ├── prediction.py                  # Integrated inference engine and invariant verifier
│   └── explainability.py              # 10-category legal discrepancy analysis engine
│
├── outputs/
│   ├── predictions/
│   │   ├── english_hindi_test_predictions.csv
│   │   └── prediction_history.csv     # Persistent run log for all terminal predictions
│   └── reports/
│       ├── english_hindi_evaluation.csv# Test benchmark evaluation metrics
│       └── sanity_check_results.csv   # 10 legal invariant sanity check results
│
├── tests/
│   └── test_judgebert_x.py            # Automated test suite (21 unit tests, 100% passing)
│
├── hindi_extension/
│   └── predict.py                     # Convenience entrypoint wrapper
│
├── predict.py                         # Primary Terminal CLI for interactive predictions
├── run_sanity_checks.py               # Automated 10-case legal sanity check suite
├── train.py                           # Training execution CLI
├── evaluate.py                        # Benchmark evaluation CLI
├── run_predict.bat                    # Windows batch launcher for interactive predictions
├── run_tests.bat                      # Windows batch launcher for test execution
├── requirements.txt                   # Dependency specifications
└── README.md                          # Repository documentation
```

---

## 6. Installation & Setup

### Prerequisites
- Python 3.10 to 3.12 (or 3.14) installed on your system.
- Windows PowerShell or Bash terminal.

### Step-by-Step Setup (PowerShell)

1. **Clone the repository:**
   ```powershell
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Create and activate a virtual environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 7. Usage Instructions

### Running Predictions from the Terminal

Run the inference CLI on any English original clause and Hindi candidate translation:

```powershell
python predict.py --original "The tenant shall not sublet the premises." --candidate "किरायेदार परिसर को उप-किराये पर नहीं देगा।"
```

### Understanding the Output Fields

The evaluation prints a standardized prediction block:

```text
=====================================================================================
 JUDGEBERT LEGAL MEANING PRESERVATION (LMP) EVALUATION 
=====================================================================================
Original Text        : The tenant shall not sublet the premises.
Candidate (Hindi)    : किरायेदार परिसर को उप-किराये पर नहीं देगा।
-------------------------------------------------------------------------------------
TF IDF ,COSINE SIMILARITY  LMP    : 0.8328  (Topic / Lexical Semantic Overlap)
SENTENCE BERT LMP  : 6.99 / 10.0
FINAL PREDICTED LMP  : 6.99 / 10.0
Preservation Decision: [ PRESERVED - Meaning and Legal Invariants Maintained ]
Legal Error Type     : none
Legal Rationale      : English and Hindi legal meaning, obligations, and invariants match.
Invariant Checks     : Negation: [OK] Preserved | Dates/Numbers: [OK] Preserved
=====================================================================================
```

1. **TF-IDF, Cosine Similarity LMP:** Raw lexical alignment and semantic vector overlap ($[0, 1]$ cosine score).
2. **Sentence-BERT LMP:** Calibrated dense semantic alignment score on the $[1.0, 10.0]$ scale.
3. **Final Predicted LMP:** The final score after adjusting for legal invariant violations and penalties.
4. **Preservation Decision:** Categorical classification:
   - `PRESERVED - Meaning and Legal Invariants Maintained` ($\ge 6.50$)
   - `PARTIALLY PRESERVED - Minor Semantic Divergence` ($4.50 - 6.49$)
   - `NOT PRESERVED - Significant Legal Discrepancy / Invariant Violation` ($< 4.50$)
5. **Legal Error Type:** Specific discrepancy category detected (e.g., `negation_flip`, `modality_change`, `condition_omission`, `party_role_reversal`, `coverage_reversal`).
6. **Legal Rationale:** Plain-language explanation of the legal alignment or divergence.
7. **Invariant Checks:** Granular statuses across Negation, Numbers, Parties, Modality, Conditions, and Exceptions.

---

## 8. Benchmark Evaluation & Sanity Checks

The repository includes a dedicated benchmark suite evaluating 10 legal boundary conditions ([run_sanity_checks.py](run_sanity_checks.py)):

| Test ID | Legal Scenario | English Clause | Hindi Clause | Expected | Invariant Status | Final LMP | Result |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **SAN-01** | Faithful Translation | Tenant shall pay rent Rs 25,000 on 5th | किरायेदार 5वीं तारीख को 25,000 देगा | `PRESERVED` | `[OK] Preserved` | **`7.77`** | `PASSED` |
| **SAN-02** | Unrelated Sentences | Insurer war exclusion clause | किरायेदार पालतू जानवर रख सकता है | `NOT PRESERVED` | `[VIOLATION] Altered` | **`1.16`** | `PASSED` |
| **SAN-03** | Modality Degradation | Insured must undergo medical exam | बीमित व्यक्ति जांच करवा सकता है | `NOT PRESERVED` | `[VIOLATION] Altered` | **`3.72`** | `PASSED` |
| **SAN-04** | Negation Reversal | Tenant shall not sublet premises | किरायेदार उप-किराये पर दे सकता है | `NOT PRESERVED` | `[VIOLATION] Flipped` | **`1.50`** | `PASSED` |
| **SAN-05** | Coverage Inversion | Policy covers accidental damage, including fire | आग और बाढ़ को छोड़कर क्षति कवर करती है | `NOT PRESERVED` | `[VIOLATION] Inverted` | **`1.50`** | `PASSED` |
| **SAN-06** | Amount Change | Deductible of Rs. 5,000 shall apply | 50,000 रुपये की कटौती लागू होगी | `NOT PRESERVED` | `[VIOLATION] Mismatch` | **`1.94`** | `PASSED` |
| **SAN-07** | Deadline Alteration | Notify insurer within seven days | तीस दिनों के भीतर सूचना देगा | `NOT PRESERVED` | `[VIOLATION] Mismatch` | **`1.99`** | `PASSED` |
| **SAN-08** | Exception Stripping | Must pay charges unless uninhabitable | किरायेदार को शुल्क का भुगतान करना होगा | `NOT PRESERVED` | `[VIOLATION] Removed` | **`2.63`** | `PASSED` |
| **SAN-09** | Proviso Omission | Pay rent, provided late penalty 5% | 15,000 रुपये मासिक किराया देगा | `NOT PRESERVED` | `[VIOLATION] Omitted` | **`2.72`** | `PASSED` |
| **SAN-10** | Party-Role Inversion | Landlord must return tenant's deposit | किरायेदार को मकान मालिक की जमा लौटानी होगी | `NOT PRESERVED` | `[VIOLATION] Role Reversed` | **`1.50`** | `PASSED` |

### Key Invariant Fixes
- **SAN-05 (Coverage Inversion):** Enhanced coverage polarity checking to identify exclusionary postpositional phrases (`को छोड़कर`, `के अलावा`, `सिवाय`) when English specifies inclusion (`including`, `covers`), dropping the score to `1.50`.
- **SAN-08 (Exception Removal):** Added subordinating conjunctions (`unless`, `notwithstanding`) to exception detection so unrepresented conditional exemptions trigger `exception_removal`.
- **SAN-10 (Party-Role Reversal):** Introduced grammatical subject-obligor role alignment so swapping the responsible entity (`Landlord` $\leftrightarrow$ `Tenant`) is penalized even when both words appear in both sentences.

---

## 9. Running Verification Tests

Run the full sanity check suite:
```powershell
python run_sanity_checks.py
```

Run the complete 21-unit-test regression suite:
```powershell
python -m unittest tests/test_judgebert_x.py
```

All 21 automated unit tests evaluate:
1. Model prediction acceptance across English-Hindi inputs.
2. Score bounds within $[1.0, 10.0]$.
3. Truthful reporting of active model backbones (preventing silent model substitution).
4. XLM-RoBERTa pair representation $[u; v; |u-v|]$ (2304-dim) and regression head structure.
5. Missing checkpoint offline safety.
6. Zero-human-label training protection (refusal to train on synthetic labels).
7. Zero monolingual cross-contamination in dataset splits.
8. Zero clause overlap between train, val, and test splits.
9. All 10 legal explainability discrepancy categories.
10. Devanagari Unicode and numeral normalization.

---

## 10. Research Limitations & Future Work

1. **Human Calibration Data:** The models operate currently in zero-shot cross-lingual alignment mode because genuine human expert annotations are pending. Calibrating the regression head requires human legal ratings recorded in `data/english_hindi/annotated.csv`.
2. **Cross-Lingual Semantic Role Labeling (SRL):** While rule-based subject-obligor alignment catches common party inversions, deploying full cross-lingual dependency parsing or semantic role labeling will improve handling of complex passive constructions.
3. **Domain Expansion:** Current datasets cover Tenancy Law and Insurance Law; expanding into Employment and Commercial Arbitration contracts represents an important future milestone.

---

## 11. Citation & Acknowledgements

If you utilize this framework or its cross-lingual legal dataset splits in your research, please acknowledge the foundational JUDGEBERT work and this implementation:

```bibtex
@misc{judgebert_x_2026,
  title={Explainable Multilingual Legal Meaning Preservation Using JUDGEBERT},
  author={nlp research j},
  year={2026},
  howpublished={\url{https://github.com/GRAAL-Research/JUDGEBERT}}
}
```
