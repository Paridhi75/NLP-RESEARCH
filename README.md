# Explainable Multilingual Legal Meaning Preservation: A Comparative Study of TF-IDF, Sentence-BERT, and MiniLM-L12 Models

**Subtitle:** *A Neuro-Symbolic Approach to Cross-Lingual Contract Verification Across English and Hindi*   
---

## 1. Overview & Research Motivation

Evaluating legal translation differs fundamentally from general machine translation evaluation. Conventional Semantic Textual Similarity (STS) metrics (such as BLEU, ROUGE, or unconstrained cosine similarity) often report high alignment when general topical vocabulary overlaps, even if critical legal operatives—such as liability direction, statutory conditions, or exception carve-outs—have been reversed or stripped.

This research investigates **Legal Meaning Preservation (LMP)** between English legal clauses and their Hindi translations across tenancy and insurance contracts. We compare three distinct approaches:
1. **Model 1: TF-IDF + Cosine Similarity** (Lexical baseline with bilingual lexicon projection)
2. **Model 2: Sentence-BERT** (Multilingual bi-encoder dense sentence representations)
3. **Model 3: Proposed MiniLM-L12 Neuro-Symbolic Model** (Dense $[u; v; |u-v|]$ pair representation + 10-category deterministic legal invariant verification engine)

---

## 2. The 10 Categories of Legal Meaning Invariants

In legal agreements, subtle linguistic modifications completely alter legal enforceability:

1. **Number and Financial / Currency Amounts:** Numerical values, currency limits, and percentages (e.g., Rs. 5,000 altered to Rs. 50,000).
2. **Negation & Prohibition Reversals:** Flipping negative restrictions to permissions (e.g., *shall not sublet* vs. *उप-किराये पर दे सकता है*).
3. **Modality & Obligation Changes:** Mandatory duties (*shall*, *must*, *करना होगा*) downgraded to permissive authorizations (*may*, *can*, *सकता है*).
4. **Entity & Party Role Reversal:** Swapping obligor and obligee (*Landlord* $\leftrightarrow$ *Tenant*), reversing who owes payment or liability.
5. **Condition & Proviso Omissions:** Dropping prerequisite terms introduced by *provided that*, *subject to*, or *बशर्ते*.
6. **Exception & Carve-Out Removals:** Stripping operative exemptions introduced by *unless*, *except for*, or *को छोड़कर*.
7. **Coverage Scope Inversion:** Flipping policy coverage from inclusion to exclusion (*including* $\rightarrow$ *को छोड़कर*).
8. **Temporal Scope & Deadlines:** Shifting notice periods or compliance timelines (e.g., *7 days* to *30 days*).
9. **Hallucinated / Extraneous Obligations:** Introducing duties not present in the original text.
10. **Clause & Penalty Enforceability Consequence Omission:** Stripping late-payment consequences, interest penalties, or forfeiture terms.

---

## 3. Architecture of the Three Models

### Model 1: TF-IDF + Cosine Similarity
- **Implementation:** `src/tfidf_model.py`, stored in `models/tfidf/`
- **Methodology:** Preprocesses Devanagari text (NFC normalization, Hindi numerals, danda elimination) and projects English domain keywords into Hindi lexical space using a bilingual legal vocabulary dictionary. Computes character and word n-gram TF-IDF representations and cosine similarity.

### Model 2: Sentence-BERT
- **Implementation:** `src/sentence_transformer_model.py`, stored in `models/sentence_transformer/`
- **Backbone:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Methodology:** Projects English and Hindi sentences into a shared 384-dimensional dense semantic embedding space:
  $$\cos(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$
  Maps continuous similarity to the $[1.0, 10.0]$ LMP scale.

### Model 3: Proposed MiniLM-L12 Neuro-Symbolic Model
- **Implementation:** `src/minilm_l12.py`, stored in `models/proposed_minilm/`
- **Architecture:** 
  - Shared Multilingual MiniLM-L12 sentence encoder (384 dimensions)
  - Tri-vector pair representation:
    $$h = [u; v; |u - v|] \in \mathbb{R}^{1152}$$
  - Multi-layer regression head: LayerNorm $\to$ GELU $\to$ Dropout (0.2) $\to$ LayerNorm $\to$ GELU $\to$ Sigmoid $\to [1.0, 10.0]$ scale
  - Coupled with the **Deterministic Legal Invariant Engine** (`src/explainability.py`) to verify all 10 legal categories and penalize invariant violations.

---


## 4. Setup & Installation

### Prerequisites
- Python 3.10 to 3.14 on Windows, Linux, or macOS.

### Step-by-Step Installation

```powershell
# 1. Clone repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd nlp_m2

# 2. Set up virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install requirements
pip install -r requirements.txt
```

---

## 5. Usage & Execution

### 1. Interactive Legal Translation Prediction

Run inference on any English original clause and Hindi translation:

```powershell
python predict.py --original "The tenant shall not sublet the premises." --candidate "किरायेदार परिसर को उप-किराये पर नहीं देगा।"
```

**Sample Output:**
```text
=====================================================================================
 LEGAL MEANING PRESERVATION (LMP) EVALUATION 
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

### 2. Run the 10 Legal Sanity Checks (Task 7)

Evaluates 10 adversarial legal boundary test cases (negation flip, modality downgrade, amount tampering, deadline shifts, party role reversal, coverage inversion, etc.):

```powershell
python run_sanity_checks.py
```

### 3. Run Benchmark Evaluation

Evaluates all three models on the aligned English-Hindi test partition:

```powershell
python evaluate.py
```

### 4. Run Unit Tests

Runs unit and integration tests verifying all 3 models:

```powershell
python -m unittest tests/test_minilm_l12.py
```

---

## 6. Results on the 10 Legal Sanity Checks

| Test ID | Legal Scenario | English Clause | Hindi Clause | Expected | Result |
| :---: | :--- | :--- | :--- | :---: | :---: |
| **SAN-01** | Faithful Translation | Tenant shall pay rent Rs 25,000 on 5th | किरायेदार 5वीं तारीख को 25,000 देगा | `PRESERVED` | `PASSED` |
| **SAN-02** | Unrelated Sentences | Insurer war exclusion clause | किरायेदार पालतू जानवर रख सकता है | `NOT PRESERVED` | `PASSED` |
| **SAN-03** | Modality Downgrade | Insured must undergo medical exam | बीमित व्यक्ति जांच करवा सकता है | `NOT PRESERVED` | `PASSED` |
| **SAN-04** | Negation Reversal | Tenant shall not sublet premises | किरायेदार उप-किराये पर दे सकता है | `NOT PRESERVED` | `PASSED` |
| **SAN-05** | Coverage Inversion | Policy covers accidental damage, including fire | आग और बाढ़ को छोड़कर क्षति कवर करती है | `NOT PRESERVED` | `PASSED` |
| **SAN-06** | Amount Tampering | Deductible of Rs. 5,000 shall apply | 50,000 रुपये की कटौती लागू होगी | `NOT PRESERVED` | `PASSED` |
| **SAN-07** | Deadline Shift | Notify insurer within seven days | तीस दिनों के भीतर सूचना देगा | `NOT PRESERVED` | `PASSED` |
| **SAN-08** | Exception Removal | Must pay charges unless uninhabitable | किरायेदार को शुल्क का भुगतान करना होगा | `NOT PRESERVED` | `PASSED` |
| **SAN-09** | Proviso Omission | Pay rent, provided late penalty 5% | 15,000 रुपये मासिक किराया देगा | `NOT PRESERVED` | `PASSED` |
| **SAN-10** | Party Role Reversal | Landlord must return tenant's deposit | किरायेदार को मकान मालिक की जमा लौटानी होगी | `NOT PRESERVED` | `PASSED` |
