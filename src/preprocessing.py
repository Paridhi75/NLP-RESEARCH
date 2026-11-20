"""
Preprocessing and Text Normalization Module for English-to-Hindi Legal Meaning Preservation.
Handles Devanagari Unicode normalization, English tokenization, and bilingual legal terminology mapping.
"""
import re
import unicodedata
from typing import List, Dict, Set, Tuple


# =====================================================================
# Domain-Specific English-to-Hindi Legal Bilingual Lexicon
# Used for cross-lingual lexical mapping in Model 1 (TF-IDF Baseline)
# =====================================================================
LEGAL_BILINGUAL_LEXICON: Dict[str, List[str]] = {
    # Insurance Entities and Concepts
    "insurer": ["बीमाकर्ता", "कंपनी"],
    "insured": ["बीमित", "पॉलिसीधारक"],
    "policyholder": ["पॉलिसीधारक", "बीमित"],
    "policy": ["पॉलिसी", "बीमा"],
    "claim": ["दावा", "दावे"],
    "claims": ["दावे", "दावा"],
    "premium": ["प्रीमियम", "किश्त"],
    "coverage": ["कवर", "कवरेज", "सुरक्षा"],
    "cover": ["कवर", "कवरेज"],
    "covers": ["कवर", "कवरेज"],
    "covered": ["कवर"],
    "deductible": ["कटौती", "कटौतियाँ"],
    "loss": ["हानि", "नुकसान"],
    "damage": ["क्षति", "नुकसान"],
    "accident": ["दुर्घटना"],
    "hospital": ["अस्पताल"],
    "hospitalisation": ["भर्ती", "अस्पताल"],
    "reimbursement": ["प्रतिपूर्ति", "भुगतान"],
    "indemnify": ["क्षतिपूर्ति", "भरपाई"],
    "liability": ["दायित्व", "उत्तरदायित्व"],
    "fraud": ["धोखाधड़ी", "कपट"],
    "fraudulent": ["कपटपूर्ण", "धोखाधड़ी"],
    "vehicle": ["वाहन", "गाड़ी"],
    "driver": ["चालक", "ड्राइवर"],
    "licence": ["लाइसेंस"],
    "license": ["लाइसेंस"],

    # Tenancy Entities and Concepts
    "landlord": ["मकान मालिक", "स्वामी"],
    "tenant": ["किरायेदार"],
    "premises": ["परिसर", "मकान"],
    "rent": ["किराया", "मासिक"],
    "monthly": ["मासिक", "महीने"],
    "deposit": ["जमा", "डिपॉज़िट"],
    "security": ["सुरक्षा", "सिक्योरिटी"],
    "agreement": ["समझौता", "करार"],
    "sublet": ["उप-किराया", "किराये"],
    "notice": ["नोटिस", "सूचना"],
    "possession": ["कब्ज़ा"],
    "vacant": ["खाली"],
    "maintenance": ["रखरखाव"],
    "repair": ["मरम्मत"],
    "repairs": ["मरम्मत"],
    "electricity": ["बिजली"],
    "water": ["पानी"],
    "taxes": ["कर", "टैक्स"],
    "municipal": ["नगरपालिका"],

    # Legal Modalities & Obligations
    "shall": ["करेगा", "होगी", "होगा", "अनिवार्य", "देगा", "पड़ेगा"],
    "must": ["होगा", "होगी", "अनिवार्य", "ज़रूरी", "पड़ेगा"],
    "mandatory": ["अनिवार्य", "बाध्यकारी"],
    "may": ["सकता", "सकती", "सकते", "अनुमति", "अधिकार"],
    "can": ["सकता", "सकती", "सकते"],
    "entitled": ["हकदार", "अधिकार"],
    "right": ["अधिकार"],

    # Negations & Exclusions
    "not": ["नहीं", "ना"],
    "no": ["कोई नहीं", "नहीं"],
    "never": ["कभी नहीं"],
    "excluding": ["छोड़कर", "सिवाय"],
    "excluded": ["अपवर्जित", "कवर नहीं"],
    "except": ["सिवाय", "छोड़कर"],
    "unless": ["जब तक कि", "सिवाय"],
    "repudiated": ["अस्वीकार", "खारिज"],
    "cancel": ["रद्द"],
    "terminate": ["समाप्त"],

    # Conditions & Temporals
    "if": ["यदि", "अगर"],
    "provided": ["बशर्ते"],
    "condition": ["शर्त"],
    "subject": ["अधीन"],
    "within": ["भीतर", "अंदर"],
    "days": ["दिन", "दिनों"],
    "day": ["दिन", "तारीख"],
    "months": ["महीने", "माह"],
    "month": ["महीना", "माह"],
    "hours": ["घंटे", "घंटों"],
    "years": ["वर्ष", "साल"],
    "before": ["पहले", "पूर्व"],
    "after": ["बाद", "पश्चात"],
    "pay": ["देगा", "अदा", "भुगतान"],
    "payment": ["भुगतान", "अदायगी"],
    "penalty": ["जुर्माना", "दंड", "शास्ति"],
    "late": ["विलंब", "देरी"],
    "incur": ["देय", "लागू", "भुगतान"],
    "interest": ["ब्याज"],
    "percent": ["प्रतिशत", "%"],
    "percentage": ["प्रतिशत", "%"]
}

# Reverse mapping: Hindi to English terms
HINDI_TO_ENGLISH_LEXICON: Dict[str, Set[str]] = {}
for en_w, hi_list in LEGAL_BILINGUAL_LEXICON.items():
    for hi_w in hi_list:
        HINDI_TO_ENGLISH_LEXICON.setdefault(hi_w, set()).add(en_w)


def normalize_english(text: str) -> str:
    """Normalizes English text: lowercase, punctuation spacing, whitespace trimming."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[\s\t\r\n]+", " ", text)
    # Retain alphanumeric and legal punctuation
    text = re.sub(r"[^\w\s\.,;:%\$\-']", "", text)
    return text.strip()


def normalize_hindi(text: str) -> str:
    """Normalizes Hindi Devanagari text using Unicode NFC normalization and clean spacing."""
    if not isinstance(text, str):
        return ""
    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)
    # Remove zero-width spaces/joiners if standalone
    text = re.sub(r"[\u200B\u200C\u200D\uFEFF]", "", text)
    # Normalize Hindi danda (।) to standard danda
    text = text.replace("|", "।")
    # Clean whitespace
    text = re.sub(r"[\s\t\r\n]+", " ", text)
    return text.strip()


def tokenize_english(text: str) -> List[str]:
    """Tokenizes English sentence into words."""
    norm = normalize_english(text)
    return [w for w in re.findall(r"\b[a-zA-Z0-9_\-\$%\.]+\b", norm) if len(w) > 1]


def tokenize_hindi(text: str) -> List[str]:
    """Tokenizes Hindi Devanagari sentence into words."""
    norm = normalize_hindi(text)
    # Retain Devanagari words and numbers
    tokens = [w for w in re.findall(r"[\u0900-\u097F0-9]+", norm) if len(w) > 1]
    return tokens


def project_english_to_hindi_lexicon(en_text: str) -> List[str]:
    """Projects English sentence into equivalent Hindi legal vocabulary."""
    tokens = tokenize_english(en_text)
    projected = []
    for t in tokens:
        if t in LEGAL_BILINGUAL_LEXICON:
            projected.extend(LEGAL_BILINGUAL_LEXICON[t])
    return projected
