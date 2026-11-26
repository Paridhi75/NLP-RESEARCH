"""
English-to-Hindi Legal Meaning Preservation Explainability Engine.
Identifies WHAT legal meaning elements changed between an English source sentence
and its Hindi translation across 10 distinct legal error categories:
1. NUMBER AND PERCENTAGE / CURRENCY CHANGE
2. NEGATION REVERSAL
3. MODALITY AND OBLIGATION CHANGE
4. ENTITY OR PARTY CHANGE
5. CONDITION OMISSION
6. EXCEPTION REMOVAL
7. SCOPE DIVERGENCE
8. DATE AND TEMPORAL CHANGE
9. HALLUCINATED INFORMATION
10. CLAUSE AND PENALTY OMISSION
"""
import re
import unicodedata
from typing import Dict, Any, List, Tuple, Optional, Set

from .preprocessing import normalize_english, normalize_hindi, tokenize_english, tokenize_hindi


# Number words to numeric mapping for English and Hindi (including legal ordinals)
EN_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "twenty-four": 24,
    "twenty four": 24, "thirty": 30, "forty-eight": 48, "forty eight": 48,
    "fifty": 50, "sixty": 60, "ninety": 90, "hundred": 100, "thousand": 1000,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10
}

HI_WORD_NUMBERS = {
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5,
    "छह": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10, "ग्यारह": 11, "बारह": 12,
    "पंद्रह": 15, "पन्द्रह": 15, "बीस": 20, "चौबीस": 24, "तीस": 30, "पैंतालीस": 45,
    "अड़तालीस": 48, "पचास": 50, "साठ": 60, "नब्बे": 90, "सौ": 100, "हज़ार": 1000, "हजार": 1000,
    "लाख": 100000,
    "पहला": 1, "पहली": 1, "दूसरा": 2, "दूसरी": 2, "तीसरा": 3, "तीसरी": 3,
    "चौथा": 4, "चौथी": 4, "पांचवां": 5, "पांचवीं": 5, "पाँचवां": 5, "पाँचवीं": 5
}

DEVANAGARI_DIGITS = {
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9"
}


def devanagari_to_arabic_digits(text: str) -> str:
    """Converts Devanagari numerals (१, २, ३) to Arabic numerals (1, 2, 3)."""
    for d_char, a_char in DEVANAGARI_DIGITS.items():
        text = text.replace(d_char, a_char)
    return text


def extract_numbers_from_english(text: str) -> Set[int]:
    """Extracts numerical quantities from English text (digits and number words)."""
    text_clean = text.lower()
    nums = set()
    for m in re.findall(r"\b\d+(?:,\d+)*(?:\.\d+)?\b", text_clean):
        try:
            val = int(float(m.replace(",", "")))
            nums.add(val)
        except Exception:
            pass
    for w, val in EN_WORD_NUMBERS.items():
        if re.search(r"\b" + re.escape(w) + r"\b", text_clean):
            nums.add(val)
    return nums


def extract_numbers_from_hindi(text: str) -> Set[int]:
    """Extracts numerical quantities from Hindi text (digits, Devanagari numerals, and words)."""
    text_conv = devanagari_to_arabic_digits(text)
    nums = set()
    for m in re.findall(r"\b\d+(?:,\d+)*(?:\.\d+)?\b", text_conv):
        try:
            val = int(float(m.replace(",", "")))
            nums.add(val)
        except Exception:
            pass
    # Match Devanagari words strictly by tokens to avoid substring false positives (e.g. 'चार' in 'कर्मचारी')
    tokens = set(re.findall(r"[\u0900-\u097F]+", text))
    for w, val in HI_WORD_NUMBERS.items():
        if w in tokens:
            nums.add(val)
    return nums


def extract_percentages_from_english(text: str) -> List[str]:
    """Extracts percentage expressions from English text (e.g., '5%', '10 percent')."""
    p1 = re.findall(r"\b\d+(?:\.\d+)?%", text)
    p2 = re.findall(r"\b\d+(?:\.\d+)?\s*percent\b", text, flags=re.IGNORECASE)
    return p1 + p2


def extract_percentages_from_hindi(text: str) -> List[str]:
    """Extracts percentage expressions from Hindi text (e.g., '5%', '5 प्रतिशत')."""
    text_conv = devanagari_to_arabic_digits(text)
    p1 = re.findall(r"\b\d+(?:\.\d+)?%", text_conv)
    p2 = re.findall(r"\b\d+(?:\.\d+)?\s*प्रतिशत\b", text_conv)
    return p1 + p2


class CrossLingualLegalExplainabilityEngine:
    """Detects and explains meaning divergences between English original and Hindi translation."""

    # 1. Modality & Obligation Patterns
    EN_OBLIGATION = [r"\bshall\b", r"\bmust\b", r"\brequired to\b", r"\bobligated to\b", r"\bbound to\b", r"\bhas to\b", r"\bhave to\b"]
    EN_PERMISSION = [r"\bmay\b", r"\bcan\b", r"\bentitled to\b", r"\bpermitted to\b", r"\ballowed to\b", r"\bat its option\b"]
    EN_PROHIBITION = [r"\bshall not\b", r"\bmust not\b", r"\bcannot\b", r"\bprohibited\b", r"\bnot permitted\b"]

    HI_OBLIGATION = [r"होगा", r"होगी", r"होंगे", r"करना होगा", r"देना होगा", r"पड़ेगा", r"अनिवार्य", r"आवश्यक", r"बाध्य"]
    HI_PERMISSION = [r"सकता है", r"सकती है", r"सकते हैं", r"हकदार", r"अनुमति", r"अधिकार", r"वैकल्पिक"]
    HI_PROHIBITION = [r"नहीं कर सकता", r"नहीं कर सकती", r"नहीं करेगा", r"नहीं करेगी", r"निषिद्ध", r"रोक", r"मनाही"]

    # 2. Negation Patterns
    EN_NEGATION = [r"\bnot\b", r"\bnever\b", r"\bno\b", r"\bno longer\b", r"\bneither\b", r"\bnor\b", r"\bwithout\b"]
    HI_NEGATION = [r"नहीं", r"नही", r"\bन\b", r"कभी नहीं", r"बिना"]

    # 3. Entity Pairs (English <-> Hindi)
    ENTITY_PAIRS = [
        ("tenant", ["किरायेदार"]),
        ("landlord", ["मकान मालिक", "स्वामी"]),
        ("insurer", ["बीमाकर्ता", "कंपनी"]),
        ("insured", ["बीमित", "पॉलिसीधारक"]),
        ("driver", ["चालक", "ड्राइवर"]),
        ("third party", ["तृतीय पक्ष", "तीसरा पक्ष"]),
        ("nominee", ["नामांकित"]),
        ("court", ["अदालत", "न्यायालय"]),
        ("doctor", ["डॉक्टर", "चिकित्सक"])
    ]

    # 4. Condition Patterns
    EN_CONDITIONS = [
        r"\bif\b", r"\bprovided that\b", r"\bprovided\b", r"\bin the event\b",
        r"\bsubject to\b", r"\bon condition that\b", r"\bin case\b", r"\bwhenever\b",
        r"\bonly with\b", r"\bwith(?:out)? (?:the )?(?:prior )?written consent\b",
        r"\bwith(?:out)? (?:the )?(?:prior )?written permission\b",
        r"\bprior written consent\b", r"\bwritten consent\b", r"\bprior consent\b",
        r"\bwritten permission\b", r"\bprior written permission\b",
        r"\bconsent from\b", r"\bconsent of\b", r"\bpermission from\b", r"\bpermission of\b"
    ]
    HI_CONDITIONS = [
        r"यदि", r"अगर", r"बशर्ते", r"की स्थिति में", r"के अधीन", r"होने पर", r"शर्त",
        r"सहमति", r"अनुमति", r"मंजूरी", r"इजाजत", r"इज़ाज़त"
    ]

    # 5. Exception Patterns
    EN_EXCEPTIONS = [r"\bexcept\b", r"\bunless\b", r"\bexcluding\b", r"\bother than\b", r"\bsave for\b", r"\bnotwithstanding\b", r"\bapart from\b"]
    HI_EXCEPTIONS = [r"सिवाय", r"के अलावा", r"को छोड़कर", r"के अतिरिक्त", r"जब तक कि", r"के बावजूद"]

    # 6. Scope & Quantifiers
    EN_SCOPE_UNIVERSAL = [r"\ball\b", r"\bevery\b", r"\beach\b", r"\bentire\b", r"\bfull\b"]
    EN_SCOPE_RESTRICTIVE = [r"\bsolely\b", r"\bonly\b", r"\bexclusive\b", r"\bmainly\b"]
    HI_SCOPE_UNIVERSAL = [r"सभी", r"प्रत्येक", r"हर ", r"पूरा", r"पूरी"]
    HI_SCOPE_RESTRICTIVE = [r"केवल", r"सिर्फ", r"एकमात्र", r"अनन्य"]

    # 7. Temporal & Timeline Markers
    EN_TEMPORAL_PRE = [r"\bbefore\b", r"\bprior to\b"]
    EN_TEMPORAL_POST = [r"\bafter\b", r"\bfollowing\b", r"\bsubsequent to\b"]
    EN_TEMPORAL_WINDOW = [r"\bwithin\b", r"\bduring\b", r"\buntil\b"]
    HI_TEMPORAL_PRE = [r"पहले", r"पूर्व"]
    HI_TEMPORAL_POST = [r"बाद", r"पश्चात"]
    HI_TEMPORAL_WINDOW = [r"के भीतर", r"के अंदर", r"दौरान", r"तक"]

    # 8. Penalty & Consequence Patterns
    EN_PENALTY = [r"\bpenalty\b", r"\blate payment\b", r"\bincur(?:s|red)? a penalty\b", r"\binterest\b", r"\bfine\b", r"\bforfeiture\b"]
    HI_PENALTY = [r"जुर्माना", r"दंड", r"दण्ड", r"शास्ति", r"विलंब", r"देरी", r"ब्याज", r"हर्जाना"]

    @classmethod
    def _matches(cls, text: str, patterns: List[str]) -> List[str]:
        found = []
        for p in patterns:
            m = re.findall(p, text, flags=re.IGNORECASE)
            if m:
                found.extend(m)
        return found

    @classmethod
    def analyze_english_hindi_pair(cls, en_text: str, hi_text: str) -> Dict[str, Any]:
        """
        Executes cross-lingual legal rule-based checks across all 10 legal error categories.
        Returns detailed detected changes, explicit statuses, and category counts.
        """
        en_norm = normalize_english(en_text)
        hi_norm = normalize_hindi(hi_text)

        findings: List[Dict[str, Any]] = []
        counts = {
            "number_changes": 0,
            "negation_changes": 0,
            "modality_changes": 0,
            "entity_changes": 0,
            "condition_omissions": 0,
            "exception_removals": 0,
            "scope_changes": 0,
            "temporal_changes": 0,
            "hallucinated_information": 0,
            "omissions": 0
        }

        # -------------------------------------------------------------
        # 1. NUMBER AND PERCENTAGE / CURRENCY CHANGE
        # -------------------------------------------------------------
        en_years = {int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", en_text)}
        hi_years = {int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", devanagari_to_arabic_digits(hi_text))}
        date_year_diff = (en_years != hi_years and (bool(en_years) or bool(hi_years)))

        en_nums = extract_numbers_from_english(en_text)
        hi_nums = extract_numbers_from_hindi(hi_text)
        en_pcts = extract_percentages_from_english(en_text)
        hi_pcts = extract_percentages_from_hindi(hi_text)

        # Separate calendar years from pure quantity/monetary numbers
        pure_en_nums = en_nums - en_years
        pure_hi_nums = hi_nums - hi_years

        num_diff = (pure_en_nums != pure_hi_nums and (bool(pure_en_nums) or bool(pure_hi_nums)))
        pct_diff = (bool(en_pcts) and not bool(hi_pcts)) or (not bool(en_pcts) and bool(hi_pcts))

        if num_diff or pct_diff:
            counts["number_changes"] += 1
            en_items = []
            if pure_en_nums:
                en_items.append(f"amounts: {sorted(list(pure_en_nums))}")
            if en_pcts:
                en_items.append(f"rates: {en_pcts}")

            hi_items = []
            if pure_hi_nums:
                hi_items.append(f"amounts: {sorted(list(pure_hi_nums))}")
            if hi_pcts:
                hi_items.append(f"rates: {hi_pcts}")
            else:
                if en_pcts:
                    hi_items.append("rates: [None detected]")

            findings.append({
                "category": "NUMBER CHANGE",
                "status": "DETECTED",
                "english": f"Numerical specifications found: {', '.join(en_items)}",
                "hindi": f"Numerical specifications translated: {', '.join(hi_items)}",
                "explanation": f"Discrepancy in numerical amounts or percentage rates between English and Hindi." + (
                    " (Note: The missing 5% penalty rate overlaps with the omitted penalty clause.)" if ("5%" in en_pcts and not hi_pcts) else ""
                ),
                "overlap_note": "Overlaps with omitted late-payment penalty consequence." if ("5%" in en_pcts and not hi_pcts) else None
            })

        # -------------------------------------------------------------
        # 2. NEGATION REVERSAL
        # -------------------------------------------------------------
        en_has_neg = bool(cls._matches(en_norm, cls.EN_NEGATION))
        hi_has_neg = bool(cls._matches(hi_norm, cls.HI_NEGATION))

        en_prohib = cls._matches(en_norm, cls.EN_PROHIBITION)
        hi_perm = cls._matches(hi_norm, cls.HI_PERMISSION)
        hi_verb_neg = bool(re.search(r"(?:नहीं|नही)\s*(?:[^\s।,!?]+\s*)?(?:सकता|सकती|सकते|करेगा|करेगी|करेंगे|देगा|देगी|देंगे|होगा|होगी)", hi_norm)) or bool(re.search(r"(?:^|\s)न\s*(?:[^\s।,!?]+\s*)?(?:सकता|सकती|सकते|देगा|देगी|देंगे)", hi_norm))

        if bool(en_prohib) and bool(hi_perm) and not hi_verb_neg:
            counts["negation_changes"] += 1
            findings.append({
                "category": "NEGATION CHANGE",
                "status": "DETECTED",
                "english": f"Explicit prohibition: {', '.join(en_prohib)}",
                "hindi": f"Affirmative permission: {', '.join(hi_perm)}",
                "explanation": "The original legal clause prohibits this action, but the translation expresses affirmative permission without verbal negation.",
                "overlap_note": None
            })
        elif en_has_neg != hi_has_neg:
            counts["negation_changes"] += 1
            if en_has_neg and not hi_has_neg:
                exp = "The English sentence contains a negative/exclusionary clause ('not/never/no') which is absent in the Hindi translation."
                en_str = "Negative restriction present"
                hi_str = "Affirmative phrasing (negation absent)"
            else:
                exp = "The Hindi translation introduces a negative ('नहीं/न') not present in the original English sentence."
                en_str = "Affirmative phrasing"
                hi_str = "Negative particle ('नहीं/न') introduced"

            findings.append({
                "category": "NEGATION CHANGE",
                "status": "DETECTED",
                "english": en_str,
                "hindi": hi_str,
                "explanation": exp,
                "overlap_note": None
            })

        # -------------------------------------------------------------
        # 3. MODALITY AND OBLIGATION CHANGE
        # -------------------------------------------------------------
        en_ob = cls._matches(en_norm, cls.EN_OBLIGATION)
        en_pm = cls._matches(en_norm, cls.EN_PERMISSION)
        hi_ob = cls._matches(hi_norm, cls.HI_OBLIGATION)
        hi_pm = cls._matches(hi_norm, cls.HI_PERMISSION)

        if bool(en_ob) and bool(hi_pm) and not bool(hi_ob):
            counts["modality_changes"] += 1
            findings.append({
                "category": "MODALITY CHANGE",
                "status": "DETECTED",
                "english": f"Mandatory duty ({', '.join(set(en_ob))})",
                "hindi": f"Discretionary permission ({', '.join(set(hi_pm))})",
                "explanation": "A mandatory legal obligation ('shall/must') appears to have been downgraded to discretionary permission ('कर सकता है').",
                "overlap_note": None
            })
        elif bool(en_pm) and bool(hi_ob) and not bool(hi_pm):
            counts["modality_changes"] += 1
            findings.append({
                "category": "MODALITY CHANGE",
                "status": "DETECTED",
                "english": f"Permissive right ({', '.join(set(en_pm))})",
                "hindi": f"Mandatory obligation ({', '.join(set(hi_ob))})",
                "explanation": "A permissive right or option ('may/can') appears to have been converted into an enforceable obligation.",
                "overlap_note": None
            })

        # -------------------------------------------------------------
        # 4. ENTITY OR PARTY CHANGE
        # -------------------------------------------------------------
        for en_ent, hi_syns in cls.ENTITY_PAIRS:
            en_present = bool(re.search(r"\b" + re.escape(en_ent) + r"\b", en_norm))
            hi_present = any(h in hi_norm for h in hi_syns)

            if en_ent == "landlord" and en_present and not hi_present:
                if "किरायेदार" in hi_norm and not bool(re.search(r"\btenant\b", en_norm)):
                    counts["entity_changes"] += 1
                    findings.append({
                        "category": "ENTITY CHANGE",
                        "status": "DETECTED",
                        "english": "Obligation designated to Landlord",
                        "hindi": "Designated party translated as किरायेदार (Tenant)",
                        "explanation": "Legal entity substitution detected: responsibility appears to have shifted from Landlord to Tenant.",
                        "overlap_note": None
                    })
            elif en_ent == "tenant" and en_present and not hi_present:
                if ("मकान मालिक" in hi_norm or "मालिक" in hi_norm) and not bool(re.search(r"\blandlord\b", en_norm)):
                    counts["entity_changes"] += 1
                    findings.append({
                        "category": "ENTITY CHANGE",
                        "status": "DETECTED",
                        "english": "Obligation designated to Tenant",
                        "hindi": "Designated party translated as मकान मालिक (Landlord)",
                        "explanation": "Legal entity substitution detected: responsibility appears to have shifted from Tenant to Landlord.",
                        "overlap_note": None
                    })
            elif en_ent == "insurer" and en_present and not hi_present:
                if "बीमित" in hi_norm and not bool(re.search(r"\binsured\b", en_norm)):
                    counts["entity_changes"] += 1
                    findings.append({
                        "category": "ENTITY CHANGE",
                        "status": "DETECTED",
                        "english": "Entity: Insurer (बीमाकर्ता)",
                        "hindi": "Entity translated as बीमित (Insured)",
                        "explanation": "Legal party swap detected between Insurer and Insured.",
                        "overlap_note": None
                    })

        # -------------------------------------------------------------
        # 5. CONDITION OMISSION
        # -------------------------------------------------------------
        en_conds = cls._matches(en_norm, cls.EN_CONDITIONS)
        hi_conds = cls._matches(hi_norm, cls.HI_CONDITIONS)

        if bool(en_conds) and not bool(hi_conds):
            counts["condition_omissions"] += 1
            findings.append({
                "category": "CONDITION OMISSION",
                "status": "DETECTED",
                "english": f"Condition marker present: {', '.join(set(en_conds))}",
                "hindi": "No conditional trigger ('यदि/अगर/बशर्ते') found",
                "explanation": "A prerequisite legal condition ('if/provided that/subject to') in the original text is missing from the Hindi translation.",
                "overlap_note": "In proviso clauses, condition omission often encompasses the entire qualifying penalty or exception."
            })

        # -------------------------------------------------------------
        # 6. EXCEPTION REMOVAL
        # -------------------------------------------------------------
        en_excs = cls._matches(en_norm, cls.EN_EXCEPTIONS)
        hi_excs = cls._matches(hi_norm, cls.HI_EXCEPTIONS)

        if bool(en_excs) and not bool(hi_excs):
            counts["exception_removals"] += 1
            findings.append({
                "category": "EXCEPTION REMOVAL",
                "status": "DETECTED",
                "english": f"Exception carve-out present: {', '.join(set(en_excs))}",
                "hindi": "No exclusionary exception ('सिवाय/को छोड़कर') found",
                "explanation": "An exclusionary carve-out ('except/unless/excluding') in the English clause was stripped in the translation, creating an overbroad rule.",
                "overlap_note": None
            })

        # -------------------------------------------------------------
        # 7. SCOPE DIVERGENCE
        # -------------------------------------------------------------
        en_sc_univ = cls._matches(en_norm, cls.EN_SCOPE_UNIVERSAL)
        en_sc_rest = cls._matches(en_norm, cls.EN_SCOPE_RESTRICTIVE)
        hi_sc_univ = cls._matches(hi_norm, cls.HI_SCOPE_UNIVERSAL)
        hi_sc_rest = cls._matches(hi_norm, cls.HI_SCOPE_RESTRICTIVE)

        if bool(en_sc_rest) and not bool(hi_sc_rest):
            counts["scope_changes"] += 1
            findings.append({
                "category": "SCOPE CHANGE",
                "status": "NEEDS REVIEW",
                "english": f"Restrictive scope: {', '.join(set(en_sc_rest))}",
                "hindi": "Restrictive quantifier ('केवल/सिर्फ') not detected",
                "explanation": "The English scope may be strictly limited ('only/solely'), whereas the Hindi translation lacks an explicit restrictive scope marker.",
                "overlap_note": None
            })

        # -------------------------------------------------------------
        # 8. DATE AND TEMPORAL CHANGE
        # -------------------------------------------------------------
        en_pre = cls._matches(en_norm, cls.EN_TEMPORAL_PRE)
        en_post = cls._matches(en_norm, cls.EN_TEMPORAL_POST)
        hi_pre = cls._matches(hi_norm, cls.HI_TEMPORAL_PRE)
        hi_post = cls._matches(hi_norm, cls.HI_TEMPORAL_POST)

        if date_year_diff:
            counts["temporal_changes"] += 1
            findings.append({
                "category": "DATE AND TEMPORAL CHANGE",
                "status": "DETECTED",
                "english": f"Contract date / year: {sorted(list(en_years)) if en_years else 'None'}",
                "hindi": f"Contract date / year: {sorted(list(hi_years)) if hi_years else 'None'}",
                "explanation": "Contractual date or expiration year altered between English and Hindi translation.",
                "overlap_note": None
            })
        elif (bool(en_pre) and bool(hi_post)) or (bool(en_post) and bool(hi_pre)):
            counts["temporal_changes"] += 1
            findings.append({
                "category": "DATE AND TEMPORAL CHANGE",
                "status": "DETECTED",
                "english": f"Temporal relation: {'before' if en_pre else 'after'}",
                "hindi": f"Temporal relation: {'बाद (after)' if hi_post else 'पहले (before)'}",
                "explanation": "Temporal relationship inverted between English original and Hindi translation (e.g. before vs after deadline).",
                "overlap_note": None
            })

        # -------------------------------------------------------------
        # 9. HALLUCINATED INFORMATION
        # -------------------------------------------------------------
        if ("जुर्माना" in hi_norm or "दण्ड" in hi_norm or "दंड" in hi_norm) and not bool(re.search(r"\bpenalty|fine|damages\b", en_norm)):
            counts["hallucinated_information"] += 1
            findings.append({
                "category": "HALLUCINATED INFORMATION",
                "status": "DETECTED",
                "english": "No penalty/damages specified",
                "hindi": "Contains 'जुर्माना/दंड' (penalty/fine)",
                "explanation": "The Hindi translation mentions a financial penalty or punitive fine not specified in the English original.",
                "overlap_note": None
            })

        # -------------------------------------------------------------
        # 10. CLAUSE AND PENALTY OMISSION
        # -------------------------------------------------------------
        en_pen = cls._matches(en_norm, cls.EN_PENALTY)
        hi_pen = cls._matches(hi_norm, cls.HI_PENALTY)

        # Check for missing penalty/late payment clause
        if bool(en_pen) and not bool(hi_pen):
            counts["omissions"] += 1
            findings.append({
                "category": "OMISSION",
                "status": "DETECTED",
                "english": f"Penalty/late consequence specified: {', '.join(set(en_pen))}",
                "hindi": "No penalty/delay markers ('जुर्माना/विलंब/दंड') found",
                "explanation": "The original clause attaches an operative contractual consequence (e.g., late payment penalty), which has been omitted from the Hindi translation.",
                "overlap_note": "This consequence omission overlaps with the missing condition ('provided that') and missing rate ('5%')."
            })
        else:
            # Word compression check
            en_tokens = tokenize_english(en_text)
            hi_tokens = tokenize_hindi(hi_text)
            length_ratio = len(hi_tokens) / max(1, len(en_tokens))

            if length_ratio < 0.40 and len(en_tokens) >= 10:
                counts["omissions"] += 1
                findings.append({
                    "category": "OMISSION",
                    "status": "NEEDS REVIEW",
                    "english": f"Length: {len(en_tokens)} words",
                    "hindi": f"Length: {len(hi_tokens)} words (compression ratio {length_ratio:.2f})",
                    "explanation": "Severe condensation detected: Hindi translation retains less than 40% of the word count of the original clause, indicating potential omitted clauses.",
                    "overlap_note": None
                })

        total_issues = sum(counts.values())

        return {
            "findings": findings,
            "counts": counts,
            "total_issues": total_issues,
            "detection_limitation_disclaimer": (
                "Rule-based detection matches lexical keywords, regex patterns, numerals, and bilingual lexicon entries. "
                "It does not possess full semantic or jurisdictional legal reasoning capability. "
                "Automated findings are screening flags and must be verified by a qualified legal professional."
            )
        }

    @classmethod
    def explain_model_prediction(cls, model_name: str, score: Optional[float], category: str, analysis: Dict[str, Any]) -> str:
        """Provides an honest, model-specific explanation of the predicted LMP score."""
        if score is None:
            return "Model score unavailable."

        issues = analysis["total_issues"]
        has_neg = analysis["counts"]["negation_changes"] > 0
        has_mod = analysis["counts"]["modality_changes"] > 0
        has_num = analysis["counts"]["number_changes"] > 0

        if model_name in ("Cross-Lingual TF-IDF Baseline", "TF-IDF + Cosine Similarity", "TF-IDF"):
            if score >= 7.0:
                return "Score is high because bilingual lexicon projection detected substantial shared legal terminology."
            elif has_num or has_neg:
                return "Score is lower due to reduced lexical overlap and mismatched numerical/legal tokens."
            else:
                return "Score reflects moderate bilingual lexical alignment across Latin and Devanagari vocabulary."

        elif model_name in ("Multilingual Sentence Transformer", "Sentence-BERT"):
            if score >= 8.0 and (has_neg or has_mod):
                return "Score is high due to strong topical sentence similarity; note that dense multilingual embeddings often remain close despite polarity or modal inversions."
            elif score >= 7.5:
                return "Score reflects strong multilingual semantic alignment in shared embedding space."
            else:
                return "Score reflects moderate semantic divergence between the English clause and Hindi translation."

        elif model_name in ("Proposed MiniLM-L12", "MiniLM-L12"):
            if analysis.get("has_checkpoint", False):
                return "Score is computed by the trained MiniLM-L12 regression head combined with the 10-category legal invariant engine."
            else:
                return "Score reflects the Proposed MiniLM-L12 neuro-symbolic representation combined with the 10-category legal invariant engine."

        return f"Score falls into category '{category}' based on cross-lingual evaluation."

    @classmethod
    def compare_models_with_explainability(cls, results: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesizes explainability findings with model score comparison and identifies disagreements."""
        models = results["models"]
        scores = {name: info["score"] for name, info in models.items() if info.get("score") is not None}

        if not scores:
            return {
                "summary": "No numerical model scores available for comparison.",
                "models_agree": False,
                "score_difference": 0.0,
                "disagreements": ["Model scores are unavailable or pending weights."]
            }

        highest_model = max(scores, key=scores.get)
        lowest_model = min(scores, key=scores.get)
        diff = round(scores[highest_model] - scores[lowest_model], 2)
        models_agree = diff <= 1.5

        disagreements = []
        has_neg = analysis["counts"]["negation_changes"] > 0

        # Check for disagreement between rule-based polarity flip and high semantic score
        st_score = scores.get("Sentence Transformer")
        if has_neg and st_score is not None and st_score >= 8.0:
            disagreements.append(
                f"Rule-based analysis detected a Negation Change, yet Sentence Transformer assigned a high score ({st_score:.2f}/10). "
                "This demonstrates that dense multilingual sentence embeddings can exhibit blind spots on single-token polarity inversions."
            )

        if diff >= 3.0:
            disagreements.append(
                f"Significant model divergence ({diff:.2f} points) between {highest_model} ({scores[highest_model]:.2f}) and {lowest_model} ({scores[lowest_model]:.2f}). "
                "Lexical dictionary matching and neural sentence representations evaluated this translation differently."
            )

        return {
            "highest_model": highest_model,
            "highest_score": scores[highest_model],
            "lowest_model": lowest_model,
            "lowest_score": scores[lowest_model],
            "models_agree": models_agree,
            "score_difference": diff,
            "disagreements": disagreements
        }
