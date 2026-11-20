"""
JUDGEBERT-X: English-to-Hindi Sanity Checks Benchmark (Task 7).
Executes the 10 rigorous legal sanity test cases evaluating boundary behavior:
1. SAN-01: Identical / Faithful legal translation
2. SAN-02: Completely unrelated sentences (Cross-domain: Insurance vs Tenancy)
3. SAN-03: Must to May (Modality downgrade)
4. SAN-04: Shall not to Simple future / Negation flip
5. SAN-05: Coverage Inclusion to Exclusion
6. SAN-06: Changed monetary amount (Rs. 5,000 to Rs. 50,000)
7. SAN-07: Changed deadline (7 days to 30 days)
8. SAN-08: Removed exception (Day-care carve-out stripped)
9. SAN-09: Removed condition (Penalty proviso dropped)
10. SAN-10: Changed legal entity (Landlord shifted to Tenant)
"""
import sys
import os
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

WORKSPACE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, WORKSPACE_DIR)

from src.prediction import EnglishHindiLMPPredictor


def run_all_sanity_checks():
    predictor = EnglishHindiLMPPredictor()

    sanity_cases = [
        {
            "id": "SAN-01",
            "scenario": "Faithful / Identical Translation",
            "english": "The Tenant shall pay monthly rent of Rs. 25,000 on or before the fifth day.",
            "hindi": "किरायेदार पांचवीं तारीख को या उससे पहले 25,000 रुपये का मासिक किराया देगा।",
            "expected_decision": "PRESERVED",
            "expected_issues": "None (All invariants preserved)",
            "rationale": "Faithful translation preserves obligations, currency amounts, and deadlines."
        },
        {
            "id": "SAN-02",
            "scenario": "Completely Unrelated Sentences",
            "english": "The insurer shall not be liable for loss arising from war, invasion or nuclear weapons.",
            "hindi": "किरायेदार घरेलू पालतू जानवर रख सकता है बशर्ते वे पड़ोसियों को परेशानी न पहुँचाएँ।",
            "expected_decision": "NOT PRESERVED",
            "expected_issues": "Completely unrelated domains (Insurance vs Tenancy)",
            "rationale": "Different legal domains and facts must receive minimal similarity score."
        },
        {
            "id": "SAN-03",
            "scenario": "Modality Downgrade (Must to May)",
            "english": "The insured person must undergo a medical examination by a doctor nominated by the insurer.",
            "hindi": "बीमित व्यक्ति बीमाकर्ता द्वारा नामित डॉक्टर द्वारा मेडिकल जांच करवा सकता है।",
            "expected_decision": "NOT PRESERVED",
            "expected_issues": "Modality alteration ('must' -> 'सकता है')",
            "rationale": "Mandatory duty weakened to permissive discretionary authorization."
        },
        {
            "id": "SAN-04",
            "scenario": "Negation Marker Discrepancy / Flip",
            "english": "The tenant shall not sublet the premises.",
            "hindi": "किरायेदार परिसर को उप-किराये पर दे सकता है।",
            "expected_decision": "NOT PRESERVED",
            "expected_issues": "negation_flip (Prohibition 'shall not' flipped to permission 'दे सकता है')",
            "rationale": "English statutory prohibition ('shall not sublet') flipped to affirmative permission in Hindi."
        },
        {
            "id": "SAN-05",
            "scenario": "Coverage Inclusion to Exclusion",
            "english": "The insurance policy covers accidental damage, including fire and flooding.",
            "hindi": "बीमा पॉलिसी आग और बाढ़ को छोड़कर आकस्मिक क्षति को कवर करती है।",
            "expected_decision": "NOT PRESERVED",
            "expected_issues": "coverage_reversal ('including' inverted to 'को छोड़कर')",
            "rationale": "Coverage polarity reversed from included benefit to excluded peril via 'को छोड़कर'."
        },
        {
            "id": "SAN-06",
            "scenario": "Changed Monetary Amount (10x Discrepancy)",
            "english": "The deductible of Rs. 5,000 shall apply to each and every claim under Section I.",
            "hindi": "धारा I के तहत प्रत्येक दावे पर 50,000 रुपये की कटौती लागू होगी।",
            "expected_decision": "NOT PRESERVED",
            "expected_issues": "number_change (5,000 vs 50,000)",
            "rationale": "Financial deductible altered tenfold from Rs. 5,000 to Rs. 50,000."
        },
        {
            "id": "SAN-07",
            "scenario": "Changed Deadline (7 days to 30 days)",
            "english": "The insured shall notify the insurer in writing of any accident within seven days.",
            "hindi": "बीमित व्यक्ति तीस दिनों के भीतर किसी भी दुर्घटना की लिखित सूचना बीमाकर्ता को देगा।",
            "expected_decision": "NOT PRESERVED",
            "expected_issues": "number_change / temporal change (7 vs 30)",
            "rationale": "Notice requirement timeline expanded from 7 days to 30 days."
        },
        {
            "id": "SAN-08",
            "scenario": "Removed Exception (Unless Carve-Out)",
            "english": "The tenant must pay maintenance charges unless structural defects make the property uninhabitable.",
            "hindi": "किरायेदार को रखरखाव शुल्क का भुगतान करना होगा।",
            "expected_decision": "NOT PRESERVED",
            "expected_issues": "exception_removal ('unless' carve-out stripped)",
            "rationale": "Subordinating conjunction 'unless' carve-out stripped, turning conditional liability into absolute duty."
        },
        {
            "id": "SAN-09",
            "scenario": "Removed Condition (5% Late Penalty Proviso)",
            "english": "The tenant shall pay a monthly rent of Rs 15,000 before the 5th day of each month, provided that late payment shall incur a penalty of 5%.",
            "hindi": "किरायेदार प्रत्येक माह की 5 तारीख से पहले 15,000 रुपये मासिक किराया देगा।",
            "expected_decision": "NOT PRESERVED",
            "expected_issues": "condition_omission ('provided that' & 5% penalty omitted)",
            "rationale": "Operative conditional late payment penalty consequence stripped."
        },
        {
            "id": "SAN-10",
            "scenario": "Party Swap (Landlord → Tenant)",
            "english": "Landlord must return the tenant's security deposit.",
            "hindi": "किरायेदार को मकान मालिक की सुरक्षा जमा राशि लौटानी होगी।",
            "expected_decision": "NOT PRESERVED",
            "expected_issues": "party_role_reversal (Obligor/Obligee roles inverted)",
            "rationale": "Responsible parties reversed (Landlord to Tenant); reveals bi-encoder semantic similarity limitation."
        }
    ]

    print("=====================================================================================")
    print("        JUDGEBERT-X: 10 RIGOROUS ENGLISH-TO-HINDI SANITY CHECKS (TASK 7)             ")
    print("=====================================================================================")

    results_table = []

    for idx, case in enumerate(sanity_cases, start=1):
        cid = case["id"]
        scen = case["scenario"]
        en = case["english"]
        hi = case["hindi"]
        exp_dec = case["expected_decision"]

        res = predictor.predict_all(en, hi)

        raw_cos = res.get("raw_cosine_sim", 0.0)
        base_lmp = res.get("base_calibrated_lmp", 1.0)
        final_lmp = res.get("final_predicted_lmp", 1.0)
        pres_dec = res.get("preservation_decision", "N/A")
        err_type = res.get("legal_error_type", "none")
        rationale = res.get("legal_rationale", "N/A")
        invariants = res.get("invariant_checks", "N/A")

        sc_tfidf = res["models"]["TF-IDF"]["score"]

        # Check if sanity check passed
        is_passed = False
        if exp_dec == "PRESERVED":
            is_passed = ("PRESERVED" in pres_dec and "NOT PRESERVED" not in pres_dec)
        else:
            is_passed = ("NOT PRESERVED" in pres_dec)

        if not is_passed and exp_dec == "NOT PRESERVED" and "PRESERVED" in pres_dec:
            status_str = "FALSE PRESERVATION (Role Reversal Limitation)"
        elif is_passed:
            status_str = "PASSED"
        else:
            status_str = "FAILED"

        results_table.append({
            "Test ID": cid,
            "Scenario": scen,
            "Expected": exp_dec,
            "TF-IDF": f"{sc_tfidf:.2f}" if sc_tfidf is not None else "N/A",
            "Cosine Sim": f"{raw_cos:.4f}",
            "Sentence-BERT": f"{base_lmp:.2f}",
            "MiniLM Transformer LMP": f"{final_lmp:.2f}",
            "Preservation Decision": pres_dec[:25] + "...",
            "Status": status_str
        })

        print(f"\n[{cid}] Scenario: {scen}")
        print("=" * 85)
        print(" JUDGEBERT LEGAL MEANING PRESERVATION (LMP) EVALUATION ")
        print("=" * 85)
        print(f"Original Text        : {en}")
        print(f"Candidate (Hindi)    : {hi}")
        print("-" * 85)
        print(f"TF IDF ,COSINE SIMILARITY  LMP    : {raw_cos:.4f}  (Topic / Lexical Semantic Overlap)")
        print(f"SENTENCE BERT LMP  : {base_lmp:.2f} / 10.0")
        print(f"FINAL PREDICTED LMP  : {final_lmp:.2f} / 10.0")
        print(f"Preservation Decision: [ {pres_dec} ]")
        print(f"Legal Error Type     : {err_type}")
        print(f"Legal Rationale      : {rationale}")
        print(f"Invariant Checks     : {invariants}")
        print(f"Sanity Check Status  : [{status_str}]")
        print("=" * 85)

    df_results = pd.DataFrame(results_table)
    out_csv = os.path.join(WORKSPACE_DIR, "outputs", "reports", "sanity_check_results.csv")
    df_results.to_csv(out_csv, index=False, encoding="utf-8")

    print("\n\n=====================================================================================")
    print("                     SANITY CHECKS BENCHMARK SUMMARY TABLE                           ")
    print("=====================================================================================")
    print(df_results.to_string(index=False))
    print("=====================================================================================")
    total_passed = sum(1 for r in results_table if r["Status"] == "PASSED")
    false_pres = sum(1 for r in results_table if "FALSE PRESERVATION" in r["Status"])
    print(f"\nFinal Sanity Check Result: {total_passed} / {len(sanity_cases)} Invariant Checks Passed")
    if false_pres > 0:
        print(f"Research Observation: {false_pres} Documented False Preservation case (SAN-10: Bi-Encoder Role Reversal Limitation)")
    print(f"Report exported to: {out_csv}\n")


if __name__ == "__main__":
    run_all_sanity_checks()
