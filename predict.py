"""
Terminal-Based English-to-Hindi Legal Meaning Preservation (LMP) Prediction and Explainability Interface.
Executes all three models and the cross-lingual legal explainability engine directly in the VS Code terminal.
Driven by sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 as the final predicted model.
"""
import sys
import os
import argparse

# Ensure standard UTF-8 stream output on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

WORKSPACE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, WORKSPACE_DIR)

from src.prediction import EnglishHindiLMPPredictor, PRESERVED_MIN_THRESHOLD, PARTIAL_MIN_THRESHOLD


def print_startup_notice():
    print("\nAnalyzing legal meaning preservation...")
    print("Loading multilingual model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2...")


def print_results(res, verbose=False):
    en_text = res["english_sentence"]
    hi_text = res["hindi_sentence"]
    raw_cos = res.get("raw_cosine_sim", 0.0)
    base_lmp = res.get("base_calibrated_lmp", 1.0)
    final_lmp = res.get("final_predicted_lmp", 1.0)
    pres_dec = res.get("preservation_decision", "N/A")
    err_type = res.get("legal_error_type", "none")
    rationale = res.get("legal_rationale", "N/A")
    invariants = res.get("invariant_checks", "N/A")

    # =====================================================================
    # 1. Primary Requested JUDGEBERT LMP Evaluation Box
    # =====================================================================
    print("\n=====================================================================================")
    print(" JUDGEBERT LEGAL MEANING PRESERVATION (LMP) EVALUATION ")
    print("=====================================================================================")
    print(f"Original Text        : {en_text}")
    print(f"Candidate (Hindi)    : {hi_text}")
    print("-------------------------------------------------------------------------------------")
    print(f"TF IDF ,COSINE SIMILARITY  LMP    : {raw_cos:.4f}  (Topic / Lexical Semantic Overlap)")
    print(f"SENTENCE BERT LMP  : {base_lmp:.2f} / 10.0")
    print(f"FINAL PREDICTED LMP  : {final_lmp:.2f} / 10.0")
    print(f"Preservation Decision: [ {pres_dec} ]")
    print(f"Legal Error Type     : {err_type}")
    print(f"Legal Rationale      : {rationale}")
    print(f"Invariant Checks     : {invariants}")
    print("=====================================================================================")

    if not verbose:
        return

    # =====================================================================
    # 2. ALL MODELS BENCHMARK COMPARISON TABLE (Verbose Mode)
    # =====================================================================
    print("\n=====================================================================================")
    print("                       ALL MODELS BENCHMARK COMPARISON                               ")
    print("=====================================================================================")
    header = f"{'Baseline / Model':<35}{'LMP Score':<16}{'Active Loaded Backbone':<33}"
    print(header)
    print("-" * 84)

    display_baselines = [
        "TF-IDF",
        "Cosine Similarity",
        "Sentence-BERT",
        "MiniLM Transformer LMP",
        "JUDGEBERT-X (XLM-R)"
    ]

    for model_name in display_baselines:
        info = res["models"].get(model_name)
        if not info:
            continue
        score = info.get("score")
        if score is not None:
            score_str = f"{score:.2f} / 10"
        else:
            score_str = "UNAVAILABLE"

        backbone = info.get("active_backbone", "N/A")
        if len(backbone) > 31:
            backbone_disp = backbone[:28] + "..."
        else:
            backbone_disp = backbone

        print(f"{model_name:<35}{score_str:<16}{backbone_disp:<33}")

    print("-" * 84)
    print(f"Final Selected Prediction Model : {res.get('final_model_name', 'MiniLM Transformer LMP')}")
    print("=====================================================================================")

    # =====================================================================
    # 3. Model Status & Lineage Details
    # =====================================================================
    print("\nModel Status & Lineage Details:")
    idx = 1
    for model_name in display_baselines:
        info = res["models"].get(model_name)
        if not info:
            continue
        tr_st = info.get("training_status", "UNKNOWN")
        cal_st = info.get("calibration_status", "UNCALIBRATED")
        bb = info.get("active_backbone", "N/A")
        tgt_bb = info.get("target_backbone", "N/A")
        sc_type = info.get("score_type", "Experimental Score")
        print(f"  [{idx}] {model_name}:")
        print(f"      • Active Backbone:   {bb}")
        if bb != tgt_bb and tgt_bb != "N/A":
            print(f"      • Target Backbone:   {tgt_bb}")
        print(f"      • Training Status:   {tr_st}")
        print(f"      • Calibration:       {cal_st}")
        print(f"      • Score Type:        {sc_type}")
        idx += 1

    # =====================================================================
    # 4. Experimental Score Warning Banner
    # =====================================================================
    print("\n=====================================================================================")
    print("⚠️  EXPERIMENTAL SCORE WARNING & ZERO-FABRICATION NOTICE")
    print("=====================================================================================")
    print("• There are currently 0 genuine human-annotated English-Hindi training pairs.")
    print("• All numerical scores above are EXPERIMENTAL ZERO-SHOT SIMILARITY/ALIGNMENT scores.")
    print("• They are NOT validated legal preservation ratings.")
    print("• No supervised fine-tuning or regression training has occurred on human labels.")
    print("• In offline sandbox mode, XLM-RoBERTa pretrained weights (~1.1GB) are unavailable;")
    print("  JUDGEBERT-X truthfully reports UNAVAILABLE rather than substituting MiniLM.")

    # =====================================================================
    # 5. Detailed Legal Meaning Change Analysis
    # =====================================================================
    explain_data = res.get("explainability", {})
    findings = explain_data.get("findings", [])
    counts = explain_data.get("counts", {})
    total_issues = explain_data.get("total_issues", 0)

    if findings:
        print("\n=====================================================================================")
        print("               LEGAL MEANING CHANGE ANALYSIS (10 CATEGORIES)                         ")
        print("=====================================================================================")
        for idx, item in enumerate(findings, start=1):
            print(f"[{idx}] {item['category']}")
            print(f"Status:      {item['status']}")
            if item.get("english"):
                print(f"English:     {item['english']}")
            if item.get("hindi"):
                print(f"Hindi:       {item['hindi']}")
            print(f"Explanation: {item['explanation']}")
            if item.get("overlap_note"):
                print(f"Overlap:     {item['overlap_note']}")
            print()

    # =====================================================================
    # 6. Final Category-Wise Summary
    # =====================================================================
    print("=====================================================================================")
    print("                       CATEGORY-WISE SUMMARY                                         ")
    print("=====================================================================================")
    print(f"1.  Number & Percentage Changes:    {counts.get('number_changes', 0)}")
    print(f"2.  Negation Reversals:             {counts.get('negation_changes', 0)}")
    print(f"3.  Modality & Duty Changes:        {counts.get('modality_changes', 0)}")
    print(f"4.  Entity / Party Swaps:           {counts.get('entity_changes', 0)}")
    print(f"5.  Condition Omissions:            {counts.get('condition_omissions', 0)}")
    print(f"6.  Exception Removals:             {counts.get('exception_removals', 0)}")
    print(f"7.  Scope Divergences:              {counts.get('scope_changes', 0)}")
    print(f"8.  Date & Temporal Changes:        {counts.get('temporal_changes', 0)}")
    print(f"9.  Hallucinated Information:       {counts.get('hallucinated_information', 0)}")
    print(f"10. Clause / Consequence Omissions: {counts.get('omissions', 0)}")
    print("-" * 75)
    print(f"Total Potential Legal Meaning Issues Flagged: {total_issues}")

    # =====================================================================
    # 7. Detection Limitation & Legal Review Warning
    # =====================================================================
    print("\n=====================================================================================")
    print("⚖️  DETECTION LIMITATION & LEGAL EXPERT REVIEW WARNING")
    print("=====================================================================================")
    print("• " + explain_data.get("detection_limitation_disclaimer", (
        "Rule-based detection matches lexical keywords, regex patterns, numerals, and bilingual lexicon entries. "
        "It does not possess full semantic or jurisdictional legal reasoning capability."
    )))
    print("• These automated screening findings are NOT a substitute for formal legal review.")
    print("• Always consult a qualified advocate or legal counsel for statutory compliance.")
    print("-------------------------------------------------------------------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="English-to-Hindi Legal Meaning Preservation (LMP) Evaluation")
    parser.add_argument("--original", "--english", dest="original", type=str, help="Original English legal sentence")
    parser.add_argument("--candidate", "--hindi", dest="candidate", type=str, help="Candidate Hindi translated legal sentence")
    parser.add_argument("--verbose", "-v", action="store_true", help="Display extended benchmark comparison table and lineage details")
    parser.add_argument("--sanity-checks", action="store_true", help="Execute the 10 English-to-Hindi Sanity Checks (Task 7)")
    args = parser.parse_args()

    if args.sanity_checks:
        from run_sanity_checks import run_all_sanity_checks
        run_all_sanity_checks()
        return

    print_startup_notice()

    english_text = args.original
    hindi_text = args.candidate

    if not english_text:
        try:
            english_text = input("Enter English legal sentence:\n").strip()
        except EOFError:
            english_text = ""

    if not hindi_text:
        try:
            hindi_text = input("\nEnter Hindi translated sentence:\n").strip()
        except EOFError:
            hindi_text = ""

    if not english_text or not hindi_text:
        print("\n[Error] Both English and Hindi sentences must be provided.")
        sys.exit(1)

    predictor = EnglishHindiLMPPredictor()
    results = predictor.predict_all(english_text, hindi_text)
    print_results(results, verbose=args.verbose)


if __name__ == "__main__":
    main()

