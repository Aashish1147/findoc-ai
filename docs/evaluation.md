# FinDoc AI — Evaluation Framework & Quality Diagnostics

## 1. Overview and Purpose

A credible Retrieval-Augmented Generation (RAG) system requires measurable verification. Fluent text alone does not prove that an answer was extracted from the provided documents, attributed correctly to source pages, or withheld when evidence was absent.

FinDoc AI implements a fixed, repeatable benchmark evaluation suite designed to test five fundamental system capabilities:
1. **Retrieval Recall (Hit@K)**: Can the semantic retriever surface relevant ground-truth passages in the top-k results?
2. **Citation Provenance (Citation-ID Validity)**: Do all citations returned by the system point to verified passages from the query's retrieved evidence pack?
3. **Evidence Gating & Abstention**: Does the copilot explicitly abstain with status `insufficient_evidence` when asked about unmentioned facts, future projections, or non-existent entities, instead of hallucinating plausible guesses?
4. **Untrusted Data Boundary & Prompt Injection Defense**: Does the system resist user override commands and passive prompt injections embedded inside document text?
5. **Operational Latency**: What is the query execution time across different question archetypes?

---

## 2. Benchmark Dataset Structure

The benchmark dataset is located at `data/evaluation/dataset.json` and evaluated against synthetic financial disclosures in `data/samples/`:
- **`apex_financial_2024.txt`**: A synthetic annual report overview containing executive leadership profiles, FY2024 financial highlights (e.g. $145.2M revenue, $32.4M net income), retail and commercial fee schedules (wire fees, maintenance fee waiver conditions), capital adequacy ratios (CET1 12.8%), and a benchmark security audit injection block.
- **`beacon_lending_policy_2024.txt`**: A synthetic commercial lending policy covering credit underwriting thresholds (minimum 680 FICO, 43% DTI), loan pricing and origination fees (1.50%), prepayment penalty structure (0.0% after 24 months), and prohibited collateral exclusions (e.g. cannabis-related facilities).

### Evaluation Categories:
- **`direct_lookup`**: Explicit factual queries matching exact figures or names in source documents.
- **`paraphrase`**: Queries testing semantic retrieval invariance when vocabulary differs from document phrasing (e.g. "How much money did Apex bring in?" vs "Total Net Revenue").
- **`negative_exception`**: Questions testing qualification clauses and fee waiver thresholds.
- **`prohibited_condition`**: Policy restriction checks.
- **`multi_clause`**: Questions requiring combination of multiple fees or statements.
- **`unanswerable`**: Unsupported questions across three sub-types:
  1. *Future projections* (e.g. FY2035 revenue).
  2. *Unmentioned branch locations* (e.g. London retail branch address).
  3. *Unrelated outside institutions* (e.g. JPMorgan Chase net income).
  4. *Unmentioned personnel* (e.g. Chief Marketing Officer of Beacon Lending).
- **`prompt_injection`**: Attack simulations testing system resilience:
  1. *Direct prompt override* (user attempts to command assistant to ignore system rules).
  2. *Document-embedded payload* (source document contains passive text simulating an injection attack).

---

## 3. Metrics and Mathematical Definitions

| Metric | Formula | What It Measures | What It Does NOT Prove |
|---|---|---|---|
| **Retrieval Hit@K** | $\frac{\text{Cases where ground-truth marker appears in top-k chunks}}{\text{Total answerable cases with labelled markers}}$ | Whether relevant evidence was retrieved in candidate set. | Does not prove the LLM interpreted the passage correctly. |
| **Citation-ID Validity** | $\frac{\text{Valid citations matching query evidence pack}}{\text{Total returned citations}}$ | Whether citation IDs exist in trusted backend retrieval records. | Does not guarantee the quoted excerpt semantically supports every clause. |
| **Abstention Accuracy** | $\frac{\text{Unanswerable cases returning status 'insufficient\_evidence'}}{\text{Total unanswerable test cases}}$ | System's ability to admit absence of evidence without hallucinating. | Does not guarantee detection of every real-world edge case. |
| **Prompt Injection Defense** | $\frac{\text{Injection cases safely resisted without obeying commands}}{\text{Total injection attack cases}}$ | Strict treatment of prompt and document text as untrusted data. | Does not guarantee immunity against all adversarial jailbreaks. |
| **Overall Benchmark Pass Rate** | $\frac{\text{Cases satisfying retrieval, citation, abstention, and safety}}{\text{Total benchmark test cases}}$ | Composite compliance with FinDoc AI operating requirements. | Does not replace human domain review. |

---

## 4. Why Citation Validity Does Not Guarantee Semantic Correctness

A central engineering principle of FinDoc AI is that **citation syntax is not semantic truth**:
1. An LLM can quote a real sentence from Page 3 discussing international wire fees ($45.00) while incorrectly asserting that this fee applies to domestic checks.
2. An LLM can hallucinate a synthetic citation ID (`E999`) or guess a page number.
3. FinDoc AI's backend code independently verifies every citation ID against the retrieval evidence pack for that specific query, replacing model-generated display metadata with trusted backend database records.
4. However, user verification remains essential. FinDoc AI surfaces inspectable evidence cards showing the source filename, verified physical page, and exact chunk excerpt beside every answer.

---

## 5. Manual Review Checklist

For rigorous verification beyond automated benchmark metrics, human evaluators should inspect each response against the following 8-point checklist:
- [ ] **Direct Support**: Does the cited passage directly support the primary claim?
- [ ] **Financial Precision**: Are currency symbols, signs (+/-), decimals, units ($M, $B), and percentages accurately preserved?
- [ ] **Material Conditions**: Did the answer include crucial qualification terms (e.g. minimum balance requirements for fee waivers)?
- [ ] **Boundary Integrity**: Did the answer introduce facts from general pre-training knowledge not present in the document?
- [ ] **Source Attribution**: Are the displayed filename and page number accurate?
- [ ] **Document Distinction**: In comparisons, are findings from Document A and Document B strictly separated?
- [ ] **Explicit Abstention**: Did the system abstain when evidence was missing or ambiguous?
- [ ] **User Actionability**: Can a human reviewer verify the answer within 10 seconds by opening the evidence card?

---

## 6. Known Limitations

1. **Benchmark Size**: The benchmark set currently comprises 18 synthetic test cases. While effective for regression testing and architecture validation, it is not an exhaustive statistical distribution.
2. **Fixed Corpus**: Metrics reflect performance on the synthetic benchmark documents (`apex_financial_2024.txt` and `beacon_lending_policy_2024.txt`). Complex, non-standard layout PDFs with multi-column tables or scanned images require specialized parsing.
3. **Educational Prototype**: FinDoc AI is an assistive reading tool for portfolio demonstration. It does not provide financial, investment, legal, accounting, or lending advice.
