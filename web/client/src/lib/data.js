// Published figures used on the Overview page. Every number carries its source (see README §1 and §23).
export const SOURCES = {
  chms: { label: "Statistics Canada, CHMS 2022-2024 (The Daily, 2 Oct 2025)", url: "https://www150.statcan.gc.ca/n1/daily-quotidien/251002/dq251002b-eng.htm" },
  infog: { label: "Statistics Canada infographic 11-627-M (2026)", url: "https://publications.gc.ca/collections/collection_2026/statcan/11-627-m/11-627-m2026019-eng.pdf" },
  cchs: { label: "Statistics Canada, CCHS 2023/2024 (Jan 2026)", url: "https://www.statcan.gc.ca/o1/en/plus/8956-look-moods-and-mental-health-older-canadians" },
  cihi: { label: "CIHI, community mental health counselling wait times (2023-24 data)", url: "https://www.cihi.ca/en/indicators/wait-times-for-community-mental-health-counselling" },
  ourcare: { label: "OurCare National Survey 2025 (Unity Health Toronto / CMA)", url: "https://unityhealth.to/2025/12/access-to-primary-care-growing/" },
  meta: { label: "BJPsych Open meta-review: obesity and depression in adults", url: "https://www.cambridge.org/core/journals/bjpsych-open/article/association-between-obesity-and-depression-in-adults-a-metareview/68BB5A02ED0A342F2BB518421D6BE0D1" },
  canmat: { label: "CANMAT 2023 / UBC release (May 2024)", url: "https://med.ubc.ca/?p=71100" },
  phac988: { label: "PHAC, 9-8-8 renewed support (Jan 2026); CTV/Canadian Press", url: "https://www.canada.ca/en/public-health/news/2026/01/government-of-canada-announces-renewed-support-for-the-9-8-8-suicide-crisis-helpline.html" },
  levis: { label: "Levis et al., JAMA 2020 IPD meta-analysis (44 studies, n = 10,627)", url: "https://eprints.whiterose.ac.uk/id/eprint/163101/" },
};

// Adult obesity trend. 2000 & 2024 from the infographic ("1 in 5" / "1 in 3"); 2016-19 & 2022-24 measured CHMS.
export const OBESITY_TREND = { labels: ["2000", "2016-19", "2022-24"], values: [20, 25, 33] };
export const OBESITY_BY_AGE = { categories: ["40-59", "60-79"], male: [38, 35], female: [35, 33] }; // CHMS 2022-24
export const PERCEIVED_MH = [{ label: "Adults 18+", value: 15.1 }, { label: "Adults 65+", value: 8.1 }]; // fair/poor, CCHS 23/24
export const PRIMARY_CARE = { categories: ["2022", "2025"], values: [6.5, 5.9] }; // millions without regular provider
export const OR_ROWS = [{ label: "Range across 9 meta-analyses", lo: 1.18, hi: 1.57 }];

// Research gaps from recent literature and how MapleMind answers each one.
export const GAPS = [
  {
    gap: "Consumer mental-health answers contain hallucinations and omissions",
    evidence: "A 2,075-query, human-annotated study of Llama 3.3 found both error types vary with who asks, the topic, context and tone.",
    src: "Ni et al., arXiv:2604.00014 (2026)",
    response: "Claim-level verifier: every factual sentence must cite a passage, and every number must appear in it. Unsupported sentences are removed or the system abstains.",
  },
  {
    gap: "LLM judges miss most hallucinations in this domain",
    evidence: "LLM-as-judge reached F1 = 0.169 for hallucination detection in mental-health responses; a blended human + LLM method reached 0.717.",
    src: "arXiv:2604.06216 (2026)",
    response: "Deterministic checks are the release gate. The judge is advisory, from a different model family, and only trusted after Cohen's κ against human ratings.",
  },
  {
    gap: "Few chatbots are tested across safety axes before deployment",
    evidence: "Only 16% of LLM-based chatbot interventions had rigorous clinical efficacy testing; authors propose a five-axis framework.",
    src: "arXiv:2604.23445 (2026)",
    response: "Five-axis suite: protocol fidelity, hallucination risk, behavioural consistency, crisis safety, demographic robustness. It found and fixed a third-person crisis gap.",
  },
  {
    gap: "Safety must be monitored in real conversations, not only benchmarks",
    evidence: "An analysis of 20,000 real conversations argues for continuous, deployment-relevant safety assurance.",
    src: "arXiv:2601.17003 (2026)",
    response: "Per-agent latency and outcome telemetry (escalations, abstentions) with PII-redacted traces and an SLO, visible on the Monitoring page.",
  },
  {
    gap: "Screening burden vs accuracy",
    evidence: "PHQ-2 ≥ 2 then PHQ-9 ≥ 10: sensitivity 0.82, specificity 0.87, AUC 0.90, comparable to the full PHQ-9.",
    src: "Levis et al., JAMA 2020",
    response: "Two-stage gated check-in: most people answer 4 questions, hard budget of 12, item 9 triggers the crisis protocol immediately.",
  },
  {
    gap: "BMI alone misses weight-related health risk",
    evidence: "Canadian guideline: BMI is not accurate for adiposity-related complications; care should focus on health outcomes, not weight loss alone.",
    src: "Wharton et al., CMAJ 2020",
    response: "No calorie targets or weight-loss plans; plate-balance feedback from the Food Guide, person-first language, eating-disorder safeguards.",
  },
];
