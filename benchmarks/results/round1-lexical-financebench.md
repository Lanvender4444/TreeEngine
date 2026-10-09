# TreeEngine retrieval benchmark — financebench

- corpus: 84 documents (84 real, 20 with 100+ nodes), 5976 nodes, 83789 blocks; ingest 754.4s (cached)
- queries: 150 (heldout 150; domain-relevant 50, metrics-generated 50, novel-generated 50)
- strategies: fts, rag_bm25, tree_structure, tree_lexical, tree_lexical+fts, managed
- skipped: none
- tokens counted with tiktoken cl100k_base; context = top 5 evidence items
- traditional RAG chunks: 600 tokens, 100 overlap (tiktoken cl100k_base)

recall@k = share of expected evidence (needles or pages) in the top k · node_recall@5 = evidence from the expected section in the top 5 · doc_recall@5 = corpus-wide queries: evidence from a document holding the answer in the top 5 · ctx_tokens@5 = tokens an answer model would read · recall@1k_tok / @2k_tok = recall within the first 1,000 / 2,000 tokens of evidence (equal reading cost for chunks and blocks) · tokens/success = LLM tokens per query answered within the top 5.

## Overall

| strategy | family | n | recall@1 | recall@3 | recall@5 | recall@10 | mrr | recall@1k_tok | recall@2k_tok | node_recall@5 | doc_recall@5 | ctx_tokens@5 | p50_ms | p95_ms | embed_calls/q | llm_tokens/q | tokens/success |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fts | treeengine | 150 | 17.0 | 27.3 | 34.7 | 43.7 | 27.3 | 35.0 | 46.0 | – | – | 824 | 38.7 | 67.3 | 0.0 | 0.0 | – |
| rag_bm25 | traditional | 150 | 20.2 | 29.9 | 35.9 | 46.6 | 29.6 | 20.2 | 29.9 | – | – | 2,982 | 33.0 | 52.3 | 0.0 | 0.0 | – |
| tree_structure | treeengine | 150 | 10.7 | 15.3 | 16.3 | 17.3 | 13.7 | 15.3 | 17.3 | – | – | 669 | 9.9 | 28.8 | 0.0 | 0.0 | – |
| tree_lexical | treeengine | 150 | 12.7 | 19.3 | 20.9 | 24.2 | 16.6 | 20.2 | 25.1 | – | – | 875 | 54.5 | 98.5 | 0.0 | 0.0 | – |
| tree_lexical+fts | treeengine | 150 | 16.7 | 27.0 | 32.0 | 38.7 | 24.7 | 32.0 | 38.9 | – | – | 842 | 82.3 | 136 | 0.0 | 0.0 | – |
| managed | treeengine | 150 | 17.0 | 27.0 | 35.0 | 43.3 | 27.1 | 34.7 | 45.7 | – | – | 838 | 33.8 | 63.9 | 0.0 | 0.0 | – |

## Significance (paired sign test on recall@5)

| strategy | W:L vs fts (recall@5) | sign test p |
|---|---|---|
| rag_bm25 | 18:15 | 0.728 |
| tree_structure | 11:42 | 0.000 ** |
| tree_lexical | 7:32 | 0.000 ** |
| tree_lexical+fts | 5:11 | 0.210 |
| managed | 1:1 | 1.000 |

## recall@5 by query type

| type | n | fts | rag_bm25 | tree_structure | tree_lexical | tree_lexical+fts | managed |
|---|---|---|---|---|---|---|---|
| domain-relevant | 50 | 18.0 | 20.7 | 10.0 | 14.7 | 14.0 | 18.0 |
| metrics-generated | 50 | 24.0 | 29.0 | 11.0 | 2.0 | 20.0 | 23.0 |
| novel-generated | 50 | 62.0 | 58.0 | 28.0 | 46.0 | 62.0 | 64.0 |

## recall@5 by category

| category | n | fts | rag_bm25 | tree_structure | tree_lexical | tree_lexical+fts | managed |
|---|---|---|---|---|---|---|---|
| information extraction | 31 | 32.3 | 38.7 | 22.6 | 19.4 | 19.4 | 32.3 |
| logical reasoning | 2 | 0.0 | 0.0 | 0.0 | 16.7 | 0.0 | 0.0 |
| numerical reasoning | 67 | 16.4 | 19.2 | 5.2 | 3.0 | 16.4 | 15.7 |
| unlabelled | 50 | 62.0 | 58.0 | 28.0 | 46.0 | 62.0 | 64.0 |

## MRR by query type

| type | n | fts | rag_bm25 | tree_structure | tree_lexical | tree_lexical+fts | managed |
|---|---|---|---|---|---|---|---|
| domain-relevant | 50 | 13.8 | 14.7 | 7.6 | 10.0 | 11.5 | 13.9 |
| metrics-generated | 50 | 23.1 | 24.5 | 10.4 | 2.2 | 19.1 | 22.3 |
| novel-generated | 50 | 45.0 | 49.4 | 23.1 | 37.4 | 43.7 | 45.1 |

## By document length

recall@5 by document pages:

| pages | n | fts | rag_bm25 | tree_structure | tree_lexical | tree_lexical+fts | managed |
|---|---|---|---|---|---|---|---|
| < 50 pages | 24 | 77.1 | 87.5 | 45.8 | 58.3 | 79.2 | 77.1 |
| 50–150 | 60 | 30.0 | 30.8 | 15.0 | 15.0 | 24.2 | 29.2 |
| 150–400 | 60 | 23.3 | 20.6 | 7.5 | 13.9 | 23.3 | 25.0 |
| 400+ | 6 | 25.0 | 33.3 | 0.0 | 0.0 | 8.3 | 25.0 |

context tokens (top 5) by document pages:

| pages | n | fts | rag_bm25 | tree_structure | tree_lexical | tree_lexical+fts | managed |
|---|---|---|---|---|---|---|---|
| < 50 pages | 24 | 625 | 2,908 | 711 | 770 | 643 | 640 |
| 50–150 | 60 | 895 | 2,993 | 688 | 918 | 891 | 901 |
| 150–400 | 60 | 836 | 3,000 | 678 | 883 | 882 | 857 |
| 400+ | 6 | 798 | 3,000 | 213 | 777 | 758 | 798 |

recall@5 by document nodes:

| nodes | n | fts | rag_bm25 | tree_structure | tree_lexical | tree_lexical+fts | managed |
|---|---|---|---|---|---|---|---|
| ≤ 20 nodes | 60 | 45.8 | 50.0 | 23.3 | 31.7 | 48.3 | 45.8 |
| 21–60 | 24 | 33.3 | 27.1 | 12.5 | 8.3 | 25.0 | 31.2 |
| 61–100 | 19 | 28.9 | 34.2 | 15.8 | 5.3 | 7.9 | 28.9 |
| 101–200 | 22 | 22.7 | 13.6 | 4.5 | 24.2 | 31.8 | 27.3 |
| 200+ | 25 | 24.0 | 31.3 | 14.0 | 16.0 | 18.0 | 24.0 |

context tokens (top 5) by document nodes:

| nodes | n | fts | rag_bm25 | tree_structure | tree_lexical | tree_lexical+fts | managed |
|---|---|---|---|---|---|---|---|
| ≤ 20 nodes | 60 | 924 | 2,963 | 795 | 957 | 934 | 936 |
| 21–60 | 24 | 814 | 2,983 | 676 | 901 | 853 | 814 |
| 61–100 | 19 | 781 | 2,999 | 491 | 820 | 762 | 790 |
| 101–200 | 22 | 724 | 3,000 | 619 | 820 | 790 | 755 |
| 200+ | 25 | 716 | 3,000 | 538 | 740 | 718 | 734 |

recall@5 by document tokens:

| tokens | n | fts | rag_bm25 | tree_structure | tree_lexical | tree_lexical+fts | managed |
|---|---|---|---|---|---|---|---|
| < 10k tokens | 9 | 100.0 | 100.0 | 33.3 | 55.6 | 100.0 | 100.0 |
| 10k–50k | 20 | 65.0 | 80.0 | 45.0 | 55.0 | 67.5 | 65.0 |
| 50k–150k | 88 | 25.0 | 24.4 | 10.2 | 12.9 | 21.0 | 25.6 |
| 150k+ | 33 | 24.2 | 22.2 | 10.6 | 12.1 | 21.2 | 24.2 |

context tokens (top 5) by document tokens:

| tokens | n | fts | rag_bm25 | tree_structure | tree_lexical | tree_lexical+fts | managed |
|---|---|---|---|---|---|---|---|
| < 10k tokens | 9 | 532 | 2,757 | 420 | 476 | 532 | 511 |
| 10k–50k | 20 | 731 | 2,999 | 782 | 951 | 752 | 758 |
| 50k–150k | 88 | 878 | 2,995 | 694 | 914 | 888 | 888 |
| 150k+ | 33 | 819 | 3,000 | 600 | 831 | 859 | 840 |


## Tree metrics

| strategy | target_recall | visited_ratio | depth reached | nodes expanded |
|---|---|---|---|---|
| tree_structure | – | 81.2 | 1.2 | 2.1 |
| tree_lexical | – | 84.1 | 1.4 | 2.6 |
| tree_lexical+fts | – | 84.1 | 1.4 | 2.6 |
| managed | – | 79.5 | 1.6 | 3.0 |

## Planner evaluation

Oracles pick, per query, the strategy that did best against the ground truth (evaluation only): the ceiling for a perfect router among the listed strategies.

| strategy | n | recall@1 | recall@3 | recall@5 | recall@10 | mrr | recall@1k_tok | recall@2k_tok | node_recall@5 | doc_recall@5 |
|---|---|---|---|---|---|---|---|---|---|---|
| always_fts | 150 | 17.0 | 27.3 | 34.7 | 43.7 | 27.3 | 35.0 | 46.0 | – | – |
| always_tree | 150 | 12.7 | 19.3 | 20.9 | 24.2 | 16.6 | 20.2 | 25.1 | – | – |
| always_tree→fts | 150 | 16.7 | 27.0 | 32.0 | 38.7 | 24.7 | 32.0 | 38.9 | – | – |
| rule_planner | 150 | 17.0 | 27.0 | 35.0 | 43.3 | 27.1 | 34.7 | 45.7 | – | – |
| oracle(fts|tree|tree→fts) | 150 | 20.3 | 33.3 | 39.9 | 48.2 | 31.4 | 39.9 | 49.8 | – | – |
| oracle(planner choices) | 150 | 17.0 | 29.3 | 37.7 | 46.7 | 28.2 | 37.0 | 48.2 | – | – |
| oracle(all strategies) | 150 | 31.6 | 43.6 | 49.4 | 60.1 | 42.9 | 47.1 | 56.3 | – | – |

recall@5 by query type:

| type | n | always_fts | always_tree | always_tree→fts | rule_planner | oracle(fts|tree|tree→fts) | oracle(planner choices) | oracle(all strategies) |
|---|---|---|---|---|---|---|---|---|
| domain-relevant | 50 | 18.0 | 14.7 | 14.0 | 18.0 | 24.7 | 20.0 | 27.3 |
| metrics-generated | 50 | 24.0 | 2.0 | 20.0 | 23.0 | 25.0 | 25.0 | 46.0 |
| novel-generated | 50 | 62.0 | 46.0 | 62.0 | 64.0 | 70.0 | 68.0 | 75.0 |

## Index cost

Cost of building everything a strategy needs, as if deployed alone. Embedding tokens are exact; seconds are this run's (a cached index costs ~0 s here; the corpus row is the original ingest time when recorded).

| strategy | indexes | units | seconds | embed calls | embed tokens | LLM calls | MB | est. cost |
|---|---|---|---|---|---|---|---|---|
| fts | corpus | 83,789 | 754.4 | 0 | 0 | 0 | 226.5 | – |
| rag_bm25 | chunks | 18,007 | 19.5 | 0 | 0 | 0 | 48.7 | – |
| tree_structure | corpus | 83,789 | 754.4 | 0 | 0 | 0 | 226.5 | – |
| tree_lexical | corpus | 83,789 | 754.4 | 0 | 0 | 0 | 226.5 | – |
| tree_lexical+fts | corpus | 83,789 | 754.4 | 0 | 0 | 0 | 226.5 | – |
| managed | corpus | 83,789 | 754.4 | 0 | 0 | 0 | 226.5 | – |

## Head-to-head at k=5

| strategy | wins vs fts | losses vs fts | both miss |
|---|---|---|---|
| rag_bm25 | 15 (fb-00464, fb-00499, fb-00995, fb-01484, fb-01902, fb-01928, +9) | 15 (fb-00299, fb-00394, fb-00702, fb-01858, fb-01912, fb-01936, +9) | 78 |
| tree_structure | 11 (fb-01028, fb-01198, fb-01484, fb-01928, fb-01964, fb-03473, +5) | 42 (fb-00288, fb-00299, fb-00394, fb-00407, fb-00460, fb-00601, +36) | 82 |
| tree_lexical | 7 (fb-00464, fb-00494, fb-00995, fb-01198, fb-01290, fb-01902, +1) | 32 (fb-00288, fb-00299, fb-00394, fb-00601, fb-00603, fb-00605, +26) | 86 |
| tree_lexical+fts | 5 (fb-00464, fb-00494, fb-01346, fb-01928, fb-04735) | 11 (fb-00724, fb-00746, fb-01254, fb-01328, fb-01912, fb-02024, +5) | 88 |
| managed | 1 (fb-00464) | 1 (fb-04412) | 92 |

## Missed by every strategy at k=5

- `fb-00005` [domain-relevant] Does Corning have positive working capital based on FY2022 data? If working capital is not a useful or relevant metric for this company, then please state that and explain why.
- `fb-00070` [domain-relevant] Does American Water Works have positive working capital based on FY2022 data? If working capital is not a useful or relevant metric for this company, then please state that and explain why.
- `fb-00080` [domain-relevant] Does Paypal have positive working capital based on FY2022 data? If working capital is not a useful or relevant metric for this company, then please state that and explain why.
- `fb-00206` [domain-relevant] Are JPM's gross margins historically consistent (not fluctuating more than roughly 2% each year)? If gross margins are not a relevant metric for a company like this, then please state that and explain why.
- `fb-00215` [domain-relevant] Is Verizon a capital intensive business based on FY 2022 data?
- `fb-00216` [domain-relevant] Does Verizon have a reasonably healthy liquidity profile based on its quick ratio for FY 2022? If the quick ratio is not relevant to measure liquidity, please state that and explain why.
- `fb-00222` [domain-relevant] Does AMD have a reasonably healthy liquidity profile based on its quick ratio for FY22? If the quick ratio is not relevant to measure liquidity, please state that and explain why.
- `fb-00283` [novel-generated] How much does Pfizer expect to pay to spin off Upjohn in the future in USD million?
- `fb-00302` [novel-generated] Did Pfizer grow its PPNE between FY20 and FY21?
- `fb-00438` [domain-relevant] Does Adobe have an improving operating margin profile as of FY2022? If operating margin is not a useful metric for a company like this, then state that and explain why.
- `fb-00517` [domain-relevant] Are there any product categories / service categories that represent more than 20% of Boeing's revenue for FY2022?
- `fb-00521` [domain-relevant] What are major acquisitions that Ulta Beauty has done in FY2023 and FY2022?
- `fb-00540` [domain-relevant] Roughly how many times has AES Corporation sold its inventory in FY2022? Calculate inventory turnover ratio for the FY2022; if conventional inventory management is not meaningful for the company then state that and explain why.
- `fb-00552` [domain-relevant] Has Microsoft increased its debt on balance sheet between FY2023 and the FY2022 period?
- `fb-00563` [novel-generated] From FY21 to FY22, excluding Embedded, in which AMD reporting segment did sales proportionally increase the most?
- `fb-00566` [domain-relevant] Has Verizon increased its debt on balance sheet between 2022 and the 2021 fiscal period?
- `fb-00585` [novel-generated] How does Boeing's effective tax rate in FY2022 compare to FY2021?
- `fb-00591` [novel-generated] Does Adobe have an improving Free cashflow conversion as of FY2022?
- `fb-00669` [domain-relevant] What drove gross margin change as of FY2022 for JnJ? If gross margin is not a useful metric for a company like this, then please state that and explain why.
- `fb-00678` [domain-relevant] Does Boeing have an improving gross margin profile as of FY2022? If gross margin is not a useful metric for a company like this, then state that and explain why.
- `fb-00684` [domain-relevant] Does AMCOR have an improving gross margin profile as of FY2023? If gross margin is not a useful metric for a company like this, then state that and explain why.
- `fb-00685` [domain-relevant] Are Best Buy's gross margins historically consistent (not fluctuating more than roughly 2% each year)? If gross margins are not a relevant metric for a company like this, then please state that and explain why.
- `fb-00711` [domain-relevant] Roughly how many times has JnJ sold its inventory in FY2022? Calculate inventory turnover ratio for FY2022; if conventional inventory management is not meaningful for the company then state that and explain why.
- `fb-00720` [domain-relevant] What drove gross margin change as of the FY2022 for American Express? If gross margin is not a useful metric for a company like this, then please state that and explain why.
- `fb-00723` [domain-relevant] Does AMEX have an improving operating margin profile as of 2022? If operating margin is not a useful metric for a company like this, then state that and explain why.
- `fb-00735` [domain-relevant] Has Pepsico reported any materially important ongoing legal battles from FY2022 and FY2021?
- `fb-00757` [novel-generated] Did AMD report customer concentration in FY22?
- `fb-00790` [domain-relevant] Is CVS Health a capital-intensive business based on FY2022 data?
- `fb-00799` [domain-relevant] Has AMCOR's quick ratio improved or declined between FY2023 and FY2022? If the quick ratio is not something that a financial analyst would ask about a company like this, then state that and explain why.
- `fb-00807` [domain-relevant] Does 3M have a reasonably healthy liquidity profile based on its quick ratio for Q2 of FY2023? If the quick ratio is not relevant to measure liquidity, please state that and explain why.
- `fb-00839` [novel-generated] Does Foot Locker's new CEO have previous CEO experience in a similar company to Footlocker?
- `fb-00917` [domain-relevant] What drove operating margin change as of the FY22 for AMD? If operating margin is not a useful metric for a company like this, then please state that and explain why.
- `fb-00956` [domain-relevant] Are JnJ's FY2022 financials that of a high growth company?
- `fb-01009` [domain-relevant] What are the geographies that Pepsico primarily operates in as of FY2022?
- `fb-01077` [domain-relevant] What are major acquisitions that Best Buy has done in FY2023, FY2022 and FY2021?
- `fb-01079` [domain-relevant] What are major acquisitions that AMCOR has done in FY2023, FY2022 and FY2021?
- `fb-01091` [domain-relevant] Has Boeing reported any materially important ongoing legal battles from FY2022?
- `fb-01107` [domain-relevant] Has CVS Health reported any materially important ongoing legal battles from 2022, 2021 and 2020?
- `fb-01148` [domain-relevant] What industry does AMCOR primarily operate in?
- `fb-01226` [domain-relevant] What drove operating margin change as of FY2022 for 3M? If operating margin is not a useful metric for a company like this, then please state that and explain why.
- `fb-01275` [domain-relevant] Among operations, investing, and financing activities, which brought in the most (or lost the least) cash flow for Best Buy in FY2023?
- `fb-01351` [domain-relevant] How much has the effective tax rate of American Express changed between FY2021 and FY2022?
- `fb-01865` [novel-generated] If we exclude the impact of M&A, which segment has dragged down 3M's overall growth in 2022?
- `fb-01930` [novel-generated] How much was the Real change in Sales for AMCOR in FY 2023 vs FY 2022, if we exclude the impact of FX movement, passthrough costs and one-off items?
- `fb-01981` [novel-generated] Was American Express able to retain card members during 2022?
- `fb-02119` [novel-generated] If JPM went bankrupted by the end by 2021 Q1 and liquidated all of its assets to pay its shareholders, how much could each shareholder get?
- `fb-02987` [metrics-generated] What is the FY2019 fixed asset turnover ratio for Activision Blizzard? Fixed asset turnover ratio is defined as: FY2019 revenue / (average PP&E between FY2018 and FY2019). Round your answer to two decimal places. Base your judgments on the information provided primarily in the statement of income and the statement of financial position.
- `fb-03069` [metrics-generated] Answer the following question as if you are an equity research analyst and have lost internet connection so you do not have access to financial metric providers. According to the details clearly outlined within the P&L statement and the statement of cash flows, what is the FY2015 depreciation and amortization (D&A from cash flow statement) % margin for AMD?
- `fb-03471` [metrics-generated] By drawing conclusions from the information stated only in the statement of financial position, what is General Mills's FY2020 working capital ratio? Define working capital ratio as total current assets divided by total current liabilities. Round your answer to two decimal places.
- `fb-03718` [metrics-generated] What is Lockheed Martin's 2 year total revenue CAGR from FY2020 to FY2022 (in units of percents and round to one decimal place)? Provide a response to the question by primarily using the statement of income.
- `fb-03838` [metrics-generated] What is the FY2019 - FY2020 total revenue growth rate for Block (formerly known as Square)? Answer in units of percents and round to one decimal place. Approach the question asked by assuming the standpoint of an investment banking analyst who only has access to the statement of income.
- `fb-04080` [metrics-generated] When primarily referencing the income statement and the statement of financial position, what is the FY2021 inventory turnover ratio for Nike? Inventory turnover ratio is defined as: (FY2021 COGS) / (average inventory between FY2020 and FY2021). Round your answer to two decimal places.
- `fb-04171` [metrics-generated] Basing your judgments off of the balance sheet, what is the year end FY2018 amount of accounts payable for MGM Resorts? Answer in USD millions.
- `fb-04302` [metrics-generated] We need to calculate a reasonable approximation (or exact number if possible) of a financial metric. Basing your judgment by information plainly provided in the statement of income, what is Nike's three year average of cost of goods sold as a % of revenue from FY2016 to FY2018? Answer in units of percents and round to one decimal place.
- `fb-04417` [metrics-generated] What is the year end FY2019 total amount of inventories for Best Buy? Answer in USD millions. Base your judgments on the information provided primarily in the balance sheet.
- `fb-04481` [metrics-generated] What is the FY2022 unadjusted EBITDA % margin for PepsiCo? Calculate unadjusted EBITDA using unadjusted operating income and D&A (from cash flow statement). Give a response to the question by relying on the details shown in the statement of cash flows and the P&L statement.
- `fb-04672` [metrics-generated] Assume that you are a public equities analyst. Answer the following question by primarily using information that is shown in the balance sheet: what is the year end FY2018 net PPNE for 3M? Answer in USD billions.
- `fb-04784` [metrics-generated] Based on the information provided primarily in the statement of income, what is the FY2018 - FY2019 change in unadjusted operating income % margin for Walmart? Answer in units of percents and round to one decimal place.
- `fb-04854` [metrics-generated] According to the information provided in the statement of cash flows, what is the FY2020 free cash flow (FCF) for General Mills? FCF here is defined as: (cash from operations - capex). Answer in USD millions.
- `fb-04980` [metrics-generated] What is the FY2021 capital expenditure amount (in USD billions) for PepsiCo? Respond to the question by assuming the perspective of an investment analyst who can only use the details shown within the statement of cash flows.
- `fb-05718` [metrics-generated] How much (in USD billions) did American Water Works pay out in cash dividends for FY2020? Compute or extract the answer by primarily using the details outlined in the statement of cash flows.
- `fb-06247` [metrics-generated] What is FY2018 days payable outstanding (DPO) for Walmart? DPO is defined as: 365 * (average accounts payable between FY2017 and FY2018) / (FY2018 COGS + change in inventory between FY2017 and FY2018). Round your answer to two decimal places. Please base your judgments on the information provided primarily in the statement of financial position and the P&L statement.
- `fb-06655` [metrics-generated] What is Amazon's FY2017 days payable outstanding (DPO)? DPO is defined as: 365 * (average accounts payable between FY2016 and FY2017) / (FY2017 COGS + change in inventory between FY2016 and FY2017). Round your answer to two decimal places. Address the question by using the line items and information shown within the balance sheet and the P&L statement.
- `fb-06741` [metrics-generated] What is the FY2018 - FY2020 3 year average unadjusted EBITDA % margin for Walmart? Define unadjusted EBITDA as unadjusted operating income + depreciation and amortization from the cash flow statement. Answer in units of percents and round to one decimal place. Calculate what was asked by utilizing the line items clearly shown in the P&L statement and the cash flow statement.
- `fb-07507` [metrics-generated] What is Adobe's year-over-year change in unadjusted operating income from FY2015 to FY2016 (in units of percents and round to one decimal place)? Give a solution to the question by using the income statement.
- `fb-08135` [metrics-generated] What is Amazon's year-over-year change in revenue from FY2016 to FY2017 (in units of percents and round to one decimal place)? Calculate what was asked by utilizing the line items clearly shown in the statement of income.
- `fb-08286` [metrics-generated] By drawing conclusions from the information stated only in the income statement, what is Amazon's FY2019 net income attributable to shareholders (in USD millions)?
- `fb-10130` [metrics-generated] Based on the information provided primarily in the balance sheet and the statement of income, what is FY2020 days payable outstanding (DPO) for Corning? DPO is defined as: 365 * (average accounts payable between FY2019 and FY2020) / (FY2020 COGS + change in inventory between FY2019 and FY2020). Round your answer to two decimal places.
- `fb-10420` [metrics-generated] Based on the information provided primarily in the statement of financial position and the statement of income, what is AES's FY2022 return on assets (ROA)? ROA is defined as: FY2022 net income / (average total assets between FY2021 and FY2022). Round your answer to two decimal places.

## Per query type, all metrics

#### domain-relevant

| strategy | n | recall@1 | recall@3 | recall@5 | recall@10 | mrr | recall@1k_tok | recall@2k_tok | node_recall@5 | doc_recall@5 | ctx_tokens@5 | p50_ms | p95_ms | embed_calls/q | llm_tokens/q | tokens/success |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fts | 50 | 8.0 | 14.0 | 18.0 | 24.0 | 13.8 | 16.0 | 28.0 | – | – | 804 | 36.4 | 64.8 | 0.0 | 0.0 | – |
| rag_bm25 | 50 | 6.7 | 14.7 | 20.7 | 22.7 | 14.7 | 6.7 | 14.7 | – | – | 2,992 | 32.2 | 50.6 | 0.0 | 0.0 | – |
| tree_structure | 50 | 6.0 | 10.0 | 10.0 | 10.0 | 7.6 | 10.0 | 10.0 | – | – | 604 | 11.2 | 23.8 | 0.0 | 0.0 | – |
| tree_lexical | 50 | 6.0 | 12.0 | 14.7 | 16.7 | 10.0 | 14.7 | 17.3 | – | – | 882 | 51.9 | 87.1 | 0.0 | 0.0 | – |
| tree_lexical+fts | 50 | 8.0 | 12.0 | 14.0 | 20.0 | 11.5 | 14.0 | 20.7 | – | – | 823 | 76.2 | 112 | 0.0 | 0.0 | – |
| managed | 50 | 8.0 | 14.0 | 18.0 | 24.0 | 13.9 | 16.0 | 28.0 | – | – | 836 | 31.5 | 68.4 | 0.0 | 0.0 | – |

#### metrics-generated

| strategy | n | recall@1 | recall@3 | recall@5 | recall@10 | mrr | recall@1k_tok | recall@2k_tok | node_recall@5 | doc_recall@5 | ctx_tokens@5 | p50_ms | p95_ms | embed_calls/q | llm_tokens/q | tokens/success |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fts | 50 | 11.0 | 19.0 | 24.0 | 35.0 | 23.1 | 25.0 | 37.0 | – | – | 952 | 51.5 | 70.6 | 0.0 | 0.0 | – |
| rag_bm25 | 50 | 14.0 | 24.0 | 29.0 | 41.0 | 24.5 | 14.0 | 24.0 | – | – | 3,000 | 39.6 | 56.6 | 0.0 | 0.0 | – |
| tree_structure | 50 | 6.0 | 10.0 | 11.0 | 14.0 | 10.4 | 12.0 | 14.0 | – | – | 686 | 14.8 | 43.8 | 0.0 | 0.0 | – |
| tree_lexical | 50 | 2.0 | 2.0 | 2.0 | 4.0 | 2.2 | 2.0 | 4.0 | – | – | 923 | 72.1 | 137 | 0.0 | 0.0 | – |
| tree_lexical+fts | 50 | 10.0 | 17.0 | 20.0 | 30.0 | 19.1 | 20.0 | 30.0 | – | – | 955 | 114 | 149 | 0.0 | 0.0 | – |
| managed | 50 | 11.0 | 18.0 | 23.0 | 34.0 | 22.3 | 24.0 | 36.0 | – | – | 945 | 44.4 | 68.0 | 0.0 | 0.0 | – |

#### novel-generated

| strategy | n | recall@1 | recall@3 | recall@5 | recall@10 | mrr | recall@1k_tok | recall@2k_tok | node_recall@5 | doc_recall@5 | ctx_tokens@5 | p50_ms | p95_ms | embed_calls/q | llm_tokens/q | tokens/success |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fts | 50 | 32.0 | 49.0 | 62.0 | 72.0 | 45.0 | 64.0 | 73.0 | – | – | 717 | 30.6 | 52.0 | 0.0 | 0.0 | – |
| rag_bm25 | 50 | 40.0 | 51.0 | 58.0 | 76.0 | 49.4 | 40.0 | 51.0 | – | – | 2,956 | 29.0 | 41.2 | 0.0 | 0.0 | – |
| tree_structure | 50 | 20.0 | 26.0 | 28.0 | 28.0 | 23.1 | 24.0 | 28.0 | – | – | 717 | 4.3 | 14.9 | 0.0 | 0.0 | – |
| tree_lexical | 50 | 30.0 | 44.0 | 46.0 | 52.0 | 37.4 | 44.0 | 54.0 | – | – | 820 | 36.8 | 61.2 | 0.0 | 0.0 | – |
| tree_lexical+fts | 50 | 32.0 | 52.0 | 62.0 | 66.0 | 43.7 | 62.0 | 66.0 | – | – | 748 | 60.5 | 94.0 | 0.0 | 0.0 | – |
| managed | 50 | 32.0 | 49.0 | 64.0 | 72.0 | 45.1 | 64.0 | 73.0 | – | – | 732 | 28.9 | 45.5 | 0.0 | 0.0 | – |
