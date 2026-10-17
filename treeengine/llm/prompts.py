"""Prompt templates. Kept small and in English; models answer in the question's language."""

SUMMARY_SYSTEM = "You write concise, factual section summaries for a document index."
SUMMARY = """Summarise the section below in at most {max_chars} characters, in the same language
as the section. Mention the key entities, numbers and topics so the summary helps someone decide
whether to open this section. Reply with the summary only.

Section title: {title}
Subsections: {children}
Section text:
{text}
"""

STRUCTURE_SYSTEM = "You recover the heading outline of a document. Reply with JSON only."
STRUCTURE = """Below are the numbered blocks of a document (first characters of each).
Identify which blocks are section headings and their level (1 = top level).
Reply with a JSON list such as [{{"block": 0, "level": 1, "title": "Introduction"}}].
Return [] if the document has no meaningful sections.

{blocks}
"""

TREE_SELECT_SYSTEM = (
    "You navigate a document's table of contents one level at a time to find where the answer "
    "to a question is. Reply with JSON only."
)
TREE_SELECT = """Question: {question}

Path so far: {path}

Candidate sections at this level:
{candidates}

Pick at most {k} candidate ids most likely to contain the answer (best first).
Set "stop" to true if the chosen sections are already specific enough to read directly.
Reply as JSON: {{"selected": ["id", ...], "stop": false, "reason": "short"}}
If none is relevant reply {{"selected": [], "stop": true}}.
"""

TREE_REASON_SYSTEM = (
    "You explore a document's section tree step by step to find evidence for a question. "
    "You see one level at a time. Reply with JSON only."
)
TREE_REASON = """Question: {question}

Document: {document}
Current position: {path}

Sections here:
{options}

Evidence found so far:
{evidence}

Budget left: {steps} steps.

Choose ONE action:
- "SELECT": open a section to see its subsections            (needs "node")
- "READ": read a section's text and keep it as evidence     (needs "node")
- "SEARCH": search for words inside a section                (needs "node" and "query")
- "PARENT": go back up one level
- "STOP": the evidence found so far is enough, or nothing here is relevant

Reply as JSON: {{"action": "SELECT", "node": "s2", "query": "", "reason": "short"}}
"""

ANSWER_SYSTEM = (
    "You answer questions strictly from the provided evidence. Cite evidence with [E<n>] markers "
    "after the sentences they support. If the evidence is insufficient, say so. Answer in the "
    "language of the question."
)
ANSWER = """Question: {question}

Evidence:
{evidence}

Write the answer with [E<n>] citations.
"""

REASON_SYSTEM = (
    "You decide which parts of a document must be read to answer a question. You see the "
    "regions a search found, like a table of contents with matching excerpts. Reply with JSON only."
)

PLANNER_SYSTEM = "You classify retrieval queries. Reply with one word."
PLANNER = """Classify the query into one of:
LOOKUP - a short fact, name, number, code identifier lookup
DOCUMENT_REASONING - needs understanding/explanation across a section or document
HYBRID - asks to find mentions of something inside a specific part/section

Query: {question}
Answer with LOOKUP, DOCUMENT_REASONING or HYBRID.
"""

EXPAND_SYSTEM = "You rewrite search queries for a document index. Reply with JSON only."
EXPAND = """Question: {question}
{document}
Write {n} short search queries that use the words the document itself would use in the passages
that answer this question: names of statements, tables, sections or line items, defined terms,
abbreviations spelled out. Do not answer the question.
Reply as JSON: {{"queries": ["...", "..."]}}
"""

TOC_ROUTE_SYSTEM = (
    "You find where in a document the answer to a question is, from its page-by-page outline. "
    "Reply with JSON only."
)
TOC_ROUTE = """Question: {question}
{document}
Page-by-page outline (page: section titles starting there, or the page's first line):
{outline}

Pick at most {k} pages most likely to hold the facts needed (for example the statement, table or
note that reports them), best first.
Reply as JSON: {{"pages": [57, 58], "reason": "short"}}
"""

PLAN_SYSTEM = (
    "You plan how to search one document for the answer to a question. Reply with JSON only."
)
PLAN = """Question: {question}
{document}
1. Write {n} short search queries that use the words the document itself would use in the
   passages that answer this question: names of statements, tables, sections or line items,
   defined terms, abbreviations spelled out. Do not answer the question.
2. Decide "outline": true if the answer most likely sits in one specific statement, table or note
   (for example financial-statement line items, or a ratio computed from statements) that is
   easier to find by its place in the document than by keywords; otherwise false.
Reply as JSON: {{"queries": ["...", "..."], "outline": false}}
"""
