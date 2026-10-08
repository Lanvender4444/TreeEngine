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

PLANNER_SYSTEM = "You classify retrieval queries. Reply with one word."
PLANNER = """Classify the query into one of:
LOOKUP - a short fact, name, number, code identifier lookup
DOCUMENT_REASONING - needs understanding/explanation across a section or document
HYBRID - asks to find mentions of something inside a specific part/section

Query: {question}
Answer with LOOKUP, DOCUMENT_REASONING or HYBRID.
"""
