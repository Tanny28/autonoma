"""O2 — Diagnostic agent layer. Owner: Tanmay Shinde.

LangGraph orchestration over a Groq-hosted Llama model. Consumes a signal
bundle plus relevant incident history and emits a cause classification, a
confidence score, a selected action, and a natural-language justification.

Evaluation arms 3 and 4 (threshold rules, trained classifier) also live here:
they consume the identical signal bundle, which is the whole point of the
ablation.
"""
