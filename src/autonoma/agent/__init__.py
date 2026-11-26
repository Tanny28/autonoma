"""O2 — Diagnostic agent layer. Owner: Tanmay Shinde.

Every evaluation arm lives here and has the same signature,
`(SignalBundle) -> Decision`:

- arms.py   arms 1-2, alert-only and always-retrain
- rules.py  arm 3, threshold rules
- llm.py    arms 5-6, LLM agent over Groq, with optional incident memory
- memory.py incident memory for arm 6

Arm 4 (trained classifier, #31) goes here too.
"""
