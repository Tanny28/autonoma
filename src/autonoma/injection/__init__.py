"""O1 — Replay & injection layer. Owner: Dushyant Dhote / Pranav Shimpi.

Replays a public dataset through the serving API one record at a time, injects
one of the five degradation classes at a known index, and records the
ground-truth (cause, onset_index) pair that makes evaluation possible.
"""
