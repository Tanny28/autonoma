"""Serving layer — FastAPI model serving. Owner: Dushyant Dhote.

Serves the monitored baseline classifier, logs every request (features,
prediction, confidence), and ingests true labels on a configurable delay so the
performance signal family behaves the way it would in production.
"""
