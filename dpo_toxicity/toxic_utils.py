"""
Utility functions for toxicity.
"""

PROFANITY = "profanity"
SEXUALLY_EXPLICIT = "sexually_explicit"
IDENTITY_ATTACK = "identity_attack"
FLIRTATION = "flirtation"
THREAT = "threat"
INSULT = "insult"
SEVERE_TOXICITY = "severe_toxicity"
TOXICITY = "toxicity"

TOXICITY_METRICS = [
    PROFANITY,
    SEXUALLY_EXPLICIT,
    IDENTITY_ATTACK,
    FLIRTATION,
    THREAT,
    INSULT,
    SEVERE_TOXICITY,
    TOXICITY,
]


PERSPECTIVE_API_ATTRIBUTES = TOXICITY_METRICS
#PERSPECTIVE_API_KEY = 
PERSPECTIVE_API_ATTRIBUTES_LOWER = tuple(
    x.lower() for x in PERSPECTIVE_API_ATTRIBUTES
)
