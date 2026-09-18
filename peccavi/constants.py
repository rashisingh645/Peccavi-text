"""
peccavi/constants.py
Shared constants for PECCAVI watermarking system.
"""

# Secret key for watermark seed derivation
# Change in production - this is the default used across all agents
SECRET_KEY = "AIISC-SECRET"

# Watermarking parameters
DEFAULT_THETA = 2.0
THETA_MIN = 0.5   # matches the bounds actually used by Magister/run_peccavi
# Must match Auctor._tournament_sample()'s generation-time cap (peccavi/auctor.py) —
# REINFORCE previously could push theta_base up to 8.0 while Auctor silently capped
# actual generation at 5.0, so a reported theta_final above 5.0 never corresponded to
# any real change in the watermarked text. Both now share this one constant.
THETA_MAX = 5.0
TOURNAMENT_K = 16
Z_DETECTION_THRESHOLD = 4.0   # z-score cutoff used by every detector's detect()/z_threshold default
# Policy learning parameters
ALPHA = 0.05  # REINFORCE learning rate
GAMMA = 0.99  # Discount factor
LAMBDA_WM = 0.6  # Weight for watermark score in reward
NU_QUALITY = 0.4  # Weight for text quality in reward

# Evaluation parameters
DEFAULT_GENERATIONS = 10
DEFAULT_N_PARAPHRASES = 5
SUCCESS_WATERMARK_RETENTION = 0.85
SUCCESS_AUC_ROC = 0.90
SUCCESS_READABILITY = 4.5
