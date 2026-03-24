# Re-export the mathematical constants and configuration from the parent module's constants.py
# This preserves backward compatibility: `from modules.continued_fractions.targets import g_const_dict`
from modules.continued_fractions.constants import (
    g_const_dict,
    g_N_verify_terms,
    g_N_verify_compare_length,
    g_N_verify_dps,
    g_N_initial_search_terms,
    g_N_initial_key_length,
    g_N_initial_search_dps,
    Khinchin,
)
