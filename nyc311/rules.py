"""Every cleaning decision in one place.

PROVISIONAL values come from the 20-file sample and are settled after discovery
runs on the full cluster data (see README "Cleaning decisions").
"""

# PROVISIONAL: text that means "missing" (compared lowercased and trimmed; "" is counted separately).
PLACEHOLDER_TOKENS = ["n/a", "na", "n/a n/a", "unspecified", "0 unspecified", "unknown",
                      "none", "null", "-", "0", "0000", "00000"]
# Dates starting with this are placeholders (closed_date 1900-01-01, due_date 1900-01-02).
PLACEHOLDER_DATE_PREFIX = "1900-"

# A row is "outside its file's window" if created more than this many days from the file's median.
OUT_OF_WINDOW_DAYS = 30

# A day is "low volume" if its count is below this share of the centered 29-day rolling median.
LOW_DAY_RATIO = 0.2
# Runs of at least this many zero-count days are reported as empty stretches, not day by day.
EMPTY_STRETCH_DAYS = 7
