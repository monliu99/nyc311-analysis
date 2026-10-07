"""Every cleaning decision in one place.

PROVISIONAL values come from the 20-file sample and are settled after discovery
runs on the full cluster data (see README "Cleaning decisions").
"""

# PROVISIONAL: text that means "missing" (compared lowercased and trimmed; "" is counted separately).
PLACEHOLDER_TOKENS = ["n/a", "na", "n/a n/a", "unspecified", "0 unspecified", "unknown",
                      "none", "null", "-", "0", "0000", "00000"]
# Dates starting with this are placeholders (closed_date 1900-01-01, due_date 1900-01-02).
PLACEHOLDER_DATE_PREFIX = "1900-"

# A row is "out of file order" if created more than this many days from its file's median. Descriptive only:
# on the full data a block of later batches holds backfilled older records, which are real requests.
OUT_OF_WINDOW_DAYS = 30
# A file counts as date-ordered if at least this share of its rows are within OUT_OF_WINDOW_DAYS of its median;
# only ordered files are used to infer which days are incomplete.
ORDERED_FILE_SHARE = 0.5

# A day is "low volume" if its count is below this share of the centered 29-day rolling median.
LOW_DAY_RATIO = 0.2
# Runs of at least this many zero-count days are reported as empty stretches, not day by day.
EMPTY_STRETCH_DAYS = 7

# ---- Cleaning decisions (settled 2026-10-06 from discovery on the 20 local files) ----
# Nothing is deleted: every rule sets a value to NULL or adds a flag column; each question decides what to exclude.

# Exact values (after trim) that mean "missing", by column. "Other" in location_type is a real category, so it stays.
NULL_IF = {
    "borough": ["Unspecified"],
    "park_borough": ["Unspecified"],
    "open_data_channel_type": ["UNKNOWN"],
    "facility_type": ["N/A"],
    "park_facility_name": ["Unspecified"],
    "location_type": ["N/A"],
    "descriptor": ["N/A"],
    "city": ["N/A", "NA"],
    "status": ["Unspecified"],
}
PLACEHOLDER_YEAR = 1900  # closed_date 1900-01-01 and due_date 1900-01-02 become NULL

# NYC ZIP codes are 100xx-104xx and 110xx-116xx; ZIP+4 keeps its first five digits; anything else becomes NULL.
NYC_ZIP_PATTERN = r"^1(0[0-4]|1[0-6])\d{2}$"

BORO_CODE = {"MANHATTAN": 1, "BRONX": 2, "BROOKLYN": 3, "QUEENS": 4, "STATEN ISLAND": 5}
# Standard community districts per borough; higher numbers in community_board are parks/airports ("special areas").
CD_COUNT = {"MANHATTAN": 12, "BRONX": 12, "BROOKLYN": 18, "QUEENS": 14, "STATEN ISLAND": 3}

TEST_STATUSES = ["Closed - Testing", "Draft"]
# Unresolved = any real status that is not closed, cancelled, or a test (so new statuses like "In Progress",
# "Email Sent", "Unassigned" count automatically). "Unspecified" is a placeholder (NULL_IF), not a status.
CLOSED_STATUSES = ["Closed"]
NOT_UNRESOLVED_STATUSES = ["Cancelled"]
LONG_DURATION_HOURS = 365 * 24  # closed more than a year after opening
# Unresolved also requires no close date: a request with an open status but a (valid) close date is a
# status_conflict (mostly DOT "Pending" rows closed before they were opened), counted neither open nor closed.

# A run of missing batch numbers at the end of the pull: from the earliest missing number g with at least
# SPARSE_TAIL_MIN_MISSING numbers missing in [g, last] and at least this share of [g, last] missing, every later
# day is possibly incomplete (neighboring files there overlap, so gaps between them can't be dated one by one).
SPARSE_TAIL_MISSING_SHARE = 0.1
SPARSE_TAIL_MIN_MISSING = 3

# ---- Analysis sets shared by every question, so totals agree across the report ----
BASE_SQL = "not is_test and not dup_key"
# Complaints: minus information requests and rows that are not service requests (taxonomy category
# "Invalid entries": test records and web-form probes; must equal taxonomy.INVALID).
INVALID_CATEGORY = "Invalid entries"
COMPLAINTS_SQL = (f"{BASE_SQL} and not coalesce(is_info_request, false) "
                  f"and coalesce(category, '') <> '{INVALID_CATEGORY}'")
# Instant closure: closed at the second of opening, with a time of day (a date-only row closed the same date at
# midnight may be a same-day closure, so it is kept).
INSTANT_SQL = "closed_at_creation and not date_only_created"
# Typical seasonal and weekly profiles use full years up to this one; the pandemic years are shown separately.
PROFILE_LAST_YEAR = 2019
