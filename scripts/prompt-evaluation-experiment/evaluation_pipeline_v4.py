#!/usr/bin/env python3
# ==============================================================================
# Prompt Evaluation Pipeline
# Compares Prompt V1 (original) vs Prompt V2 (improved)
#
# Compatible with: Google Colab · Jupyter Notebook · VS Code (# %% cells)
# Requirements:    litellm  pandas  tabulate
# ==============================================================================
#
# HOW TO RUN IN COLAB:
#   1. !pip install -q litellm pandas tabulate
#   2. Set your API key in Colab Secrets (🔑 icon in left sidebar):
#
#      Provider       | Secret name          | Model string example
#      ───────────────┼──────────────────────┼──────────────────────────────────
#      OpenAI         | OPENAI_API_KEY        | "gpt-4o-mini" / "gpt-4o"
#      Mistral        | MISTRAL_API_KEY       | "mistral/mistral-large-latest"
#      Anthropic      | ANTHROPIC_API_KEY     | "claude-3-5-haiku-20241022"
#      Google Gemini  | GEMINI_API_KEY        | "gemini/gemini-1.5-flash"
#
#   3. Upload evaluation_pipeline.py via Files panel (📁) → Upload
#   4. In a new cell run:  %run evaluation_pipeline.py
#
# ==============================================================================

# %% [markdown]
# ## Cell 1 — Installation (run once)

# %% ── install ────────────────────────────────────────────────────────────────
# !pip install -q litellm pandas tabulate

# %% [markdown]
# ## Cell 2 — Imports & Configuration

# %% ── imports ────────────────────────────────────────────────────────────────
import os
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import litellm
from litellm import completion as litellm_completion
from dotenv import load_dotenv

load_dotenv()

litellm.set_verbose = False   # silence litellm debug output

try:
    from IPython.display import display, HTML
    IN_NOTEBOOK = True
except ImportError:
    IN_NOTEBOOK = False



# ══════════════════════════════════════════════════════════════════════════════
# ▶  CONFIGURATION — edit these two lines only
# ══════════════════════════════════════════════════════════════════════════════
#
# MODEL string examples (litellm format):
#   OpenAI   →  "gpt-4o-mini"                       key: OPENAI_API_KEY
#   OpenAI   →  "gpt-4o"                             key: OPENAI_API_KEY
#   Mistral  →  "mistral/mistral-large-latest"       key: MISTRAL_API_KEY
#   Mistral  →  "mistral/mistral-small-latest"       key: MISTRAL_API_KEY
#   Anthropic→  "claude-3-5-haiku-20241022"          key: ANTHROPIC_API_KEY
#   Gemini   →  "gemini/gemini-1.5-flash"            key: GEMINI_API_KEY

MODEL: str = "mistral/mistral-large-latest"          # ← change to your model
API_KEY_ENV: str = "MISTRAL_API_KEY" # ← change to matching secret name
# ══════════════════════════════════════════════════════════════════════════════

SLEEP_BETWEEN_CALLS: float = 15.0    # seconds between API calls; raise if rate-limited

# Pass the key to litellm via environment (works with any provider)
_api_key = os.environ.get(API_KEY_ENV, "")
if _api_key:
    os.environ[API_KEY_ENV] = _api_key  # already set; litellm reads it automatically
else:
    print(f"⚠  {API_KEY_ENV} not found in environment. "
          "Set it in Colab Secrets (🔑 icon) and restart.")

# %% [markdown]

# %% [markdown]
# ## External modules

# %% ── imports from companion files ──────────────────────────────────────────────
from dataset   import DATASET
from prompt_v1 import PROMPT_V1
from prompt_v2 import PROMPT_V2

# ── Prompt registry ──────────────────────────────────────────────────────────────
# Contract: every prompt string must contain JOB_REQUEST_MARKER exactly once.
# To add a new version: create prompt_vN.py, import PROMPT_VN, add 'vN': PROMPT_VN.
JOB_REQUEST_MARKER: str = '<<<JOB_REQUEST>>>'

PROMPT_REGISTRY: Dict[str, str] = {
    "v1": PROMPT_V1,
    "v2": PROMPT_V2,
}


def build_prompt(version: str, job_request: str) -> str:
    """
    Return the complete prompt for *version* with the candidate's input text
    injected in place of JOB_REQUEST_MARKER.
    The pipeline is fully agnostic to prompt structure.
    """
    if version not in PROMPT_REGISTRY:
        raise ValueError(
            f"Unknown version {version!r}. "
            f"Available: {list(PROMPT_REGISTRY)}"
        )
    return PROMPT_REGISTRY[version].replace(JOB_REQUEST_MARKER, job_request)


# ## Cell 6 — LLM interface

# %% ── llm call ────────────────────────────────────────────────────────────────
def call_llm(system_prompt: str, model: str = MODEL) -> Tuple[Optional[dict], str]:
    """
    Call any LLM via litellm. The full prompt (including job_request) is passed
    as the user message, matching the original prompt structure.

    Returns
    -------
    (parsed_json, raw_text)
        parsed_json is None when the response cannot be parsed as valid JSON.
    """
    raw = ""
    try:
        response = litellm_completion(
            model=model,
            messages=[{"role": "user", "content": system_prompt}],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if the model wrapped output in them
        raw_clean = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw_clean = re.sub(r"\s*```$",           "", raw_clean, flags=re.MULTILINE).strip()

        return json.loads(raw_clean), raw

    except json.JSONDecodeError as e:
        return None, f"JSON_PARSE_ERROR: {e}\nRaw output was:\n{raw}"
    except Exception as e:
        return None, f"API_ERROR: {type(e).__name__}: {e}"


# %% [markdown]
# ## Cell 7 — Evaluation functions & checks

# %% ── schema defaults ────────────────────────────────────────────────────────

def make_empty_output() -> dict:
    """
    Canonical default output: every Optional field is null, every List is [].
    Used by run_field_checks to verify non-asserted fields.
    When the schema changes, update this function — it is the single source of
    truth for what an 'empty' extraction looks like.
    """
    return {
        "candidate_description": {
            "years_experience": None,
            "languages":        [],
            "education_level":  None,
            "skills":           [],
        },
        "job_description": {
            "desired_positions":             [],
            "desired_tech_stack":            [],
            "preferred_domains":             [],
            "preferred_activities":          [],
            "preferred_companies":           [],
            "desired_compensation_monthly":  None,
            "preferred_work_mode":           None,
        },
    }


# %% ── field checks registry ─────────────────────────────────────────────────
#
# FIELD_CHECKS is the single source of truth for:
#   • which schema fields exist
#   • how each field is compared (exact / set_f1 / set_f1_soft / set_exact / companies_f1)
#
# When the schema changes:
#   1. Update make_empty_output() above.
#   2. Add / remove / modify entries here.
#   → No other changes needed in the evaluation code.
#
# Check types:
#   exact        — exact_match(expected, actual)             passes if equal (case-insensitive for str)
#   set_f1       — set_f1(expected, actual) == 1.0           passes on perfect set overlap
#   set_f1_soft  — set_f1(expected, actual) >= 0.5           soft pass (wording variance tolerance)
#   set_exact    — set(expected) == set(actual)              order-independent exact set match
#   companies_f1 — companies_f1(expected, actual) == 1.0    (industry, location) pair F1

FIELD_CHECKS: List[Dict] = [
    # ── CandidateDescription ──────────────────────────────────────────────────
    {
        "id":          "CHKF-01",
        "path":        "candidate_description.years_experience",
        "description": "years_experience — exact match",
        "type":        "exact",
    },
    {
        "id":          "CHKF-02",
        "path":        "candidate_description.languages",
        "description": "languages — set-F1 = 1.0  (case-insensitive)",
        "type":        "set_f1",
    },
    {
        "id":          "CHKF-03",
        "path":        "candidate_description.education_level",
        "description": "education_level — exact match  (case-insensitive)",
        "type":        "exact",
    },
    {
        "id":          "CHKF-04",
        "path":        "candidate_description.skills",
        "description": "skills — set-F1 = 1.0  (case-insensitive)",
        "type":        "set_f1",
    },
    # ── JobDescription — simple list / scalar fields ──────────────────────────
    {
        "id":          "CHKF-05",
        "path":        "job_description.desired_positions",
        "description": "desired_positions — set-F1 = 1.0",
        "type":        "set_f1",
    },
    {
        "id":          "CHKF-06",
        "path":        "job_description.desired_tech_stack",
        "description": "desired_tech_stack — set-F1 = 1.0",
        "type":        "set_f1",
    },
    {
        "id":          "CHKF-07",
        "path":        "job_description.preferred_domains",
        "description": "preferred_domains — set-F1 = 1.0",
        "type":        "set_f1",
    },
    {
        "id":          "CHKF-08",
        "path":        "job_description.preferred_activities",
        "description": "preferred_activities — set-F1 ≥ 0.5  (gerund variance)",
        "type":        "set_f1_soft",
    },
    {
        "id":          "CHKF-09",
        "path":        "job_description.preferred_companies",
        "description": "preferred_companies — (industry, location) pair F1 = 1.0",
        "type":        "companies_f1",
    },
    # ── desired_compensation_monthly sub-fields ───────────────────────────────
    {
        "id":          "CHKF-10",
        "path":        "job_description.desired_compensation_monthly.salary_min",
        "description": "salary_min — exact match",
        "type":        "exact",
    },
    {
        "id":          "CHKF-11",
        "path":        "job_description.desired_compensation_monthly.salary_max",
        "description": "salary_max — exact match",
        "type":        "exact",
    },
    {
        "id":          "CHKF-12",
        "path":        "job_description.desired_compensation_monthly.currency",
        "description": "currency — exact match  (3-letter ISO or null)",
        "type":        "exact",
    },
    {
        "id":          "CHKF-13",
        "path":        "job_description.desired_compensation_monthly.is_gross",
        "description": "is_gross — exact bool match",
        "type":        "exact",
    },
    {
        "id":          "CHKF-14",
        "path":        "job_description.desired_compensation_monthly.benefits",
        "description": "benefits — set-F1 = 1.0",
        "type":        "set_f1",
    },
    # ── preferred_work_mode sub-fields ────────────────────────────────────────
    {
        "id":          "CHKF-15",
        "path":        "job_description.preferred_work_mode.employment_type",
        "description": "employment_type — set exact match",
        "type":        "set_exact",
    },
    {
        "id":          "CHKF-16",
        "path":        "job_description.preferred_work_mode.preferred_remote_policy",
        "description": "preferred_remote_policy — set exact match",
        "type":        "set_exact",
    },
    {
        "id":          "CHKF-17",
        "path":        "job_description.preferred_work_mode.acceptable_remote_policy",
        "description": "acceptable_remote_policy — set exact match",
        "type":        "set_exact",
    },
]


# %% ── evaluation helpers ─────────────────────────────────────────────────────

def get_nested(d: dict, path: str) -> Any:
    """
    Retrieve a nested value using dot-notation path.
    E.g. get_nested(obj, 'job_description.desired_compensation_monthly.salary_min')
    Returns None if any key is missing or intermediate value is None.
    """
    parts = path.split(".")
    current = d
    for p in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(p)
    return current


def _normalise(v: Any) -> Any:
    """Lowercase strings for case-insensitive comparison."""
    if isinstance(v, str):
        return v.lower().strip()
    return v


def exact_match(expected: Any, actual: Any) -> bool:
    """True if expected == actual (case-insensitive for strings)."""
    return _normalise(expected) == _normalise(actual)


def set_f1(expected: List, actual: List) -> float:
    """
    Compute token-level set F1 between two lists (case-insensitive).
    Returns a float in [0.0, 1.0].
    Perfect match (including both empty) → 1.0.
    """
    e_set = {_normalise(x) for x in expected}
    a_set = {_normalise(x) for x in actual}

    if not e_set and not a_set:
        return 1.0
    if not e_set or not a_set:
        return 0.0

    intersection = e_set & a_set
    precision = len(intersection) / len(a_set)
    recall    = len(intersection) / len(e_set)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def companies_f1(expected: List[dict], actual: List[dict]) -> float:
    """
    Set F1 on (industry, location) tuples (case-insensitive).
    """
    def normalise_pair(c):
        ind = _normalise(c.get("industry") or "")
        loc = _normalise(c.get("location") or "")
        return (ind, loc)

    e_set = {normalise_pair(c) for c in expected}
    a_set = {normalise_pair(c) for c in (actual or [])}

    if not e_set and not a_set:
        return 1.0
    if not e_set or not a_set:
        return 0.0

    intersection = e_set & a_set
    precision = len(intersection) / len(a_set)
    recall    = len(intersection) / len(e_set)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


VALID_EMPLOYMENT_TYPES = {
    "full_time", "part_time", "contract", "fixed_term",
    "freelance", "internship", "temporary", "self_employed",
}

VALID_REMOTE_POLICIES = {"remote", "hybrid", "onsite"}


def _dispatch_check(
    check_type: str, expected_val: Any, actual_val: Any
) -> Tuple[bool, str]:
    """
    Apply the comparison function for *check_type* and return (passed, detail).
    None-safety: list functions use `or []`; exact handles None == None naturally.
    """
    if check_type == "exact":
        passed = exact_match(expected_val, actual_val)
        detail = f"expected={expected_val!r}  actual={actual_val!r}"

    elif check_type in ("set_f1", "set_f1_soft"):
        threshold = 0.5 if check_type == "set_f1_soft" else 1.0
        score  = set_f1(expected_val or [], actual_val or [])
        passed = score >= threshold
        detail = f"F1={score:.2f}  expected={expected_val}  actual={actual_val}"

    elif check_type == "set_exact":
        exp_s  = set(expected_val or [])
        act_s  = set(actual_val   or [])
        passed = exp_s == act_s
        detail = f"expected={sorted(exp_s)}  actual={sorted(act_s)}"

    elif check_type == "companies_f1":
        score  = companies_f1(expected_val or [], actual_val or [])
        passed = score == 1.0
        detail = f"F1={score:.2f}  expected={expected_val}  actual={actual_val}"

    else:
        raise ValueError(f"Unknown check type: {check_type!r}")

    return passed, detail


def run_field_checks(assertions: Dict[str, Any], actual: dict) -> List[dict]:
    """
    Run all field-level checks defined in FIELD_CHECKS.

    For each field:
      • If the path is in *assertions* → compare actual against the asserted value.
      • Otherwise              → compare actual against the schema default
                                 (from make_empty_output).

    This means adding a new field to the schema requires only updating
    FIELD_CHECKS and make_empty_output — no changes to individual test cases.
    """
    defaults = make_empty_output()
    results  = []

    for check in FIELD_CHECKS:
        path        = check["path"]
        is_asserted = path in assertions
        expected_val = assertions[path] if is_asserted else get_nested(defaults, path)
        actual_val   = get_nested(actual, path)

        passed, detail = _dispatch_check(check["type"], expected_val, actual_val)

        results.append({
            "check_id":    check["id"],
            "description": check["description"],
            "passed":      passed,
            "detail":      detail,
            "is_asserted": is_asserted,
        })

    return results


def run_invariant_checks(actual: dict) -> List[dict]:
    """
    Run all CHKI (invariant) checks on the model output.
    These are structural correctness checks independent of expected values.
    """
    results = []

    def _chk(check_id: str, description: str, passed: bool, detail: str = "") -> None:
        results.append({
            "check_id":    check_id,
            "description": description,
            "passed":      passed,
            "detail":      detail,
        })

    if actual is None:
        _chk("CHKI-ALL", "output is not None", False, "LLM returned None / parse error")
        return results

    # ── CHKI-01: top-level keys always present ─────────────────────────────────
    has_cd = isinstance(actual.get("candidate_description"), dict)
    has_jd = isinstance(actual.get("job_description"), dict)
    _chk("CHKI-01", "candidate_description is always present and is a dict", has_cd)
    _chk("CHKI-02", "job_description is always present and is a dict", has_jd)

    wm = get_nested(actual, "job_description.preferred_work_mode")
    comp = get_nested(actual, "job_description.desired_compensation_monthly")

    # ── CHKI-03: remote policy sets are disjoint ───────────────────────────────
    if isinstance(wm, dict):
        pref = set(wm.get("preferred_remote_policy") or [])
        acc  = set(wm.get("acceptable_remote_policy") or [])
        overlap = pref & acc
        _chk("CHKI-03", "preferred_remote_policy ∩ acceptable_remote_policy = ∅",
             len(overlap) == 0,
             f"overlap={overlap}")
    else:
        _chk("CHKI-03", "preferred_remote_policy ∩ acceptable_remote_policy = ∅",
             True, "preferred_work_mode is null → not applicable")

    # ── CHKI-04: employment_type values are valid enum ─────────────────────────
    if isinstance(wm, dict):
        et_vals = wm.get("employment_type") or []
        invalid = [v for v in et_vals if v not in VALID_EMPLOYMENT_TYPES]
        _chk("CHKI-04", "all employment_type values are valid enum members",
             len(invalid) == 0,
             f"invalid values: {invalid}" if invalid else "")
    else:
        _chk("CHKI-04", "all employment_type values are valid enum members",
             True, "preferred_work_mode is null → not applicable")

    # ── CHKI-05: remote policy values are valid ────────────────────────────────
    if isinstance(wm, dict):
        all_remote = (
            list(wm.get("preferred_remote_policy") or []) +
            list(wm.get("acceptable_remote_policy") or [])
        )
        invalid = [v for v in all_remote if v not in VALID_REMOTE_POLICIES]
        _chk("CHKI-05", "all remote policy values are in {remote, hybrid, onsite}",
             len(invalid) == 0,
             f"invalid values: {invalid}" if invalid else "")
    else:
        _chk("CHKI-05", "all remote policy values are in {remote, hybrid, onsite}",
             True, "preferred_work_mode is null → not applicable")

    # ── CHKI-06: currency is 3-letter ISO or null ──────────────────────────────
    if isinstance(comp, dict):
        currency = comp.get("currency")
        valid_cur = (currency is None) or (isinstance(currency, str) and len(currency) == 3 and currency.isalpha())
        _chk("CHKI-06", "currency is 3-letter ISO 4217 code or null",
             valid_cur,
             f"currency={currency!r}")
    else:
        _chk("CHKI-06", "currency is 3-letter ISO 4217 code or null",
             True, "desired_compensation_monthly is null → not applicable")

    # ── CHKI-07: preferred_work_mode not null iff employment info extracted ─────
    if isinstance(wm, dict):
        et_non_empty = bool(wm.get("employment_type"))
        rp_non_empty = bool(wm.get("preferred_remote_policy")) or bool(wm.get("acceptable_remote_policy"))
        _chk("CHKI-07", "if preferred_work_mode exists, it contains at least one work-mode signal",
             et_non_empty or rp_non_empty,
             f"employment_type={wm.get('employment_type')}  remote={wm.get('preferred_remote_policy')}")
    else:
        _chk("CHKI-07", "if preferred_work_mode exists, it contains at least one work-mode signal",
             True, "preferred_work_mode is null → not applicable")

    return results


# %% [markdown]
# ## Cell 8 — Pipeline runner

# %% ── pipeline ────────────────────────────────────────────────────────────────

def run_pipeline(
    dataset: List[dict],
    versions: List[str] = ("v1", "v2"),
    model: str = MODEL,
    sleep: float = SLEEP_BETWEEN_CALLS,
    verbose: bool = True,
) -> List[dict]:
    """
    For each test case in *dataset*, call the LLM with each prompt version,
    then run all checks.

    Returns
    -------
    List of result records — one per (test_case, version) combination.
    """
    records = []

    active = [tc for tc in dataset if tc.get("active", True)]
    total  = len(active) * len(versions)
    done   = 0

    for tc in active:
        for ver in versions:
            done += 1
            if verbose:
                print(f"[{done}/{total}] {tc['id']} · prompt {ver} … ", end="", flush=True)

            prompt = build_prompt(ver, tc["input"])
            parsed, raw = call_llm(prompt, model=model)

            field_checks     = run_field_checks(tc["assertions"], parsed or {})
            invariant_checks = run_invariant_checks(parsed)

            all_checks = field_checks + invariant_checks
            passed     = sum(1 for c in all_checks if c["passed"])
            total_c    = len(all_checks)

            if verbose:
                status = "✓" if passed == total_c else f"✗ ({total_c - passed} fail)"
                print(status)

            records.append({
                "test_id":        tc["id"],
                "target_field":   tc["target_field"],
                "rule_type":      tc["rule_type"],
                "prompt_version": ver,
                "input":          tc["input"],
                "notes":          tc["notes"],
                "parsed_output":  parsed,
                "raw_output":     raw,
                "checks":         all_checks,
                "passed":         passed,
                "total_checks":   total_c,
                "score":          round(passed / total_c, 3) if total_c else 0.0,
            })

            time.sleep(sleep)

    return records


# %% [markdown]
# ## Cell 9 — Results reporter

# %% ── reporter ────────────────────────────────────────────────────────────────

# ── ANSI colour helpers (work in Colab output & most terminals) ───────────────
class C:
    """Simple ANSI colour codes. Disabled automatically when not in a TTY."""
    _on = True  # set to False to disable all colour

    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RESET  = "\033[0m"

    @classmethod
    def g(cls, s):   return f"{cls.GREEN}{s}{cls.RESET}"   if cls._on else str(s)
    @classmethod
    def r(cls, s):   return f"{cls.RED}{s}{cls.RESET}"     if cls._on else str(s)
    @classmethod
    def y(cls, s):   return f"{cls.YELLOW}{s}{cls.RESET}"  if cls._on else str(s)
    @classmethod
    def b(cls, s):   return f"{cls.BOLD}{s}{cls.RESET}"    if cls._on else str(s)
    @classmethod
    def dim(cls, s): return f"{cls.DIM}{s}{cls.RESET}"     if cls._on else str(s)
    @classmethod
    def cyan(cls, s):return f"{cls.CYAN}{s}{cls.RESET}"    if cls._on else str(s)


def _tick(passed: bool) -> str:
    return C.g("✓ PASS") if passed else C.r("✗ FAIL")

def _emoji(passed: bool) -> str:
    return "✅" if passed else "❌"

def _score_bar(score: float, width: int = 10) -> str:
    """Visual progress bar, e.g. [████████░░] 0.80"""
    filled = round(score * width)
    bar = "█" * filled + "░" * (width - filled)
    colour = C.g if score == 1.0 else (C.y if score >= 0.7 else C.r)
    return colour(f"[{bar}] {score:.0%}")

def _delta_str(delta: float) -> str:
    if delta > 0:
        return C.g(f"+{delta:+.0%} ▲ v2")
    elif delta < 0:
        return C.r(f"{delta:+.0%} ▼ v1")
    return C.dim("  0%  tie")

def _sep(char: str = "─", width: int = 80) -> str:
    return char * width


# ── Public report functions ───────────────────────────────────────────────────

def report_summary(records: List[dict]) -> pd.DataFrame:
    """DataFrame: one row per (test_id, prompt_version)."""
    rows = []
    for r in records:
        rows.append({
            "test_id":      r["test_id"],
            "target_field": r["target_field"],
            "version":      r["prompt_version"],
            "score":        r["score"],
            "passed":       r["passed"],
            "total":        r["total_checks"],
        })
    return pd.DataFrame(rows)


def report_comparison(records: List[dict]) -> pd.DataFrame:
    """DataFrame: side-by-side v1 vs v2 score per test case."""
    by_id: Dict[str, dict] = {}
    for r in records:
        tid = r["test_id"]
        if tid not in by_id:
            by_id[tid] = {"target_field": r["target_field"]}
        by_id[tid][f"score_{r['prompt_version']}"] = r["score"]
        by_id[tid]["total"] = r["total_checks"]

    rows = []
    for tid, d in by_id.items():
        s1 = d.get("score_v1", 0.0)
        s2 = d.get("score_v2", 0.0)
        rows.append({
            "test_id":      tid,
            "target_field": d["target_field"],
            "score_v1":     s1,
            "score_v2":     s2,
            "delta":        round(s2 - s1, 3),
            "winner":       "v2 ↑" if s2 > s1 else ("v1 ↑" if s1 > s2 else "tie"),
        })
    return pd.DataFrame(rows)


def report_check_detail(records: List[dict], test_id: str) -> pd.DataFrame:
    """DataFrame: per-check results for one test case, v1 and v2 side by side."""
    filtered = [r for r in records if r["test_id"] == test_id]
    check_map: Dict[str, dict] = {}
    for r in filtered:
        ver = r["prompt_version"]
        for c in r["checks"]:
            cid = c["check_id"]
            if cid not in check_map:
                check_map[cid] = {"check_id": cid, "description": c["description"]}
            check_map[cid][f"{ver}_passed"] = c["passed"]
            check_map[cid][f"{ver}_detail"] = c["detail"]

    rows = []
    for cid, d in check_map.items():
        rows.append({
            "check_id":    cid,
            "description": d["description"],
            "v1":          _emoji(d.get("v1_passed", False)),
            "v2":          _emoji(d.get("v2_passed", False)),
            "v1_detail":   d.get("v1_detail", ""),
            "v2_detail":   d.get("v2_detail", ""),
        })
    return pd.DataFrame(rows)


def report_aggregate(records: List[dict]) -> pd.DataFrame:
    """DataFrame: aggregate score per prompt version across the full dataset."""
    from collections import defaultdict
    totals: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"passed": 0, "total": 0, "cases": 0}
    )
    for r in records:
        ver = r["prompt_version"]
        totals[ver]["passed"] += r["passed"]
        totals[ver]["total"]  += r["total_checks"]
        totals[ver]["cases"]  += 1

    rows = []
    for ver, d in totals.items():
        rows.append({
            "version":       ver,
            "test_cases":    d["cases"],
            "checks_passed": d["passed"],
            "checks_total":  d["total"],
            "score":         round(d["passed"] / d["total"], 3) if d["total"] else 0.0,
        })
    return pd.DataFrame(rows).sort_values("version").reset_index(drop=True)


# ── Pretty-print functions ────────────────────────────────────────────────────

_INVARIANT_REGISTRY = [
    ("CHKI-01", "candidate_description always present and is a dict"),
    ("CHKI-02", "job_description always present and is a dict"),
    ("CHKI-03", "preferred_remote_policy ∩ acceptable_remote_policy = ∅"),
    ("CHKI-04", "all employment_type values are valid enum members"),
    ("CHKI-05", "all remote policy values ∈ {remote, hybrid, onsite}"),
    ("CHKI-06", "currency is 3-letter ISO 4217 or null"),
    ("CHKI-07", "preferred_work_mode, if present, has ≥1 work-mode signal"),
]


def print_checks_registry() -> None:
    """
    Print the full list of checks derived directly from FIELD_CHECKS.
    Field checks are never out of sync with the registry: both share one source.
    """
    n_field = sum(1 for c in FIELD_CHECKS if c["type"] != "set_f1_soft")
    n_soft  = sum(1 for c in FIELD_CHECKS if c["type"] == "set_f1_soft")
    n_inv   = len(_INVARIANT_REGISTRY)

    print(f"\n{C.b('CHECKS REGISTRY')}")
    print(_sep("═"))
    print(f"  {'ID':<10} {'TYPE':<12} DESCRIPTION")
    print(_sep())
    for check in FIELD_CHECKS:
        is_soft = check["type"] == "set_f1_soft"
        colour  = C.y if is_soft else str
        ctype   = "field/soft" if is_soft else "field"
        print(f"  {colour(check['id']):<21} {C.dim(ctype):<21} {check['description']}")
    for cid, desc in _INVARIANT_REGISTRY:
        print(f"  {C.cyan(cid):<21} {C.dim('invariant'):<21} {desc}")
    print(_sep())
    print(f"  Total {len(FIELD_CHECKS) + n_inv} checks:  "
          f"{C.g(str(n_field))} field  ·  "
          f"{C.y(str(n_soft))} soft  ·  "
          f"{C.cyan(str(n_inv))} invariant\n")


def print_aggregate(records: List[dict]) -> None:
    """Print a compact aggregate score table with visual bars."""
    agg = report_aggregate(records)
    print(f"\n{C.b('AGGREGATE SCORES')}")
    print(_sep("═"))
    print(f"  {'Version':<10} {'Cases':>6}  {'Passed':>7}  {'Total':>6}  Score")
    print(_sep())
    for _, row in agg.iterrows():
        print(f"  {C.b(row['version']):<19} {row['test_cases']:>6}  "
              f"{row['checks_passed']:>7}  {row['checks_total']:>6}  "
              f"{_score_bar(row['score'])}")
    print()


def print_comparison(records: List[dict]) -> None:
    """Print side-by-side v1 vs v2 comparison with delta bars."""
    comp = report_comparison(records)
    print(f"\n{C.b('V1 vs V2 — PER TEST CASE')}")
    print(_sep("═"))
    print(f"  {'ID':<7} {'Target field':<45} {'v1':^12} {'v2':^12} Delta")
    print(_sep())
    for _, row in comp.iterrows():
        print(f"  {row['test_id']:<7} {C.dim(row['target_field']):<54} "
              f"{_score_bar(row['score_v1'], 6)}  "
              f"{_score_bar(row['score_v2'], 6)}  "
              f"{_delta_str(row['delta'])}")
    print()


def print_detail(records: List[dict], test_id: str) -> None:
    """Print per-check drill-down for one test case."""
    tc = next((t for t in DATASET if t["id"] == test_id), None)
    detail = report_check_detail(records, test_id)

    n_v1 = (detail["v1"] == "✅").sum()
    n_v2 = (detail["v2"] == "✅").sum()

    print(f"\n{C.b(f'DETAIL: {test_id}')}")
    print(_sep("═"))
    if tc:
        print(f"  {C.dim('Field:')}   {tc['target_field']}")
        print(f"  {C.dim('Rule:')}    {tc['rule_type']}")
        input_preview = tc["input"][:100] + ("…" if len(tc["input"]) > 100 else "")
        print(f"  {C.dim('Input:')}   {input_preview}")
        print(f"  {C.dim('Notes:')}   {tc['notes']}")
    print(_sep())
    print(f"  {'ID':<10} {'v1':^8} {'v2':^8}  Description / Detail")
    print(_sep())

    for _, row in detail.iterrows():
        v1_ok = row["v1"] == "✅"
        v2_ok = row["v2"] == "✅"
        status_v1 = C.g("  ✓   ") if v1_ok else C.r("  ✗   ")
        status_v2 = C.g("  ✓   ") if v2_ok else C.r("  ✗   ")
        print(f"  {row['check_id']:<10} {status_v1} {status_v2}  {row['description']}")
        # Show detail line only when at least one version failed
        if not v1_ok or not v2_ok:
            det_v1 = row.get("v1_detail", "")
            det_v2 = row.get("v2_detail", "")
            if det_v1 and not v1_ok:
                print(f"  {'':<10}   {C.dim('v1: ' + det_v1[:72])}")
            if det_v2 and not v2_ok:
                print(f"  {'':<10}   {C.dim('v2: ' + det_v2[:72])}")

    print(_sep())
    print(f"  Result:  v1 {n_v1}/{len(detail)}  ·  v2 {n_v2}/{len(detail)}\n")


def print_full_report(records: List[dict]) -> None:
    """
    One-stop function: prints checks registry, aggregate, comparison,
    and detail for every test case.
    """
    print_checks_registry()
    print_aggregate(records)
    print_comparison(records)
    for tc in DATASET:
        print_detail(records, tc["id"])


# %% [markdown]
# ## Cell 10 — Run everything

# %% ── run ────────────────────────────────────────────────────────────────────

if __name__ == "__main__" or IN_NOTEBOOK:

    # ── 1. Run the pipeline ───────────────────────────────────────────────────
    print(C.b(f"\nRunning evaluation pipeline  (model: {MODEL})\n"))
    results = run_pipeline(DATASET, versions=["v1", "v2"], verbose=True)

    # ── 2. Full human-readable report ─────────────────────────────────────────
    print_full_report(results)

    # ── 3. DataFrames for further analysis in the notebook ────────────────────
    summary_df    = report_summary(results)
    comparison_df = report_comparison(results)
    aggregate_df  = report_aggregate(results)

    if IN_NOTEBOOK:
        # Styled table works best in Colab / Jupyter
        def _style_df(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
            def colour_cell(val):
                if val == "✅":  return "color: green; font-weight: bold"
                if val == "❌":  return "color: red"
                if isinstance(val, float):
                    if val == 1.0:   return "background-color: #d4edda"
                    if val >= 0.7:   return "background-color: #fff3cd"
                    if val < 0.7:    return "background-color: #f8d7da"
                return ""
            return df.style.applymap(colour_cell)

        print("\n── Summary (all checks per case per version) ──")
        display(_style_df(summary_df))
        print("\n── Comparison (v1 vs v2 delta) ──")
        display(_style_df(comparison_df))
    else:
        print("\nDataFrames available: summary_df · comparison_df · aggregate_df")

    print(f"\n{C.b('API reference:')}")
    print("  results         — raw list of all run records")
    print("  print_detail(results, 'TC-01')       — drill-down for one test case")
    print("  report_check_detail(results, 'TC-01')— same as DataFrame")
    print("  print_full_report(results)            — reprint the full report")
