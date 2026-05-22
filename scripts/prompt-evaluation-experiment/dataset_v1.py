# dataset.py
# Test cases for the evaluation pipeline.
# Each entry: input text + full expected JobAndCandidateDescription output.

from typing import Any, Dict, List, Optional

# ## Cell 3 — Dataset helpers

# %% ── helpers ────────────────────────────────────────────────────────────────
def make_empty_output() -> dict:
    """Fully-defaulted JobAndCandidateDescription skeleton."""
    return {
        "candidate_description": {
            "years_experience": None,
            "languages": [],
            "education_level": None,
            "skills": [],
        },
        "job_description": {
            "desired_positions": [],
            "desired_tech_stack": [],
            "preferred_domains": [],
            "preferred_activities": [],
            "preferred_companies": [],
            "desired_compensation_monthly": None,
            "preferred_work_mode": None,
        },
    }


def _deep_update(base: dict, overrides: dict) -> None:
    """Recursively merge *overrides* into *base* in-place."""
    for k, v in overrides.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def build_expected(overrides: dict) -> dict:
    """
    Build a complete expected-output dict starting from defaults,
    then applying *overrides* with deep merge.

    Example
    -------
    build_expected({
        "candidate_description": {"years_experience": 7}
    })
    """
    base = make_empty_output()
    _deep_update(base, overrides)
    return base


# %% [markdown]
# ## Cell 4 — Dataset (11 test cases, one per schema field)

# %% ── dataset ────────────────────────────────────────────────────────────────
DATASET: List[dict] = [

    # ── TC-01: years_experience ───────────────────────────────────────────────
    {
        "id": "TC-01",
        "target_field": "candidate_description.years_experience",
        "rule_type": "positive_extraction",
        "input": (
            "I have 7 years of experience in backend development."
        ),
        "expected": build_expected({
            "candidate_description": {
                "years_experience": 7,
            }
        }),
        "notes": (
            "Explicit year count → 7. No other extractable signals present. "
            "Tests basic integer extraction and all-other-fields-empty constraint."
        ),
    },

    # ── TC-02: languages ──────────────────────────────────────────────────────
    {
        "id": "TC-02",
        "target_field": "candidate_description.languages",
        "rule_type": "normalization + negative (programming languages excluded)",
        "input": (
            "Я свободно говорю на русском и английском языках. "
            "Также хорошо знаю Python и SQL."
        ),
        "expected": build_expected({
            "candidate_description": {
                "languages": ["Russian", "English"],
                "skills":    ["Python", "SQL"],
            }
        }),
        "notes": (
            "Russian input: natural languages normalised to English names. "
            "Python / SQL → skills, NOT languages. "
            "Tests multilingual normalisation (v2 addition)."
        ),
    },

    # ── TC-03: education_level ────────────────────────────────────────────────
    {
        "id": "TC-03",
        "target_field": "candidate_description.education_level",
        "rule_type": "positive_extraction + negative (in-progress excluded)",
        "input": (
            "I finished my bachelor's degree in Economics in 2018. "
            "I am currently enrolled in a master's program."
        ),
        "expected": build_expected({
            "candidate_description": {
                "education_level": "bachelor's",
            }
        }),
        "notes": (
            "bachelor's = completed → keep. "
            "master's = in progress → discard. "
            "Tests the 'only completed degrees' rule."
        ),
    },

    # ── TC-04: skills (casing normalisation) ─────────────────────────────────
    {
        "id": "TC-04",
        "target_field": "candidate_description.skills",
        "rule_type": "normalization (casing) + negative (desired skills excluded)",
        "input": (
            "I have experience with python, aws, and docker. "
            "I want to learn Rust."
        ),
        "expected": build_expected({
            "candidate_description": {
                "skills": ["Python", "AWS", "Docker"],
            }
        }),
        "notes": (
            "python→Python, aws→AWS, docker→Docker (conventional casing rule, v2). "
            "Rust excluded: 'want to learn' = future intent, not current skill."
        ),
    },

    # ── TC-05: desired_positions ──────────────────────────────────────────────
    {
        "id": "TC-05",
        "target_field": "job_description.desired_positions",
        "rule_type": "positive_extraction + negative (current title excluded)",
        "input": (
            "I am currently working as a QA engineer. "
            "I am looking for a DevOps or SRE role."
        ),
        "expected": build_expected({
            "job_description": {
                "desired_positions": ["DevOps", "SRE"],
            }
        }),
        "notes": (
            "'QA engineer' = current title → excluded. "
            "'DevOps', 'SRE' = forward-looking → included. "
            "Tests past-vs-future signal discrimination."
        ),
    },

    # ── TC-06: desired_tech_stack (cross-field: skills vs desired_tech_stack) ─
    {
        "id": "TC-06",
        "target_field": "job_description.desired_tech_stack",
        "rule_type": "cross_field_disambiguation (skills ↔ desired_tech_stack)",
        "input": (
            "I know Vue.js well and have used it for 3 years. "
            "For my next role, I'd love to work with React and TypeScript."
        ),
        "expected": build_expected({
            "candidate_description": {
                "skills": ["Vue.js"],
            },
            "job_description": {
                "desired_tech_stack": ["React", "TypeScript"],
            },
        }),
        "notes": (
            "Vue.js = current knowledge → skills only. "
            "React / TypeScript = explicit future preference → desired_tech_stack only. "
            "Key cross-field check: no overlap."
        ),
    },

    # ── TC-07: preferred_domains (cross-field: domains vs industry) ───────────
    {
        "id": "TC-07",
        "target_field": "job_description.preferred_domains",
        "rule_type": "cross_field_disambiguation (preferred_domains ↔ preferred_companies.industry)",
        "input": (
            "I want to work in the field of applied machine learning and computer vision. "
            "I'm particularly interested in the healthcare sector."
        ),
        "expected": build_expected({
            "job_description": {
                "preferred_domains": ["applied machine learning", "computer vision"],
                "preferred_companies": [
                    {"industry": "healthcare", "location": None}
                ],
            }
        }),
        "notes": (
            "ML/CV = subject domains → preferred_domains. "
            "Healthcare = sector/industry → preferred_companies[0].industry. "
            "Tests the domain-vs-industry disambiguation rule."
        ),
    },

    # ── TC-08: preferred_activities ───────────────────────────────────────────
    {
        "id": "TC-08",
        "target_field": "job_description.preferred_activities",
        "rule_type": "positive_extraction + gerund normalisation",
        "input": (
            "I'm looking for a role where I can mentor junior developers "
            "and lead technical architecture discussions."
        ),
        "expected": build_expected({
            "job_description": {
                "preferred_activities": [
                    "mentoring junior developers",
                    "leading technical architecture discussions",
                ],
            }
        }),
        "notes": (
            "Forward-looking activities: filler ('role where I can') stripped, "
            "verbs normalised to gerund form. "
            "Soft check: allow minor wording variation (see CHKF-08 tolerance note)."
        ),
    },

    # ── TC-09: preferred_companies ────────────────────────────────────────────
    {
        "id": "TC-09",
        "target_field": "job_description.preferred_companies",
        "rule_type": "positive_extraction (industry + location pairing)",
        "input": (
            "I'm looking for opportunities in fintech companies, "
            "ideally based in Berlin."
        ),
        "expected": build_expected({
            "job_description": {
                "preferred_companies": [
                    {"industry": "fintech", "location": "Berlin"}
                ],
            }
        }),
        "notes": (
            "Industry and location explicitly linked by user → one object. "
            "Tests correct (industry, location) pairing."
        ),
    },

    # ── TC-10: desired_compensation_monthly (all sub-fields) ──────────────────
    {
        "id": "TC-10",
        "target_field": "job_description.desired_compensation_monthly",
        "rule_type": "positive_extraction (salary + currency + is_gross + benefits)",
        "input": (
            "I'm expecting a salary of at least 6,000 EUR per month net. "
            "Health insurance and stock options would be a great bonus."
        ),
        "expected": build_expected({
            "job_description": {
                "desired_compensation_monthly": {
                    "salary_min": 6000,
                    "salary_max": None,
                    "currency":   "EUR",
                    "is_gross":   False,
                    "benefits":   ["health insurance", "stock options"],
                },
            }
        }),
        "notes": (
            "salary_min=6000, no max. 'net' → is_gross=False. "
            "Benefits extracted as a separate list, not as salary. "
            "Tests all sub-fields of CompensationPreference."
        ),
    },

    # ── TC-11: preferred_work_mode (employment_type + remote policy split) ────
    {
        "id": "TC-11",
        "target_field": "job_description.preferred_work_mode",
        "rule_type": "positive_extraction + preferred/acceptable split invariant",
        "input": (
            "I'm looking for a full-time position. "
            "I prefer to work fully remotely, but I'm open to hybrid if needed."
        ),
        "expected": build_expected({
            "job_description": {
                "preferred_work_mode": {
                    "employment_type":          ["full_time"],
                    "preferred_remote_policy":  ["remote"],
                    "acceptable_remote_policy": ["hybrid"],
                },
            }
        }),
        "notes": (
            "'full-time' → full_time. 'prefer … fully remotely' → preferred. "
            "'open to hybrid' → acceptable. "
            "Invariant CHKI-01: preferred ∩ acceptable = ∅."
        ),
    },
]


# %% [markdown]