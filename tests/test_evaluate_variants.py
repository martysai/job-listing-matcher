"""Tests for the multi-variant LambdaRank ablation glue in `scripts/evaluate.py`.

These cover the pure-Python helpers (variant construction, method ordering)
so the test suite stays fast and does not depend on LightGBM being installed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_evaluate_module():
    """Import `scripts/evaluate.py` as a module under a unique name."""
    if "evaluate_script" in sys.modules:
        return sys.modules["evaluate_script"]

    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_script"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def evaluate_module():
    return _load_evaluate_module()


def test_build_lambdarank_variants_full_set(evaluate_module):
    rows = [
        {"cosine_similarity": 0.5, "bm25_score": 1.0, "general_score": 0.7, "cand_skill_count": 3.0},
    ]
    variants = evaluate_module._build_lambdarank_variants(
        rows=rows,
        weighted_fields=["general_score"],
        has_schema_features=True,
        use_bm25_feature=True,
    )

    assert set(variants) == {
        "lambdarank_score",
        "lambdarank_no_llm_expert",
        "lambdarank_no_schema",
    }

    main_features = variants["lambdarank_score"]["features"]
    no_llm_features = variants["lambdarank_no_llm_expert"]["features"]
    no_schema_features = variants["lambdarank_no_schema"]["features"]

    # bm25_score appears in every variant when use_bm25_feature is True.
    assert "bm25_score" in main_features
    assert "bm25_score" in no_llm_features
    assert "bm25_score" in no_schema_features

    # General score is in the main + no_schema, absent in no_llm_expert.
    assert "general_score" in main_features
    assert "general_score" in no_schema_features
    assert "general_score" not in no_llm_features

    # Schema feature is in main + no_llm_expert, absent in no_schema.
    assert "cand_skill_count" in main_features
    assert "cand_skill_count" in no_llm_features
    assert "cand_skill_count" not in no_schema_features


def test_build_lambdarank_variants_without_schema(evaluate_module):
    rows = [
        {"cosine_similarity": 0.5, "bm25_score": 1.0, "general_score": 0.7},
    ]
    variants = evaluate_module._build_lambdarank_variants(
        rows=rows,
        weighted_fields=["general_score"],
        has_schema_features=False,
        use_bm25_feature=True,
    )

    # Without schema features the `no_schema` ablation would duplicate the
    # main variant; only `no_llm_expert` remains as a meaningful ablation.
    assert set(variants) == {"lambdarank_score", "lambdarank_no_llm_expert"}
    assert (
        variants["lambdarank_no_llm_expert"]["features"]
        == ("cosine_similarity", "bm25_score")
    )


def test_build_lambdarank_variants_without_llm_experts(evaluate_module):
    rows = [
        {"cosine_similarity": 0.5, "bm25_score": 1.0, "cand_skill_count": 3.0},
    ]
    variants = evaluate_module._build_lambdarank_variants(
        rows=rows,
        weighted_fields=[],
        has_schema_features=True,
        use_bm25_feature=True,
    )
    # Without LLM experts, `no_llm_expert` is a no-op (matches the main set),
    # but `no_schema` is still meaningful.
    assert set(variants) == {"lambdarank_score", "lambdarank_no_schema"}
    assert (
        variants["lambdarank_no_schema"]["features"]
        == ("cosine_similarity", "bm25_score")
    )


def test_validation_method_order_places_main_last(evaluate_module):
    order = evaluate_module._validation_method_order(
        include_bm25=True,
        include_weighted=True,
        ablation_variant_names=["lambdarank_no_schema", "lambdarank_no_llm_expert"],
    )
    assert order[0] == "cosine_similarity"
    assert order[-1] == "lambdarank_score"
    assert "bm25_score" in order
    assert "weighted_llm_score" in order
    assert "lambdarank_no_schema" in order
    assert "lambdarank_no_llm_expert" in order


def test_order_method_metrics_keeps_unmentioned_methods_at_end(evaluate_module):
    metrics = {
        "extra_metric": {"recall@1": 0.1},
        "cosine_similarity": {"recall@1": 0.5},
        "lambdarank_score": {"recall@1": 0.9},
    }
    ordered = evaluate_module._order_method_metrics(
        metrics,
        preferred_order=["cosine_similarity", "lambdarank_score"],
    )
    assert list(ordered) == ["cosine_similarity", "lambdarank_score", "extra_metric"]


