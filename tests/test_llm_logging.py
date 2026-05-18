import json

from sara_retrieve_rerank.llm_logging import (
    setup_llm_jsonl_logger,
    wrap_invoke_with_logging,
)


def test_setup_llm_jsonl_logger_writes_records(tmp_path):
    log_path = tmp_path / "llm.log.jsonl"
    logger = setup_llm_jsonl_logger(log_path, logger_name="sara.test.basic")

    def invoke(_llm, prompt):
        return f"echo: {prompt}"

    wrapped = wrap_invoke_with_logging(invoke, logger=logger, component="unit")

    result = wrapped(object(), "hello")

    assert result == "echo: hello"
    line = log_path.read_text(encoding="utf-8").strip().splitlines()[0]
    record = json.loads(line)
    assert record["component"] == "unit"
    assert record["prompt"] == "hello"
    assert record["response"] == "echo: hello"
    assert isinstance(record["latency_ms"], float)


def test_wrap_invoke_logs_errors_and_reraises(tmp_path):
    log_path = tmp_path / "llm-errors.jsonl"
    logger = setup_llm_jsonl_logger(log_path, logger_name="sara.test.error")

    class BoomError(RuntimeError):
        pass

    def invoke(_llm, _prompt):
        raise BoomError("nope")

    wrapped = wrap_invoke_with_logging(invoke, logger=logger, component="unit")

    try:
        wrapped(object(), "trigger")
    except BoomError:
        pass
    else:
        raise AssertionError("Expected BoomError")

    record = json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert record["component"] == "unit"
    assert "BoomError" in record["error"]


def test_setup_logger_is_idempotent_for_same_path(tmp_path):
    log_path = tmp_path / "idempotent.jsonl"
    logger_a = setup_llm_jsonl_logger(log_path, logger_name="sara.test.idem")
    logger_b = setup_llm_jsonl_logger(log_path, logger_name="sara.test.idem")

    assert logger_a is logger_b
    # No duplicated file handler for the same path.
    file_handlers = [h for h in logger_a.handlers if hasattr(h, "baseFilename")]
    assert sum(1 for _ in file_handlers) == 1
