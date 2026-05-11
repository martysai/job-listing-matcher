from sara_retrieve_rerank.preprocessing import clean_html, candidate_to_query, join_list


def test_clean_html_handles_empty_values():
    assert clean_html(None) == ""
    assert clean_html("") == ""


def test_clean_html_converts_tags_to_text():
    assert clean_html("<p>Hello <b>world</b></p>") == "Hello world"


def test_join_list_formats_lists_and_scalars():
    assert join_list(["Python", "ML"]) == "Python, ML"
    assert join_list("Remote") == "Remote"
    assert join_list(None) == ""


def test_candidate_to_query_uses_text_field():
    assert candidate_to_query({"text": "candidate profile"}) == "candidate profile"
    assert candidate_to_query({}) == ""
