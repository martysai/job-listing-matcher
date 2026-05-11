from sara_retrieve_rerank.documents import vacancy_to_metadata, vacancy_to_text


def test_vacancy_to_metadata_stringifies_values():
    vacancy = {
        "dataset_id": 123,
        "title": "ML Engineer",
        "work_format": ["remote"],
        "vacancy_score": None,
    }

    metadata = vacancy_to_metadata(vacancy)

    assert metadata["dataset_id"] == "123"
    assert metadata["title"] == "ML Engineer"
    assert metadata["work_format"] == "remote"
    assert metadata["vacancy_score"] == 0


def test_vacancy_to_text_contains_core_fields():
    vacancy = {
        "title": "ML Engineer",
        "specializations": ["Machine Learning"],
        "tldr_sanitized": "<p>Build ranking systems</p>",
    }

    text = vacancy_to_text(vacancy)

    assert "Title: ML Engineer" in text
    assert "Specializations: Machine Learning" in text
    assert "Description: Build ranking systems" in text
