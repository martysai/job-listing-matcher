from __future__ import annotations

from langchain_core.documents import Document

from sara_retrieve_rerank.preprocessing import clean_html, join_list


def vacancy_to_text(vacancy: dict) -> str:
    """Build the text that is embedded and stored for each vacancy."""
    parts = [
        f"Title: {vacancy.get('title', '')}",
        f"Original title: {vacancy.get('original_title', '')}",
        f"Work format: {join_list(vacancy.get('work_format'))}",
        f"Work type: {vacancy.get('work_type', '')}",
        f"English level: {vacancy.get('english_level', '')}",
        f"Employee type: {join_list(vacancy.get('employee_type'))}",
        f"Company type: {vacancy.get('company_type', '')}",
        f"Tags: {join_list(vacancy.get('tags'))}",
        f"Grades: {join_list(vacancy.get('grades'))}",
        f"Specializations: {join_list(vacancy.get('specializations'))}",
        f"Regions: {join_list(vacancy.get('regions'))}",
        f"Cities: {join_list(vacancy.get('cities'))}",
        f"Company domains: {join_list(vacancy.get('company_domains'))}",
        f"Title aliases: {join_list(vacancy.get('title_aliases'))}",
        f"Description: {clean_html(vacancy.get('tldr_sanitized') or vacancy.get('text_sanitized'))}",
    ]

    return "\n".join(part for part in parts if part.strip())


def vacancy_to_metadata(vacancy: dict) -> dict:
    """Build Chroma-compatible metadata for a vacancy document."""
    return {
        "dataset_id": str(vacancy.get("dataset_id", "")),
        "title": str(vacancy.get("title", "")),
        "work_type": str(vacancy.get("work_type") or ""),
        "work_format": join_list(vacancy.get("work_format")),
        "english_level": str(vacancy.get("english_level") or ""),
        "employee_type": join_list(vacancy.get("employee_type")),
        "regions": join_list(vacancy.get("regions")),
        "cities": join_list(vacancy.get("cities")),
        "specializations": join_list(vacancy.get("specializations")),
        "tags": join_list(vacancy.get("tags")),
        "vacancy_score": vacancy.get("vacancy_score") or 0,
    }


def create_vacancy_documents(vacancies: list[dict]) -> list[Document]:
    """Convert raw vacancy records into LangChain documents."""
    return [
        Document(
            page_content=vacancy_to_text(vacancy),
            metadata=vacancy_to_metadata(vacancy),
        )
        for vacancy in vacancies
    ]
