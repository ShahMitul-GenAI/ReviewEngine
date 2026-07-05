"""Normalize review CSV exports into the ReviewEngine schema."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


REVIEW_COLUMN_ALIASES = {
    "review",
    "review text",
    "review_text",
    "tweet text",
    "tweet_text",
    "text",
    "feedback",
    "comment",
    "content",
    "body",
}

RATING_COLUMN_ALIASES = {
    "rating",
    "stars",
    "score",
}

CUSTOMER_COLUMN_ALIASES = {
    "customer name",
    "customer_name",
    "user name",
    "user_name",
    "author",
    "username",
    "name",
}


@dataclass(frozen=True)
class ReviewSchema:
    review_column: str
    rating_column: str | None
    customer_column: str | None
    total_rows: int
    dropped_rows: int


def _normalize_column_name(column: object) -> str:
    return str(column).strip().lower().replace("_", " ")


def _find_column(columns: pd.Index, aliases: set[str]) -> str | None:
    for column in columns:
        if _normalize_column_name(column) in aliases:
            return str(column)
    return None


def _clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def normalize_review_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, ReviewSchema]:
    review_column = _find_column(df.columns, REVIEW_COLUMN_ALIASES)
    if review_column is None:
        raise ValueError(
            "CSV must include a review, feedback, comment, text, or Xquik Tweet Text column."
        )

    rating_column = _find_column(df.columns, RATING_COLUMN_ALIASES)
    customer_column = _find_column(df.columns, CUSTOMER_COLUMN_ALIASES)
    normalized = pd.DataFrame({"review": _clean_text(df[review_column])})

    if rating_column is not None:
        normalized["rating"] = pd.to_numeric(df[rating_column], errors="coerce")
    if customer_column is not None:
        normalized["customer_name"] = _clean_text(df[customer_column])

    normalized = normalized[normalized["review"] != ""].reset_index(drop=True)
    if normalized.empty:
        raise ValueError("CSV review rows are empty. Add at least 1 non-empty row.")

    return normalized, ReviewSchema(
        review_column=review_column,
        rating_column=rating_column,
        customer_column=customer_column,
        total_rows=len(df),
        dropped_rows=len(df) - len(normalized),
    )
