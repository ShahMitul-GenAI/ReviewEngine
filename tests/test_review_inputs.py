import pandas as pd
import pytest

from customer_reviews.review_inputs import normalize_review_dataframe


def test_accepts_xquik_tweet_text_export():
    df, schema = normalize_review_dataframe(
        pd.DataFrame({"Tweet Text": ["Loved the launch", " ", None], "rating": [5, 1, 3]})
    )

    assert df["review"].tolist() == ["Loved the launch"]
    assert df["rating"].tolist() == [5.0]
    assert schema.review_column == "Tweet Text"
    assert schema.dropped_rows == 2


def test_keeps_customer_name_when_present():
    df, schema = normalize_review_dataframe(
        pd.DataFrame({"feedback": ["Great staff"], "user_name": ["Ava"]})
    )

    assert df.to_dict("records") == [{"review": "Great staff", "customer_name": "Ava"}]
    assert schema.customer_column == "user_name"


def test_rejects_missing_review_column():
    with pytest.raises(ValueError, match="review, feedback"):
        normalize_review_dataframe(pd.DataFrame({"message": ["hello"]}))
