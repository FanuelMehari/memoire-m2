import polars as pl

HEALTH_CONSULTATION_QUESTION_INDEX = 19

df = pl.read_csv("data/forbidden_question_large_set.csv")

df_new = df.filter(
    df["q_id"] == (
        pl.when(pl.col("content_policy_name") == "Health Consultation").then(HEALTH_CONSULTATION_QUESTION_INDEX)
        .otherwise(0)
    )
).select(["content_policy_name", "question"])

df_new.write_csv("data/forbidden_question_small_set.csv")
