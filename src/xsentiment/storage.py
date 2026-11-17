"""Transactional SQLite repository. A database belongs to one asset and bar specification."""

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.pool import NullPool

schema = sa.MetaData()


def field(name, kind=sa.String, key=False, index=False):
    return sa.Column(name, kind, primary_key=key, index=index)


def table(name, *fields):
    return sa.Table(name, schema, *fields)


raw = table(
    "raw_tweets",
    field("source", key=True),
    field("digest", key=True),
    field("platform"),
    field("original_id", index=True),
    field("acquired_at"),
    field("payload", sa.JSON),
)
posts = table(
    "normalized_posts",
    field("platform", key=True),
    field("id", key=True),
    field("source", key=True),
    field("created_at", index=True),
    field("text"),
    field("likes", sa.Integer),
    field("reposts", sa.Integer),
    field("views", sa.Integer),
    field("followers", sa.Integer),
    field("author_id"),
    field("username"),
    field("author_name"),
    field("post_type"),
    field("original_id"),
    field("original_timestamp"),
    field("method"),
    field("license"),
    field("observed_at"),
    field("available_at"),
    field("processed_at"),
    field("native_label", sa.Float),
)
prices = table(
    "price_bars",
    field("time", key=True),
    *[field(k, sa.Float) for k in ("open", "high", "low", "close", "adj_close", "volume")],
    field("source"),
)
news = table(
    "news_articles",
    field("url", key=True),
    field("title"),
    field("content"),
    field("published_at"),
    field("retrieved_at"),
    field("source"),
    field("publisher"),
)
metadata = table("metadata", field("key", key=True), field("value", sa.JSON))
runs = table(
    "ingestion_runs",
    field("id", key=True),
    field("source", index=True),
    field("status"),
    field("updated_at"),
    field("details", sa.JSON),
)
processed = table(
    "processed_tweets",
    field("experiment", key=True),
    field("id", key=True),
    field("time", index=True),
    field("engine"),
    field("positive", sa.Float),
    field("negative", sa.Float),
    field("neutral", sa.Float),
    field("sentiment", sa.Float),
    field("label"),
    field("source"),
    field("platform"),
    field("processed_at"),
)
sentiment = table(
    "sentiment_features", field("experiment", key=True), field("time", key=True), field("values", sa.JSON)
)
features = table(
    "model_features", field("experiment", key=True), field("time", key=True), field("values", sa.JSON)
)
forecasts = table(
    "predictions",
    field("experiment", key=True),
    field("model", key=True),
    field("time", key=True),
    field("horizon", sa.Integer, key=True),
    field("phase"),
    field("fold", sa.Integer),
    field("target_time"),
    *[
        field(k, sa.Float)
        for k in ("origin_price", "predicted_price", "actual_price", "predicted_return", "actual_return")
    ],
)
metrics = table(
    "evaluation_metrics",
    field("experiment", key=True),
    field("phase", key=True),
    field("model", key=True),
    field("horizon", sa.Integer, key=True),
    field("values", sa.JSON),
)
experiments = table("experiments", field("id", key=True), field("manifest", sa.JSON))


def timestamp():
    return pd.Timestamp.now(tz="UTC").isoformat()


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = sa.create_engine(sa.URL.create("sqlite", database=str(self.path)), poolclass=NullPool)
        existing = set(sa.inspect(self.engine).get_table_names())
        schema.create_all(self.engine)
        if "native_label" not in {
            c["name"] for c in sa.inspect(self.engine).get_columns("normalized_posts")
        }:
            with self.engine.begin() as con:
                con.exec_driver_sql("ALTER TABLE normalized_posts ADD COLUMN native_label FLOAT")
        # One-time, non-destructive migration of the original BTC-only schema.
        if "posts" in existing and "normalized_posts" not in existing:
            for old in ("posts", "prices", "news"):
                self.put(old, pd.read_sql_table(old, self.engine).to_dict("records"))
            self.meta("asset_spec", ["BTC-USD", "24/7", "1h", "UTC"])
        with self.engine.begin() as con:
            if "hourly_features" not in existing:
                con.exec_driver_sql(
                    "CREATE VIEW IF NOT EXISTS hourly_features AS SELECT * FROM model_features"
                )

    def bind(self, cfg):
        spec = [cfg.asset, cfg.exchange, cfg.interval, cfg.timezone]
        with self.engine.begin() as con:
            con.execute(insert(metadata).values(key="asset_spec", value=spec).on_conflict_do_nothing())
            if con.scalar(sa.select(metadata.c.value).where(metadata.c.key == "asset_spec")) != spec:
                raise ValueError("Database asset/calendar/interval differs from configuration")

    @contextmanager
    def connect(self):
        con = self.engine.raw_connection()
        try:
            yield con
            con.commit()
        except BaseException:
            con.rollback()
            raise
        finally:
            con.close()

    def upsert(self, target, rows, connection=None):
        if not rows:
            return
        if connection is None:
            with self.engine.begin() as con:
                self.upsert(target, rows, con)
            return
        stmt = insert(target)
        changes = {c.name: stmt.excluded[c.name] for c in target.columns if not c.primary_key}
        statement = stmt.on_conflict_do_update(index_elements=list(target.primary_key), set_=changes)
        if target in (raw, posts):
            statement = stmt.on_conflict_do_nothing()
        for start in range(0, len(rows), 2000):
            connection.execute(
                statement,
                [{c.name: row.get(c.name) for c in target.columns} for row in rows[start : start + 2000]],
            )

    def save_raw(self, items, source, platform, observed):
        rows = []
        for item in items:
            digest = hashlib.sha256(
                json.dumps(item, sort_keys=True, ensure_ascii=False, default=str).encode()
            ).hexdigest()
            rows.append(
                dict(
                    source=source,
                    digest=digest,
                    platform=platform,
                    original_id=item.get("id") or item.get("id_str"),
                    acquired_at=observed,
                    payload=item,
                )
            )
        self.upsert(raw, rows)

    def put(self, name, rows):
        target = {"posts": posts, "prices": prices, "news": news}[name]
        if name == "posts":
            rows = [{"platform": "X", "method": "legacy_snapshot", **row} for row in rows]
        self.upsert(target, rows)

    def frame(self, name):
        target = {"posts": posts, "prices": prices, "news": news}[name]
        return pd.read_sql(sa.select(target), self.engine)

    def meta(self, key, value):
        self.upsert(metadata, [{"key": key, "value": value}])

    def metadata(self):
        with self.engine.connect() as con:
            return dict(con.execute(sa.select(metadata)).tuples().all())

    def record(self, run_id, source, status, details):
        self.upsert(
            runs, [dict(id=run_id, source=source, status=status, updated_at=timestamp(), details=details)]
        )
