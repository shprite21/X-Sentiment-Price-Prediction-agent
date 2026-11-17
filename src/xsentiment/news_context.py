"""Select dated, asset-relevant headlines for the outlook and retain retrieval provenance."""

import pandas as pd


def select_news(news, cfg, asof, collection=None):
    asof = pd.Timestamp(asof)
    contexts = []
    cited = []
    publishers = set()
    terms = {term.casefold() for term in (cfg.company, cfg.ticker, *cfg.keywords) if term}
    for row in news.sort_values("published_at", ascending=False).itertuples():
        published = pd.to_datetime(row.published_at, utc=True, errors="coerce")
        title = " ".join(str(row.title).split()).replace("[", "(").replace("]", ")")
        if not any(term in title.casefold() for term in terms):
            continue
        if (
            asof - pd.Timedelta(days=7) <= published <= asof
            and pd.Timestamp(row.retrieved_at) <= asof
            and row.publisher not in publishers
            and str(row.url).startswith(("https://", "http://"))
        ):
            publishers.add(row.publisher)
            contexts.append(f"- {published.date()} · {row.publisher}. [{title}]({row.url})")
            cited.append(
                dict(
                    provider=row.source,
                    publisher=row.publisher,
                    headline=title,
                    url=row.url,
                    published_at=row.published_at,
                    retrieved_at=row.retrieved_at,
                )
            )
            if len(cited) == 3:
                break
    return contexts, dict(
        report_time=asof.isoformat(),
        providers=sorted(news.source.unique().tolist()),
        articles_stored=len(news),
        articles_cited=cited,
        collection=collection,
        predictive_news_columns=[],
    )
