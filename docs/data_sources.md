# Source documentation

The full catalogue, asset decision, source probes and exclusions are recorded in [the discovery report](../reports/dataset_discovery_report.md).

The baseline recent price adapter uses split-adjusted Yahoo OHLC and retains the dividend-adjusted close independently. The historical BTC run restores Coinbase bucket opens to completed hourly closes. [yfinance history](https://ranaroussi.github.io/yfinance/reference/api/yfinance.PriceHistory.history.html) · [Coinbase candles](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles).

The Apify integration uses the official client to start, resume and page daily capped searches. The submitted run used Coregent's keyword-array schema and two date-stratified requests. Its ranking is retrospective. The configured alternative supports the original Apidojo schema. [Coregent schema](https://apify.com/coregent/x-tweet-scraper/input-schema) · [Run charge audit](../reference/audit/apify_costs.json). [Apify Python client](https://github.com/apify/apify-client-python) · [Actor input schema](https://apify.com/apidojo/tweet-scraper/input-schema).

The news adapters use the providers' Python clients. Tavily retrieves news published in the past week. When configured, Exa supplements it with dated results. News remains context for the outlook. [Tavily Python](https://github.com/tavily-ai/tavily-python) · [Exa Python](https://github.com/exa-labs/exa-py).

The historical X run uses pinned financial-domain model weights locally. ProsusAI lists CC-BY-NC-SA-4.0 for this model. The offline fixture uses the fixed VADER lexicon with six documented crypto term overrides in code. [FinBERT model card](https://huggingface.co/ProsusAI/finbert) · [VADER](https://github.com/cjhutto/vaderSentiment).
