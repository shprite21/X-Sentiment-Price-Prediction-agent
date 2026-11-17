# Submission checklist

- Share the private GitHub repository with the evaluator's confirmed GitHub account.
- Include the primary recent evaluation, one-page sentiment outlook, dataset-discovery report, figures and checksum manifest. The historical X, cross-period and supplementary StockTwits reports have separate links.
- For offline assessment run the tests, synthetic demo and published feature replays. Review the five-bar purges, paired Ridge differences and held-out strategy results.
- Provide the Actor ID `coregent~x-tweet-scraper`, `config/apify_coregent.json`, two recorded run IDs and the data cutoff from the reproducibility manifest. The two completed runs together reported USD 0.84.
- Rotate the Apify and Tavily credentials shared during setup. If an evaluator needs live access, provide scoped credentials through an agreed private secret-sharing channel. The read-only restore requires access to the recorded Apify datasets while the provider retains them. Optional Exa live collection requires a separate Exa key.
- Keep the private database, raw posts, user identifiers, model binaries and API credentials outside Git. Check access and licensing before redistributing original provider data.
