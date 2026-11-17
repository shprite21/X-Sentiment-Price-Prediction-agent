# BTC-USD | Historical X research

**REAL HISTORICAL**. Adding X sentiment to Ridge reduced walk-forward MAE at 1/5 horizons and RMSE at 3/5. On the final holdout the counts were 0/5 and 0/5. At 0/5 horizons the paired holdout interval for MAE reduction lay entirely above zero. The sample provides no clear evidence of a repeatable reduction in absolute forecast error.

**Validation selected boosting** by pooled forward log-return RMSE with equal weight per origin and horizon. Selection scores used only walk-forward labels maturing before the final holdout. The model was then frozen for the final block.

## Data and availability

The 1h sample contains 4,320 completed bars from 2021-02-06T01:00:00+00:00 through 2021-08-05T00:00:00+00:00. 24,834 source posts yielded 20,243 scored observations. ID duplicates within the aligned window totalled 0, repeated cleaned texts 1,148, and empty or off-topic texts 3,443. After 72 bars of warmup, 4,248 origins remained, including 3,103 without posts. Coverage was 27.0%. 4,243 origins had all five matured targets. Price sources: coinbase.

The archive contains 4,693,091 CSV records. The parser rejected 3,803 malformed rows. 720,378 valid originals fell inside the window. A seeded hash sample retained up to 20 per UTC hour before scoring. The cap changes the interpretation of volume. Author display names are an imperfect proxy for identity. The source supplies no post IDs, language field or reliable per-post engagement. Timestamp plus exact text identifies a post.

Posts are grouped strictly before each close. A reported availability time delays a post. The run used the publication policy. Publication-time reconstruction assumes archived text could have been read at publication. Retrospective deletions, edits and collection delays remain unobservable. First-seen mode requires a collector observation before the origin. Engagement, follower snapshots, exchange volume and news are excluded from both model matrices.

## Sentiment and supervised models

The sentiment engine was finbert. FinBERT uses positive minus negative softmax probability and the most likely class. The lightweight VADER option uses normalized lexical weights and a ±0.05 polarity band. StockTwits uses its own binary label. Missing intervals retain undefined sentiment, zero sampled posts and an explicit presence indicator. Returns, acceleration, trading ranges, rolling extremes, volatility and moving-average ratios use trailing prices. Sentiment aggregates include dispersion, proportions, imbalance, lags, momentum and log counts. Observed neutral posts remain in their interval.

Five direct targets are log(close[t+h]/close[t]). Multiplying the observed n+0 close by the exponential of each predicted return gives the price forecast. Random walk, price Ridge, price + sentiment Ridge (alpha 100) and five regularized gradient-boosting estimators use identical origins. Ridge fits median imputation, missingness indicators and scaling separately on each training fold. Boosting handles missing values natively, with 45 iterations, at most seven leaves and fixed seed. Observable future closes provide supervised labels. Learning a reinforcement policy would require an execution and reward environment that these data do not identify.

## Expanding-window evaluation

The table gives realized fold sizes. Each fit purges five origins plus 0 embargo bars. The last development test label also precedes the first final holdout origin. Model settings were fixed. The final block is held out within this protocol. This repository had previously evaluated an earlier specification on the historical period, which limits its status as independently unseen evidence.

| fold | phase | train_rows | test_rows | last_train_label | test_start | last_test_label |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | walk_forward | 2110 | 336 | 2021-05-08T03:00:00+00:00 | 2021-05-08T04:00:00+00:00 | 2021-05-22T08:00:00+00:00 |
| 2 | walk_forward | 2446 | 336 | 2021-05-22T03:00:00+00:00 | 2021-05-22T04:00:00+00:00 | 2021-06-05T08:00:00+00:00 |
| 3 | walk_forward | 2782 | 336 | 2021-06-05T03:00:00+00:00 | 2021-06-05T04:00:00+00:00 | 2021-06-19T08:00:00+00:00 |
| 4 | walk_forward | 3118 | 336 | 2021-06-19T03:00:00+00:00 | 2021-06-19T04:00:00+00:00 | 2021-07-03T08:00:00+00:00 |
| 5 | walk_forward | 3454 | 58 | 2021-07-03T03:00:00+00:00 | 2021-07-03T04:00:00+00:00 | 2021-07-05T18:00:00+00:00 |
| 6 | holdout | 3517 | 721 | 2021-07-05T18:00:00+00:00 | 2021-07-05T19:00:00+00:00 | 2021-08-05T00:00:00+00:00 |

## Incremental value of sentiment

Adding X sentiment to Ridge reduced walk-forward MAE at 1/5 horizons and RMSE at 3/5. On the final holdout the counts were 0/5 and 0/5. At 0/5 horizons the paired holdout interval for MAE reduction lay entirely above zero. The sample provides no clear evidence of a repeatable reduction in absolute forecast error. Positive gains favor adding sentiment. Direction gains are percentage points. Paired 95% percentile intervals use 500 circular moving-block resamples of the absolute loss difference. Block lengths exceed or equal the forecast horizon. Intervals are descriptive and unadjusted for testing multiple horizons. Fold wins count lower MAE.

| phase | horizon | rmse_gain_usd | mae_gain_usd | direction_gain_pp | mae_ci_low | mae_ci_high | block_bars | winning_folds | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| holdout | 1 | -1.0493 | -1.9531 | -1.6644 | -3.2977 | -0.6764 | 9 | 0 | 1 |
| holdout | 2 | -3.0352 | -4.0545 | -1.8031 | -6.6848 | -1.2930 | 9 | 0 | 1 |
| holdout | 3 | -6.9227 | -7.8900 | -1.5257 | -13.6581 | -2.7175 | 9 | 0 | 1 |
| holdout | 4 | -9.2713 | -9.9545 | -2.9126 | -17.5028 | -2.3904 | 9 | 0 | 1 |
| holdout | 5 | -6.1175 | -8.6574 | -2.4965 | -19.8790 | 1.7228 | 9 | 0 | 1 |
| walk_forward | 1 | -0.2988 | -0.9889 | -0.1427 | -2.1551 | 0.1467 | 12 | 1 | 5 |
| walk_forward | 2 | -0.0960 | -0.8886 | 1.4979 | -3.3669 | 1.7282 | 12 | 1 | 5 |
| walk_forward | 3 | 0.1991 | -0.2578 | 1.7118 | -5.6260 | 4.3764 | 12 | 2 | 5 |
| walk_forward | 4 | 0.6526 | 0.1512 | 0.7846 | -6.7289 | 6.5339 | 12 | 2 | 5 |
| walk_forward | 5 | 0.1044 | -0.6522 | -0.8559 | -8.3568 | 8.2755 | 12 | 2 | 5 |

## Out-of-sample forecast errors

MAE, RMSE and mean signed error use price units. Both price-level and return R² are reported because a persistent price level can produce a high R² with little return predictability. Direction is exact sign agreement, including a zero-return forecast. Positive MAE skill indicates an improvement over random walk.

| phase | model | horizon | n | mae_usd | rmse_usd | r2_price | r2_return | mean_error_usd | direction_accuracy | mae_skill_vs_persistence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| holdout | boosting | 1 | 721 | 173.4086 | 259.4971 | 0.9944 | -0.0031 | -11.8962 | 0.5104 | -0.0014 |
| holdout | boosting | 2 | 721 | 244.5051 | 369.4268 | 0.9886 | -0.0068 | -24.2922 | 0.4924 | -0.0045 |
| holdout | boosting | 3 | 721 | 308.9988 | 461.9573 | 0.9823 | -0.0166 | -42.0389 | 0.4965 | -0.0101 |
| holdout | boosting | 4 | 721 | 365.0542 | 541.7618 | 0.9757 | -0.0168 | -47.0998 | 0.5257 | -0.0096 |
| holdout | boosting | 5 | 721 | 403.0171 | 605.9963 | 0.9697 | -0.0144 | -47.7393 | 0.5506 | -0.0021 |
| holdout | persistence | 1 | 721 | 173.1697 | 259.2189 | 0.9944 | -0.0009 | -7.9825 | 0.0000 | 0.0000 |
| holdout | persistence | 2 | 721 | 243.3995 | 368.8850 | 0.9887 | -0.0018 | -15.7746 | 0.0000 | 0.0000 |
| holdout | persistence | 3 | 721 | 305.8973 | 459.2232 | 0.9825 | -0.0027 | -24.0606 | 0.0000 | 0.0000 |
| holdout | persistence | 4 | 721 | 361.5968 | 538.5577 | 0.9760 | -0.0034 | -31.7527 | 0.0000 | 0.0000 |
| holdout | persistence | 5 | 721 | 402.1853 | 603.7972 | 0.9699 | -0.0042 | -39.7314 | 0.0000 | 0.0000 |
| holdout | price_only | 1 | 721 | 173.6183 | 261.6410 | 0.9943 | -0.0188 | -13.6474 | 0.5187 | -0.0026 |
| holdout | price_only | 2 | 721 | 245.3876 | 374.0141 | 0.9884 | -0.0280 | -26.1263 | 0.5173 | -0.0082 |
| holdout | price_only | 3 | 721 | 308.8347 | 465.6207 | 0.9820 | -0.0296 | -36.1038 | 0.5035 | -0.0096 |
| holdout | price_only | 4 | 721 | 363.8849 | 546.0109 | 0.9753 | -0.0297 | -44.7607 | 0.5298 | -0.0063 |
| holdout | price_only | 5 | 721 | 407.0245 | 611.1359 | 0.9692 | -0.0267 | -52.4434 | 0.5340 | -0.0120 |
| holdout | sentiment_price | 1 | 721 | 175.5715 | 262.6903 | 0.9942 | -0.0276 | -12.2348 | 0.5021 | -0.0139 |
| holdout | sentiment_price | 2 | 721 | 249.4421 | 377.0493 | 0.9882 | -0.0469 | -31.0246 | 0.4993 | -0.0248 |
| holdout | sentiment_price | 3 | 721 | 316.7247 | 472.5434 | 0.9815 | -0.0653 | -44.1149 | 0.4882 | -0.0354 |
| holdout | sentiment_price | 4 | 721 | 373.8394 | 555.2822 | 0.9745 | -0.0707 | -49.9473 | 0.5007 | -0.0339 |
| holdout | sentiment_price | 5 | 721 | 415.6820 | 617.2534 | 0.9686 | -0.0532 | -52.1367 | 0.5090 | -0.0336 |
| walk_forward | boosting | 1 | 1402 | 332.4378 | 488.8417 | 0.9949 | -0.0171 | 14.0464 | 0.5064 | -0.0089 |
| walk_forward | boosting | 2 | 1402 | 477.0485 | 686.0731 | 0.9899 | 0.0007 | 34.3040 | 0.5014 | 0.0042 |
| walk_forward | boosting | 3 | 1402 | 581.7223 | 823.7358 | 0.9854 | -0.0011 | 50.1867 | 0.5150 | 0.0016 |
| walk_forward | boosting | 4 | 1402 | 662.5846 | 931.6226 | 0.9813 | 0.0074 | 54.1818 | 0.5314 | 0.0043 |
| walk_forward | boosting | 5 | 1402 | 734.9394 | 1013.0814 | 0.9777 | 0.0045 | 56.2890 | 0.5571 | 0.0068 |
| walk_forward | persistence | 1 | 1402 | 329.4961 | 484.2214 | 0.9950 | -0.0010 | 17.5163 | 0.0000 | 0.0000 |
| walk_forward | persistence | 2 | 1402 | 479.0638 | 686.4273 | 0.9899 | -0.0019 | 34.9282 | 0.0000 | 0.0000 |
| walk_forward | persistence | 3 | 1402 | 582.6589 | 823.7719 | 0.9854 | -0.0030 | 52.2952 | 0.0000 | 0.0000 |
| walk_forward | persistence | 4 | 1402 | 665.4787 | 937.7936 | 0.9810 | -0.0042 | 69.5593 | 0.0000 | 0.0000 |
| walk_forward | persistence | 5 | 1402 | 739.9375 | 1018.5501 | 0.9775 | -0.0055 | 87.2539 | 0.0000 | 0.0000 |
| walk_forward | price_only | 1 | 1402 | 329.1050 | 486.2248 | 0.9950 | -0.0111 | 10.1542 | 0.5257 | 0.0012 |
| walk_forward | price_only | 2 | 1402 | 478.3974 | 687.3254 | 0.9899 | -0.0060 | 20.6647 | 0.5164 | 0.0014 |
| walk_forward | price_only | 3 | 1402 | 580.6386 | 825.7598 | 0.9854 | -0.0087 | 33.9343 | 0.5200 | 0.0035 |
| walk_forward | price_only | 4 | 1402 | 665.4233 | 939.0759 | 0.9810 | -0.0078 | 47.6227 | 0.5178 | 0.0001 |
| walk_forward | price_only | 5 | 1402 | 737.1309 | 1024.2224 | 0.9772 | -0.0165 | 61.7180 | 0.5399 | 0.0038 |
| walk_forward | sentiment_price | 1 | 1402 | 330.0939 | 486.5236 | 0.9950 | -0.0130 | 5.1266 | 0.5243 | -0.0018 |
| walk_forward | sentiment_price | 2 | 1402 | 479.2859 | 687.4214 | 0.9899 | -0.0074 | 10.9799 | 0.5314 | -0.0005 |
| walk_forward | sentiment_price | 3 | 1402 | 580.8963 | 825.5607 | 0.9854 | -0.0096 | 18.4091 | 0.5371 | 0.0030 |
| walk_forward | sentiment_price | 4 | 1402 | 665.2721 | 938.4233 | 0.9810 | -0.0081 | 28.8341 | 0.5257 | 0.0003 |
| walk_forward | sentiment_price | 5 | 1402 | 737.7832 | 1024.1180 | 0.9772 | -0.0187 | 39.4003 | 0.5314 | 0.0029 |

![Model comparison and fold performance](figures/model_evaluation.png)

![Actual versus predicted target closes](figures/predicted_actual.png)

## Trading simulation

The one-bar forecast sets a long/cash position. Each training fold sets the entry threshold to the larger of the median absolute one-bar training log return and two one-way costs. Enter at the next bar open, liquidate at its close, with unit maximum exposure and no borrowing. One-way cost is 10 bp including 2 bp of slippage. Each active bar is one round trip, even for consecutive signals. The historical opening print is a fill assumption, especially for adjacent crypto bars. Buy-and-hold enters once and exits once per phase. Sharpe uses sample standard deviation and 8760 periods per year. Sortino uses the root mean squared negative return across all bars. Both use a zero hurdle. Drawdown includes initial wealth. Gross and net are reported separately. Annualized growth rates extrapolate short samples.

**Selected model, final holdout, net of costs**

| phase | model | basis | sharpe | sortino | max_drawdown | total_return | annualized_return | bars | trades | win_rate | turnover | exposure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| holdout | boosting | net | N/A | N/A | 0.0000 | 0.0000 | 0.0000 | 721 | 0 | N/A | 0.0000 | 0.0000 |

| phase | model | basis | sharpe | sortino | max_drawdown | total_return | annualized_return | bars | trades | win_rate | turnover | exposure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| holdout | boosting | gross | N/A | N/A | 0.0000 | 0.0000 | 0.0000 | 721 | 0 | N/A | 0.0000 | 0.0000 |
| holdout | boosting | net | N/A | N/A | 0.0000 | 0.0000 | 0.0000 | 721 | 0 | N/A | 0.0000 | 0.0000 |
| holdout | persistence | gross | N/A | N/A | 0.0000 | 0.0000 | 0.0000 | 721 | 0 | N/A | 0.0000 | 0.0000 |
| holdout | persistence | net | N/A | N/A | 0.0000 | 0.0000 | 0.0000 | 721 | 0 | N/A | 0.0000 | 0.0000 |
| holdout | buy_hold | gross | 3.1678 | 4.6796 | -0.1590 | 0.1693 | 5.6874 | 721 | 1 | 1.0000 | 2.0000 | 1.0000 |
| holdout | buy_hold | net | 3.1320 | 4.6259 | -0.1590 | 0.1670 | 5.5268 | 721 | 1 | 1.0000 | 2.0000 | 1.0000 |
| holdout | price_only | gross | N/A | N/A | 0.0000 | 0.0000 | 0.0000 | 721 | 0 | N/A | 0.0000 | 0.0000 |
| holdout | price_only | net | N/A | N/A | 0.0000 | 0.0000 | 0.0000 | 721 | 0 | N/A | 0.0000 | 0.0000 |
| holdout | sentiment_price | gross | N/A | N/A | 0.0000 | 0.0000 | 0.0000 | 721 | 0 | N/A | 0.0000 | 0.0000 |
| holdout | sentiment_price | net | N/A | N/A | 0.0000 | 0.0000 | 0.0000 | 721 | 0 | N/A | 0.0000 | 0.0000 |
| walk_forward | boosting | gross | 1.7244 | 2.6257 | -0.0708 | 0.0872 | 0.6856 | 1402 | 27 | 0.5556 | 54.0000 | 0.0193 |
| walk_forward | boosting | net | 0.7221 | 1.0290 | -0.0745 | 0.0300 | 0.2027 | 1402 | 27 | 0.5185 | 54.0000 | 0.0193 |
| walk_forward | persistence | gross | N/A | N/A | 0.0000 | 0.0000 | 0.0000 | 1402 | 0 | N/A | 0.0000 | 0.0000 |
| walk_forward | persistence | net | N/A | N/A | 0.0000 | 0.0000 | 0.0000 | 1402 | 0 | N/A | 0.0000 | 0.0000 |
| walk_forward | buy_hold | gross | -2.3730 | -3.1922 | -0.5082 | -0.4220 | -0.9675 | 1402 | 1 | 0.0000 | 2.0000 | 1.0000 |
| walk_forward | buy_hold | net | -2.3839 | -3.2067 | -0.5082 | -0.4232 | -0.9679 | 1402 | 1 | 0.0000 | 2.0000 | 1.0000 |
| walk_forward | price_only | gross | 0.7689 | 1.4342 | -0.0284 | 0.0150 | 0.0976 | 1402 | 2 | 0.5000 | 4.0000 | 0.0014 |
| walk_forward | price_only | net | 0.5857 | 1.0100 | -0.0303 | 0.0110 | 0.0705 | 1402 | 2 | 0.5000 | 4.0000 | 0.0014 |
| walk_forward | sentiment_price | gross | 0.7689 | 1.4342 | -0.0284 | 0.0150 | 0.0976 | 1402 | 2 | 0.5000 | 4.0000 | 0.0014 |
| walk_forward | sentiment_price | net | 0.5857 | 1.0100 | -0.0303 | 0.0110 | 0.0705 | 1402 | 2 | 0.5000 | 4.0000 | 0.0014 |

![Net equity and drawdown](figures/strategy_equity_drawdown.png)

## Limitations and reproduction

Capped sampling, sparse coverage, changing regimes and domain mismatch can swamp small sentiment effects. Repeated texts, multilingual posts, quoted prices and bots complicate interpretation. FinBERT scores in this retrospective reconstruction use a pinned model artifact distributed after the 2021 sample. They do not establish that this exact text model was deployable at each origin. Correlation does not identify information arrival or causation. A historical backtest cannot establish present market sentiment. News is retrieved separately for dated outlooks.

The manifest records the exact feature order, model parameters, input and software hashes, provider runs, forecast cutoffs and engine version. The README gives the offline replay, public-source rebuild and credential-gated collection commands. See forecast_metrics.csv, fold_metrics.csv, incremental.csv, trades.csv and the forecast matrix for machine-readable results.

## Other experiments in this submission

| Experiment | Evidence available | Finding |
|---|---|---|
| A · Recent X and price | 180 days of recent Yahoo prices. Current news providers: tavily. Authenticated X collection pending. | Current X sentiment and its incremental value are unmeasured. |
| B · Historical X and price | FinBERT-scored Twitter sample and Coinbase hourly bars | Adding sentiment improved Ridge MAE at 1/5 development horizons and 0/5 holdout horizons. |
| C · Historical and recent X | No validated recent X corpus overlapping the historical source transition | No pooled result or evidence of cross-period generalization. |
| D · Supplementary StockTwits | 180 daily Coinbase bars and 2,140 aligned self-labelled posts | 103 labelled origins, 3 development folds, 17 holdout origins. Validation selected price_only. Adding StockTwits labels reduced holdout Ridge MAE at 0/5 horizons. |

The independent recent price-only control selected persistence and supplies the latest available projections. See the [current outlook](sentiment_outlook.md), [StockTwits evaluation](stocktwits/evaluation_report.md), [catalogue review](dataset_discovery_report.md) and [historical run tables](historical/forecast_metrics.csv).
