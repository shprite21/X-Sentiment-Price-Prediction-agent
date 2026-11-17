# BTC-USD | Experiment C across two market periods

Adding the archived training segment reduced the sentiment Ridge's holdout MAE at **0/5** horizons relative to training only on the recent sample. Validation selected **boosting** among the four pooled candidates. The recent X run remains the primary assessment.

## Cohorts and target construction

Each period was independently aligned, scored and warmed up on its own complete price grid. The last five origins from each training segment require matured within-segment labels. No feature, target or synthetic holding return crosses the gap between 2021 and 2026. Each recent fold receives the same historical observations and only its purged expanding block of recent training origins. Imputation and scaling are fitted again on this union inside every fold. Recent test origins, costs, models and baseline match Experiment A. The historical Coinbase series and recent Yahoo series use different price providers, and the two X sampling mechanisms differ.

## Sentiment distributions

| era | scored_posts | coverage | post_mean | positive_share | negative_share | neutral_share |
| --- | --- | --- | --- | --- | --- | --- |
| Historical | 20243 | 0.2695 | 0.0100 | 0.0559 | 0.0597 | 0.8844 |
| Recent | 401 | 0.0862 | -0.0061 | 0.0835 | 0.1038 | 0.8127 |

Shares and coverage refer to scored samples. Topic mix, ranking, post caps and language selection can change them independently of market sentiment.

## Does historical training help on recent dates?

Positive dollar gains indicate lower absolute error from adding the historical segment.

| phase | model | horizon | mae_usd_recent | mae_usd_pooled | mae_gain_from_archive |
| --- | --- | --- | --- | --- | --- |
| holdout | sentiment_price | 1 | 218.1507 | 218.3314 | -0.1807 |
| holdout | sentiment_price | 2 | 315.2956 | 315.6380 | -0.3424 |
| holdout | sentiment_price | 3 | 389.3931 | 396.7862 | -7.3931 |
| holdout | sentiment_price | 4 | 459.9153 | 472.3920 | -12.4767 |
| holdout | sentiment_price | 5 | 524.1428 | 538.2682 | -14.1254 |

As a second test, both Ridge models were frozen after fitting only historical labels and applied to the same recent origins. The sentiment Ridge holdout results are below. All five horizons and both Ridges are available in historical_only_on_recent.csv.

| horizon | mae_usd | r2_return |
| --- | --- | --- |
| 1 | 434.0588 | -2.1616 |
| 2 | 641.1598 | -2.3947 |
| 3 | 881.1606 | -3.2074 |
| 4 | 1045.0321 | -3.4529 |
| 5 | 1205.5426 | -3.7604 |

## Incremental sentiment effect within pooled training

| phase | horizon | rmse_gain_usd | mae_gain_usd | direction_gain_pp | mae_ci_low | mae_ci_high | block_bars | winning_folds | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| holdout | 1 | -0.5124 | -1.5484 | -2.3578 | -3.8656 | 0.8358 | 9 | 0 | 1 |
| holdout | 2 | -2.7842 | -4.4347 | -2.4965 | -10.5876 | 0.7016 | 9 | 0 | 1 |
| holdout | 3 | -4.8420 | -7.2945 | 0.8322 | -16.7341 | 1.8644 | 9 | 0 | 1 |
| holdout | 4 | -8.8832 | -13.3750 | -3.8835 | -25.3786 | -0.3055 | 9 | 0 | 1 |
| holdout | 5 | -11.5768 | -16.9282 | -3.1900 | -31.3174 | -1.3417 | 9 | 0 | 1 |
| walk_forward | 1 | -0.5084 | -1.1856 | 2.1398 | -3.0030 | 0.5852 | 12 | 1 | 5 |
| walk_forward | 2 | -1.3680 | -3.3424 | 3.7803 | -6.8754 | 0.4107 | 12 | 1 | 5 |
| walk_forward | 3 | -3.3779 | -4.9045 | 1.7118 | -11.6984 | 0.9817 | 12 | 0 | 5 |
| walk_forward | 4 | -4.5260 | -8.6385 | 1.7832 | -18.3986 | 1.9608 | 12 | 1 | 5 |
| walk_forward | 5 | -4.9060 | -9.1505 | 1.7832 | -21.1872 | 2.0250 | 12 | 1 | 5 |

## All horizons and trading results

| phase | model | horizon | n | mae_usd | rmse_usd | r2_price | r2_return | mean_error_usd | direction_accuracy | mae_skill_vs_persistence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| holdout | boosting | 1 | 721 | 214.5454 | 330.6437 | 0.9960 | 0.0029 | 22.2387 | 0.5354 | 0.0036 |
| holdout | boosting | 2 | 721 | 308.1354 | 470.6328 | 0.9919 | -0.0072 | 46.4478 | 0.5049 | -0.0025 |
| holdout | boosting | 3 | 721 | 379.2467 | 580.2633 | 0.9878 | -0.0145 | 72.4213 | 0.4965 | 0.0000 |
| holdout | boosting | 4 | 721 | 446.7479 | 674.6523 | 0.9836 | -0.0188 | 97.6916 | 0.4535 | -0.0003 |
| holdout | boosting | 5 | 721 | 508.6770 | 759.1267 | 0.9794 | -0.0271 | 119.0172 | 0.4286 | -0.0011 |
| holdout | persistence | 1 | 721 | 215.3157 | 331.8077 | 0.9960 | -0.0051 | 23.4820 | 0.0000 | 0.0000 |
| holdout | persistence | 2 | 721 | 307.3822 | 471.1357 | 0.9919 | -0.0108 | 47.4735 | 0.0000 | 0.0000 |
| holdout | persistence | 3 | 721 | 379.2576 | 580.5315 | 0.9878 | -0.0163 | 71.5282 | 0.0000 | 0.0000 |
| holdout | persistence | 4 | 721 | 446.6226 | 675.1846 | 0.9836 | -0.0223 | 96.5022 | 0.0000 | 0.0000 |
| holdout | persistence | 5 | 721 | 508.1108 | 758.7499 | 0.9794 | -0.0275 | 120.1558 | 0.0000 | 0.0000 |
| holdout | price_only | 1 | 721 | 216.7830 | 334.3284 | 0.9959 | -0.0190 | 21.5981 | 0.5215 | -0.0068 |
| holdout | price_only | 2 | 721 | 311.2033 | 476.6355 | 0.9917 | -0.0322 | 44.5680 | 0.5229 | -0.0124 |
| holdout | price_only | 3 | 721 | 389.4917 | 588.1102 | 0.9875 | -0.0412 | 65.1057 | 0.5118 | -0.0270 |
| holdout | price_only | 4 | 721 | 459.0170 | 684.8555 | 0.9831 | -0.0499 | 86.7780 | 0.5354 | -0.0278 |
| holdout | price_only | 5 | 721 | 521.3400 | 768.9246 | 0.9788 | -0.0551 | 108.5450 | 0.5284 | -0.0260 |
| holdout | sentiment_price | 1 | 721 | 218.3314 | 334.8408 | 0.9959 | -0.0232 | 17.6967 | 0.4979 | -0.0140 |
| holdout | sentiment_price | 2 | 721 | 315.6380 | 479.4196 | 0.9916 | -0.0470 | 37.4090 | 0.4979 | -0.0269 |
| holdout | sentiment_price | 3 | 721 | 396.7862 | 592.9522 | 0.9873 | -0.0615 | 56.8483 | 0.5201 | -0.0462 |
| holdout | sentiment_price | 4 | 721 | 472.3920 | 693.7387 | 0.9827 | -0.0813 | 77.2385 | 0.4965 | -0.0577 |
| holdout | sentiment_price | 5 | 721 | 538.2682 | 780.5014 | 0.9782 | -0.0923 | 97.0716 | 0.4965 | -0.0594 |
| walk_forward | boosting | 1 | 1402 | 203.9398 | 315.4381 | 0.9899 | -0.0013 | -10.0249 | 0.4822 | -0.0009 |
| walk_forward | boosting | 2 | 1402 | 286.1636 | 439.8866 | 0.9802 | 0.0000 | -16.4068 | 0.4986 | 0.0005 |
| walk_forward | boosting | 3 | 1402 | 352.1736 | 532.2180 | 0.9709 | 0.0083 | -19.2131 | 0.5093 | 0.0042 |
| walk_forward | boosting | 4 | 1402 | 407.4156 | 611.8400 | 0.9613 | 0.0017 | -30.7912 | 0.5057 | 0.0042 |
| walk_forward | boosting | 5 | 1402 | 462.5065 | 681.8221 | 0.9517 | -0.0045 | -41.0067 | 0.5171 | -0.0006 |
| walk_forward | persistence | 1 | 1402 | 203.7561 | 315.3471 | 0.9899 | -0.0005 | -6.5004 | 0.0000 | 0.0000 |
| walk_forward | persistence | 2 | 1402 | 286.3084 | 440.1750 | 0.9802 | -0.0009 | -12.8407 | 0.0000 | 0.0000 |
| walk_forward | persistence | 3 | 1402 | 353.6705 | 534.6787 | 0.9706 | -0.0014 | -19.3510 | 0.0000 | 0.0000 |
| walk_forward | persistence | 4 | 1402 | 409.1358 | 613.0882 | 0.9611 | -0.0020 | -25.9711 | 0.0000 | 0.0000 |
| walk_forward | persistence | 5 | 1402 | 462.2100 | 681.3153 | 0.9518 | -0.0025 | -32.3896 | 0.0000 | 0.0000 |
| walk_forward | price_only | 1 | 1402 | 205.6828 | 317.3066 | 0.9897 | -0.0140 | -12.9829 | 0.4929 | -0.0095 |
| walk_forward | price_only | 2 | 1402 | 293.1804 | 444.0880 | 0.9798 | -0.0199 | -23.4532 | 0.4793 | -0.0240 |
| walk_forward | price_only | 3 | 1402 | 365.0187 | 542.4149 | 0.9697 | -0.0322 | -34.6327 | 0.4864 | -0.0321 |
| walk_forward | price_only | 4 | 1402 | 423.6256 | 624.4020 | 0.9597 | -0.0416 | -46.3017 | 0.5007 | -0.0354 |
| walk_forward | price_only | 5 | 1402 | 479.7434 | 694.8797 | 0.9498 | -0.0453 | -55.0457 | 0.4929 | -0.0379 |
| walk_forward | sentiment_price | 1 | 1402 | 206.8685 | 317.8150 | 0.9897 | -0.0175 | -9.1801 | 0.5143 | -0.0153 |
| walk_forward | sentiment_price | 2 | 1402 | 296.5228 | 445.4560 | 0.9797 | -0.0267 | -17.2006 | 0.5171 | -0.0357 |
| walk_forward | sentiment_price | 3 | 1402 | 369.9232 | 545.7928 | 0.9694 | -0.0459 | -22.8305 | 0.5036 | -0.0460 |
| walk_forward | sentiment_price | 4 | 1402 | 432.2642 | 628.9280 | 0.9591 | -0.0577 | -31.0922 | 0.5185 | -0.0565 |
| walk_forward | sentiment_price | 5 | 1402 | 488.8938 | 699.7856 | 0.9491 | -0.0615 | -38.2365 | 0.5107 | -0.0577 |

| phase | model | basis | sharpe | sortino | max_drawdown | total_return | annualized_return | bars | trades | win_rate | turnover | exposure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| holdout | boosting | gross | 3.4857 | N/A | 0.0000 | 0.0052 | 0.0646 | 721 | 1 | 1.0000 | 2.0000 | 0.0014 |
| holdout | boosting | net | 3.4857 | N/A | 0.0000 | 0.0032 | 0.0390 | 721 | 1 | 1.0000 | 2.0000 | 0.0014 |
| holdout | persistence | gross | N/A | N/A | 0.0000 | 0.0000 | 0.0000 | 721 | 0 | N/A | 0.0000 | 0.0000 |
| holdout | persistence | net | N/A | N/A | 0.0000 | 0.0000 | 0.0000 | 721 | 0 | N/A | 0.0000 | 0.0000 |
| holdout | buy_hold | gross | -6.5050 | -8.5458 | -0.2619 | -0.2074 | -0.9407 | 721 | 1 | 0.0000 | 2.0000 | 1.0000 |
| holdout | buy_hold | net | -6.5638 | -8.6207 | -0.2619 | -0.2090 | -0.9421 | 721 | 1 | 0.0000 | 2.0000 | 1.0000 |
| holdout | price_only | gross | N/A | N/A | 0.0000 | 0.0000 | 0.0000 | 721 | 0 | N/A | 0.0000 | 0.0000 |
| holdout | price_only | net | N/A | N/A | 0.0000 | 0.0000 | 0.0000 | 721 | 0 | N/A | 0.0000 | 0.0000 |
| holdout | sentiment_price | gross | N/A | N/A | 0.0000 | 0.0000 | 0.0000 | 721 | 0 | N/A | 0.0000 | 0.0000 |
| holdout | sentiment_price | net | N/A | N/A | 0.0000 | 0.0000 | 0.0000 | 721 | 0 | N/A | 0.0000 | 0.0000 |
| walk_forward | boosting | gross | N/A | N/A | 0.0000 | 0.0000 | 0.0000 | 1402 | 0 | N/A | 0.0000 | 0.0000 |
| walk_forward | boosting | net | N/A | N/A | 0.0000 | 0.0000 | 0.0000 | 1402 | 0 | N/A | 0.0000 | 0.0000 |
| walk_forward | persistence | gross | N/A | N/A | 0.0000 | 0.0000 | 0.0000 | 1402 | 0 | N/A | 0.0000 | 0.0000 |
| walk_forward | persistence | net | N/A | N/A | 0.0000 | 0.0000 | 0.0000 | 1402 | 0 | N/A | 0.0000 | 0.0000 |
| walk_forward | buy_hold | gross | 2.1967 | 3.3288 | -0.0810 | 0.1259 | 1.0977 | 1402 | 1 | 1.0000 | 2.0000 | 1.0000 |
| walk_forward | buy_hold | net | 2.1632 | 3.2771 | -0.0810 | 0.1236 | 1.0717 | 1402 | 1 | 1.0000 | 2.0000 | 1.0000 |
| walk_forward | price_only | gross | N/A | N/A | 0.0000 | 0.0000 | 0.0000 | 1402 | 0 | N/A | 0.0000 | 0.0000 |
| walk_forward | price_only | net | N/A | N/A | 0.0000 | 0.0000 | 0.0000 | 1402 | 0 | N/A | 0.0000 | 0.0000 |
| walk_forward | sentiment_price | gross | N/A | N/A | 0.0000 | 0.0000 | 0.0000 | 1402 | 0 | N/A | 0.0000 | 0.0000 |
| walk_forward | sentiment_price | net | N/A | N/A | 0.0000 | 0.0000 | 0.0000 | 1402 | 0 | N/A | 0.0000 | 0.0000 |

Five-bar purges isolate validation and final holdout labels. The selected specification was determined on recent validation returns. There are only two sampled eras, so this check cannot isolate platform drift from price-provider, coverage, regime and sampling changes.
