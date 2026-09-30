# Churn Prediction → Personalised Retention Actions

**Amit Gajbhiye** · a worked example on a simulated multi-vertical delivery app · synthetic data only

## 100-word summary

Churn prediction is only half the problem; the costly decision is which retention action each
user gets. I simulated 50,000 delivery-app users across food, grocery and retail with two randomised
retention campaigns, then built (1) a gradient-boosted 28-day churn model validated on a later
time window (AUC 0.834 vs 0.819 baseline); (2) per-action uplift models for discount, service
recovery and cross-vertical cross-sell; (3) a policy choosing the most profitable action per
user. On a second randomised campaign, risk-targeted discounts lost money while the uplift
policy matched their churn at 31% lower spend with positive net value, which is enough to
justify the A/B test I specify.

## Headline results (evaluated on the later randomised campaign)

| Policy | 28-day churn | Spend / user | Net value vs no action, AED per 1,000 users (95% CI) |
|---|---|---|---|
| No action | 27.9% | 0.00 AED | 0 |
| Discount to everyone | 22.8% | 15.00 AED | -9,241 [-15,077, -3,603] |
| Discount to top-30% risk | 25.9% | 4.50 AED | -2,053 [-3,265, -938] |
| Random action | 25.4% | 7.50 AED | -4,747 [-9,512, -178] |
| **Uplift policy** | **25.9%** | **3.12 AED** | **+2,976 [-1,043, 7,618]** |

The uplift policy's interval crosses zero, so the result is promising, not proven. That is why
the next step is the online A/B test designed in the notebook (13,125 users per arm).

## Files

| File | What it is |
|---|---|
| `Churn_to_Retention_Actions_Amit_Gajbhiye.pdf` | 2-page write-up (start here) |
| `churn_retention.ipynb` | Executed notebook: features, risk model, uplift, policy evaluation, A/B design |
| `data/generate_data.py` | Seeded simulator that produces the dataset |
| `data/*.csv(.gz)` | The generated dataset (orders, users, campaigns, plus simulator ground truth) |
| `outputs/` | Figures and `metrics.json` |
| `narrator_prompt.md` / `narrator_output.md` | LLM prompt that turns `metrics.json` into a stakeholder conclusion, and its output |
| `build_report.py` | Rebuilds the PDF from the notebook outputs |

## Reproduce

```bash
pip install -r requirements.txt
python data/generate_data.py          # ~10 s, seeded
jupyter nbconvert --to notebook --execute --inplace churn_retention.ipynb   # ~3 min
python build_report.py
```

Or open `churn_retention.ipynb` in Google Colab. Upload the `data/` folder first, or run
`generate_data.py` in the first cell.

## Dataset

Self-created and fully synthetic; no real or confidential company data. `data/generate_data.py`
regenerates it exactly (seed 7). `users_latent.csv` holds the simulator's hidden traits. It is
used only to sanity-check the models and never as a model input.