#!/usr/bin/env python3
"""Build the 2-page PDF write-up from outputs/metrics.json and the notebook's figures.

Run after executing churn_retention.ipynb:  python build_report.py
"""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

HERE = Path(__file__).parent
OUT = HERE / "outputs"
PDF = HERE / "Churn_to_Retention_Actions_Amit_Gajbhiye.pdf"
NAVY, TEAL, INK, GREY = (colors.HexColor(c) for c in ("#1B3A5B", "#0B7C8C", "#1A222C", "#525E6B"))

m = json.loads((OUT / "metrics.json").read_text())
pol = {p["policy"]: p for p in m["policies"]}
risk = m["risk_model"]
up = pol["Uplift policy (this notebook)"]
top30 = pol["Discount to top-30% risk"]
spend_cut = 1 - up["spend_per_user_aed"] / top30["spend_per_user_aed"]

SUMMARY = (
    "Churn prediction is only half the problem; the costly decision is which retention action "
    "each user gets. I simulated 50,000 delivery-app users across food, grocery and retail with two "
    "randomised retention campaigns, then built (1) a gradient-boosted 28-day churn model "
    f"validated on a later time window (AUC {risk['Gradient boosting (all features)']['roc_auc']:.3f} "
    f"vs {risk['RFM logistic baseline']['roc_auc']:.3f} baseline); (2) per-action uplift models for "
    "discount, service recovery and cross-vertical cross-sell; (3) a policy choosing the most "
    "profitable action per user. On a second randomised campaign, risk-targeted discounts lost "
    f"money while the uplift policy matched their churn at {spend_cut:.0%} lower spend with "
    "positive net value, which is enough to justify the A/B test I specify."
)

s = {
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=15, leading=18, textColor=NAVY),
    "sub": ParagraphStyle("sub", fontName="Helvetica", fontSize=8.8, leading=11.5, textColor=GREY,
                          spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=10.2, leading=12.5,
                         textColor=NAVY, spaceBefore=7, spaceAfter=3),
    "p": ParagraphStyle("p", fontName="Helvetica", fontSize=8.9, leading=12, textColor=INK,
                        spaceAfter=4),
    "b": ParagraphStyle("b", fontName="Helvetica", fontSize=8.9, leading=12, textColor=INK,
                        leftIndent=10, firstLineIndent=-7, spaceAfter=2.5),
    "cap": ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=7.8, leading=10,
                          textColor=GREY, spaceAfter=4),
}


def bullets(items):
    return [Paragraph(f"• {t}", s["b"]) for t in items]


def aed(x):
    return f"{x:+,.0f}" if x else "0"


rows = [["Policy", "28-day churn", "Spend / user", "Net value vs no action\n(AED per 1,000 users, 95% CI)"]]
for name, p in pol.items():
    ci = "" if name == "No action" else f"  [{p['ci_low']:,.0f}, {p['ci_high']:,.0f}]"
    rows.append([name, f"{p['churn_rate']:.1%}", f"{p['spend_per_user_aed']:.2f} AED",
                 aed(p["incremental_value_per_1k_users_aed"]) + ci])
table = Table(rows, colWidths=[52 * mm, 24 * mm, 24 * mm, 70 * mm])
table.setStyle(TableStyle([
    ("FONT", (0, 0), (-1, -1), "Helvetica", 8.3),
    ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.3),
    ("TEXTCOLOR", (0, 0), (-1, 0), NAVY),
    ("LINEBELOW", (0, 0), (-1, 0), 0.6, TEAL),
    ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#D5DBE1")),
    ("BACKGROUND", (0, len(rows) - 1), (-1, len(rows) - 1), colors.HexColor("#E7F3F4")),
    ("FONT", (0, len(rows) - 1), (-1, len(rows) - 1), "Helvetica-Bold", 8.3),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
]))

mix = m["uplift_policy_action_mix"]
ab = m["ab_test"]
story = [
    Paragraph("Churn Prediction to Personalised Retention Actions", s["h1"]),
    Paragraph("A worked example on a simulated multi-vertical delivery app  |  Amit Gajbhiye  |  "
              "Synthetic data only", s["sub"]),
    Paragraph("Summary", s["h2"]),
    Paragraph(SUMMARY, s["p"]),
    Paragraph("Approach", s["h2"]),
    *bullets([
        "<b>Data.</b> A seeded simulator of 50,000 users over 150 days across food, grocery and "
        "retail in five cities. It plants known churn drivers: late deliveries, low ratings, "
        "single-vertical usage, price sensitivity and tenure. Randomised campaigns on day 90 and "
        "day 120 assign control, discount (15 AED), service recovery (10 AED) or cross-sell into "
        "a new vertical (5 AED).",
        "<b>Label and split.</b> Churn means no order in any vertical in the 28 days after the "
        "cutoff. Features use only data up to the cutoff, and the code asserts this. The model "
        "trains at day 90 and tests at day 120, so the test is always a future window.",
        "<b>Risk model.</b> Gradient boosting over recency, frequency, trend, cross-vertical "
        "breadth, basket value, discount reliance and service quality, compared against a "
        f"recency-frequency baseline. Top-decile lift is "
        f"{risk['Gradient boosting (all features)']['top_decile_lift']:.2f}x.",
        "<b>Uplift and policy.</b> A T-learner, with one churn model per arm, estimates how much "
        "each action lowers each user's churn probability. Each user gets the action with the "
        "highest <i>uplift x 90-day margin - cost</i>, or none if that is not positive. Policies "
        "are evaluated on the day-120 randomised campaign with inverse propensity weighting, and "
        "confidence intervals come from 500 bootstrap resamples.",
    ]),
    Paragraph("Results on the later randomised campaign", s["h2"]),
    table,
    Spacer(1, 3),
    Paragraph(f"The uplift policy leaves {mix.get('control', 0):.0%} of users alone and sends "
              f"cross-sell to {mix.get('cross_sell', 0):.0%}, service recovery to "
              f"{mix.get('service_recovery', 0):.0%} and discounts to {mix.get('discount', 0):.0%}. "
              "Targeting by risk alone pays for users who would have stayed anyway. The uplift "
              "policy's interval crosses zero, so it is promising, not proven.", s["cap"]),
    Image(str(OUT / "fig2_policies.png"), width=170 * mm, height=60 * mm),
    KeepTogether([
        Paragraph("Does the uplift model rank users correctly?", s["h2"]),
        Image(str(OUT / "fig3_uplift_validation.png"), width=120 * mm, height=62 * mm),
        Paragraph("Observed churn reduction in randomised data, by third of predicted uplift. It "
                  "is higher in the high third than the low third for every action, so the "
                  "ranking holds, though service recovery dips in the middle third. The model does overstate the size of effects: for the top third on "
                  "discount it predicted 13.8 points and 8.8 were observed. Estimates should be "
                  "shrunk before they drive spend.", s["cap"]),
    ]),
    Paragraph("Next step: the online A/B test", s["h2"]),
    *bullets([
        "<b>Design:</b> users randomised by a stable ID hash, stratified by city and number of "
        "verticals used. The current rule (discount to top-30% risk) runs against the uplift "
        "policy, with a small untreated holdout to keep measuring uplift.",
        f"<b>Size:</b> {ab['users_per_arm']:,} users per arm to detect a {ab['mde_abs']:.1%} "
        f"absolute change from {ab['baseline_churn']:.1%} churn (alpha 0.05, power 0.8), read "
        "after one full 28-day window with no peeking.",
        "<b>Primary metric:</b> 28-day retention in any vertical. <b>Guardrails:</b> incentive "
        "spend and margin per user, order value, delivery complaints and opt-outs.",
    ]),
    Paragraph("Limits", s["h2"]),
    *bullets([
        "The data is simulated, and the effects were planted. Recovering them shows the method "
        "works, not how real users behave.",
        "The 90-day margin proxy ignores cannibalised orders, long-term value and incentive "
        "fatigue.",
        "The T-learner is the simplest uplift method. With real data I would compare X-learner "
        "and DR-learner variants on Qini curves. Decisions here are daily batch; triggering "
        "service recovery within a session, right after a late order, is the natural next step.",
    ]),
    Paragraph("How I used AI", s["h2"]),
    Paragraph("Built with an AI coding agent (Claude Code). I chose the problem and the "
              "framing with its help (churn as an action-choice problem, cross-vertical signals, "
              "time-based validation, randomised campaigns for uplift), directed the build and "
              "reviewed the code and numbers. The agent wrote the code. Changes made after "
              "checking intermediate results:", s["p"]),
    *bullets([
        "The first version valued retention over 28 days, so every action looked unprofitable. "
        "It now uses a 90-day margin, stated as an assumption.",
        "A risk model trained only on control users did no better than the baseline. Training "
        "on all arms, with the arm as a feature, fixed that.",
        "A 20,000-user simulation was too noisy to learn effects per user, so it was scaled to "
        "50,000 users and the planted effects were strengthened. That is exactly why these results "
        "validate the method, not the business case.",
        "Weak correlation checks against hidden traits were replaced with validation on the "
        "later randomised campaign. That check is also what showed the effect sizes are "
        "overstated.",
    ]),
    Paragraph("Files: churn_retention.ipynb (executed), data/generate_data.py (seeded simulator), "
              "outputs/metrics.json, narrator_prompt.md and narrator_output.md (an LLM prompt "
              "that turns the metrics into a stakeholder summary).", s["cap"]),
]

doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                        topMargin=13 * mm, bottomMargin=12 * mm,
                        title="Churn Prediction to Personalised Retention Actions - Amit Gajbhiye",
                        author="Amit Gajbhiye")
doc.build(story)
print(PDF)
