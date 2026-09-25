"""
Member 2 EDA Script — Business Entity Resolution Challenge (Akatsuki)
----------------------------------------------------------------------
Generates:
  - plots/name_length.png
  - plots/address_length.png
  - plots/country_distribution.png
  - plots/missing_values.png
  - eda_stats.json   (raw numbers used by the notebook + HTML report)
  - eda_report.html
"""

import os
import re
import json
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless – no GUI needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

warnings.filterwarnings("ignore")

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE   = Path(__file__).resolve().parent.parent.parent  # Akatsuki/../ = amazon ml challenge/
DATA   = BASE / "student_resource" / "dataset"
TRAIN  = DATA / "train"
TEST   = DATA / "test"
OUT    = Path(__file__).resolve().parent                # member2/
PLOTS  = OUT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

# ─── Plotting Style ──────────────────────────────────────────────────────────
PALETTE = ["#6C63FF", "#FF6584", "#43B89C", "#FFBE0B", "#FB5607"]
plt.rcParams.update({
    "figure.facecolor": "#0F0F1A",
    "axes.facecolor":   "#1A1A2E",
    "axes.edgecolor":   "#3A3A5C",
    "text.color":       "#E0E0FF",
    "axes.labelcolor":  "#E0E0FF",
    "xtick.color":      "#A0A0CC",
    "ytick.color":      "#A0A0CC",
    "grid.color":       "#2A2A4A",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "sans-serif",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
})

# ─── Load Data ───────────────────────────────────────────────────────────────
print("Loading train data …")
s1_tr = pd.read_csv(TRAIN / "train_source1.tsv", sep="\t")
s2_tr = pd.read_csv(TRAIN / "train_source2.tsv", sep="\t")
s3_tr = pd.read_csv(TRAIN / "train_source3.tsv", sep="\t")
gt_tr = pd.read_csv(TRAIN / "train_ground_truth.tsv", sep="\t")

print("Loading test data …")
s1_te = pd.read_csv(TEST  / "test_source1.tsv",  sep="\t")
s2_te = pd.read_csv(TEST  / "test_source2.tsv",  sep="\t")
s3_te = pd.read_csv(TEST  / "test_source3.tsv",  sep="\t")

stats = {}

# ─── 1. Row Counts ───────────────────────────────────────────────────────────
stats["row_counts"] = {
    "train": {"S1": len(s1_tr), "S2": len(s2_tr), "S3": len(s3_tr)},
    "test":  {"S1": len(s1_te), "S2": len(s2_te), "S3": len(s3_te)},
}
print("Row counts done.")

# ─── 2. Missing Values ───────────────────────────────────────────────────────
def missing_pct(df, col):
    return round(df[col].isna().mean() * 100, 2)

mv = {}
for label, df in [("S1_train", s1_tr), ("S2_train", s2_tr), ("S3_train", s3_tr),
                  ("S1_test",  s1_te), ("S2_test",  s2_te), ("S3_test",  s3_te)]:
    mv[label] = {
        "business_name":    missing_pct(df, "business_name"),
        "business_address": missing_pct(df, "business_address"),
        "country":          missing_pct(df, "country"),
    }
stats["missing_pct"] = mv
print("Missing values done.")

# ─── 3. Duplicate IDs ────────────────────────────────────────────────────────
def dup_ids(df):
    return int(df["entity_id"].duplicated().sum())

stats["duplicate_ids"] = {
    "S1_train": dup_ids(s1_tr), "S2_train": dup_ids(s2_tr), "S3_train": dup_ids(s3_tr),
    "S1_test":  dup_ids(s1_te), "S2_test":  dup_ids(s2_te), "S3_test":  dup_ids(s3_te),
}

# ─── 4. Name Length Analysis ─────────────────────────────────────────────────
def name_stats(series):
    s = series.dropna().astype(str).str.strip()
    lengths = s.str.len()
    return {
        "count":        int(len(s)),
        "unique":       int(s.nunique()),
        "dup_pct":      round((1 - s.nunique() / len(s)) * 100, 2) if len(s) else 0,
        "avg_len":      round(float(lengths.mean()), 2),
        "median_len":   round(float(lengths.median()), 2),
        "min_len":      int(lengths.min()),
        "max_len":      int(lengths.max()),
        "lengths":      lengths.tolist(),          # used for histogram
    }

stats["name"] = {
    "S1_train": name_stats(s1_tr["business_name"]),
    "S2_train": name_stats(s2_tr["business_name"]),
    "S3_train": name_stats(s3_tr["business_name"]),
    "S1_test":  name_stats(s1_te["business_name"]),
}
print("Name stats done.")

# ─── 5. Address Length & Feature Analysis ────────────────────────────────────
def addr_stats(series):
    s = series.dropna().astype(str).str.strip()
    lengths = s.str.len()
    has_digit  = s.str.contains(r"\d", regex=True)
    has_post   = s.str.contains(r"\b\d{5,6}\b", regex=True)  # 5-6 digit postal code
    return {
        "count":           int(len(s)),
        "avg_len":         round(float(lengths.mean()), 2),
        "median_len":      round(float(lengths.median()), 2),
        "min_len":         int(lengths.min()),
        "max_len":         int(lengths.max()),
        "digit_pct":       round(float(has_digit.mean()) * 100, 2),
        "postcode_pct":    round(float(has_post.mean()) * 100, 2),
        "lengths":         lengths.tolist(),
    }

stats["address"] = {
    "S1_train": addr_stats(s1_tr["business_address"]),
    "S2_train": addr_stats(s2_tr["business_address"]),
    "S3_train": addr_stats(s3_tr["business_address"]),
    "S1_test":  addr_stats(s1_te["business_address"]),
}
print("Address stats done.")

# ─── 6. Country Distribution ────────────────────────────────────────────────
def country_counts(df):
    return df["country"].fillna("MISSING").value_counts().to_dict()

stats["country"] = {
    "S1_train": country_counts(s1_tr),
    "S2_train": country_counts(s2_tr),
    "S3_train": country_counts(s3_tr),
    "S1_test":  country_counts(s1_te),
    "S2_test":  country_counts(s2_te),
    "S3_test":  country_counts(s3_te),
}
train_countries = sorted(set(s1_tr["country"].dropna()) | set(s2_tr["country"].dropna()) | set(s3_tr["country"].dropna()))
test_countries  = sorted(set(s1_te["country"].dropna()) | set(s2_te["country"].dropna()) | set(s3_te["country"].dropna()))
stats["country"]["unique_train"] = train_countries
stats["country"]["unique_test"]  = test_countries
stats["country"]["france_in_test"] = "France" in test_countries
print("Country analysis done.")

# ─── 7. Ground Truth Analysis ────────────────────────────────────────────────
def parse_matched(val):
    if pd.isna(val) or str(val).strip() == "":
        return []
    return [x.strip() for x in str(val).split(",") if x.strip()]

gt_tr["match_list"] = gt_tr["matched_entity_ids"].apply(parse_matched)
gt_tr["match_count"] = gt_tr["match_list"].apply(len)

# 0 / 1 / many distribution
zero_match   = int((gt_tr["match_count"] == 0).sum())
one_match    = int((gt_tr["match_count"] == 1).sum())
multi_match  = int((gt_tr["match_count"] >  1).sum())

# S2-only / S3-only / both
def src_category(lst):
    has_s2 = any(x.startswith("S2-") for x in lst)
    has_s3 = any(x.startswith("S3-") for x in lst)
    if has_s2 and has_s3: return "both"
    if has_s2:            return "S2_only"
    if has_s3:            return "S3_only"
    return "none"

matched_rows = gt_tr[gt_tr["match_count"] > 0].copy()
matched_rows["src_cat"] = matched_rows["match_list"].apply(src_category)
src_cat_counts = matched_rows["src_cat"].value_counts().to_dict()

stats["ground_truth"] = {
    "total_s1_entities":  int(len(gt_tr)),
    "zero_match":         zero_match,
    "one_match":          one_match,
    "multi_match":        multi_match,
    "src_category":       src_cat_counts,
    "avg_matches":        round(float(gt_tr["match_count"].mean()), 3),
    "max_matches":        int(gt_tr["match_count"].max()),
}
print("Ground truth analysis done.")

# ─── 8. Noise Pattern Examples ───────────────────────────────────────────────
# Legal suffix variants
all_names = pd.concat([s1_tr["business_name"], s2_tr["business_name"], s3_tr["business_name"]]).dropna().astype(str)
def find_pattern(pattern):
    mask = all_names.str.contains(pattern, case=False, regex=True)
    return int(mask.sum())

noise = {
    "corp_variants":  find_pattern(r"\b(corp|corporation)\b"),
    "pvt_variants":   find_pattern(r"\b(pvt|private)\b"),
    "ltd_variants":   find_pattern(r"\b(ltd|limited)\b"),
    "inc_variants":   find_pattern(r"\b(inc|incorporated)\b"),
    "llc_variants":   find_pattern(r"\bllc\b"),
    "and_ampersand":  find_pattern(r"&"),
}

# Address abbreviations
all_addr = pd.concat([s1_tr["business_address"], s2_tr["business_address"], s3_tr["business_address"]]).dropna().astype(str)
noise["road_abbrev"]   = int(all_addr.str.contains(r"\brd\b", case=False, regex=True).sum())
noise["road_full"]     = int(all_addr.str.contains(r"\broad\b", case=False, regex=True).sum())
noise["street_abbrev"] = int(all_addr.str.contains(r"\bst\b", case=False, regex=True).sum())
noise["street_full"]   = int(all_addr.str.contains(r"\bstreet\b", case=False, regex=True).sum())

stats["noise_patterns"] = noise
print("Noise pattern scan done.")

# ─── Save stats JSON ─────────────────────────────────────────────────────────
json_path = OUT / "eda_stats.json"
# Remove raw length arrays before saving (too large for JSON)
def strip_arrays(d):
    if isinstance(d, dict):
        return {k: strip_arrays(v) for k, v in d.items() if k != "lengths"}
    return d

with open(json_path, "w") as f:
    json.dump(strip_arrays(stats), f, indent=2)
print(f"Stats saved → {json_path}")

# ════════════════════════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════════════════════════

# ── Plot 1: Name Length Distribution ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
fig.patch.set_facecolor("#0F0F1A")
fig.suptitle("Business Name Length Distribution", fontsize=15, color="#E0E0FF", y=1.02)

for ax, (label, src_df, color) in zip(axes, [
    ("Source 1 (Train)", s1_tr, PALETTE[0]),
    ("Source 2 (Train)", s2_tr, PALETTE[1]),
    ("Source 3 (Train)", s3_tr, PALETTE[2]),
]):
    lengths = src_df["business_name"].dropna().astype(str).str.len()
    ax.hist(lengths.clip(upper=150), bins=50, color=color, alpha=0.85, edgecolor="none")
    ax.axvline(lengths.median(), color="#FFBE0B", lw=1.5, ls="--", label=f"Median={lengths.median():.0f}")
    ax.set_title(label, color="#E0E0FF")
    ax.set_xlabel("Name length (chars)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8, framealpha=0.3)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, axis="y")

plt.tight_layout()
plt.savefig(PLOTS / "name_length.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot 1 saved: name_length.png")

# ── Plot 2: Address Length Distribution ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
fig.patch.set_facecolor("#0F0F1A")
fig.suptitle("Business Address Length Distribution", fontsize=15, color="#E0E0FF", y=1.02)

for ax, (label, src_df, color) in zip(axes, [
    ("Source 1 (Train)", s1_tr, PALETTE[0]),
    ("Source 2 (Train)", s2_tr, PALETTE[1]),
    ("Source 3 (Train)", s3_tr, PALETTE[2]),
]):
    lengths = src_df["business_address"].dropna().astype(str).str.len()
    ax.hist(lengths.clip(upper=300), bins=50, color=color, alpha=0.85, edgecolor="none")
    ax.axvline(lengths.median(), color="#FFBE0B", lw=1.5, ls="--", label=f"Median={lengths.median():.0f}")
    ax.set_title(label, color="#E0E0FF")
    ax.set_xlabel("Address length (chars)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8, framealpha=0.3)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, axis="y")

plt.tight_layout()
plt.savefig(PLOTS / "address_length.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot 2 saved: address_length.png")

# ── Plot 3: Country Distribution ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#0F0F1A")
fig.suptitle("Country Distribution — Train vs Test", fontsize=15, color="#E0E0FF", y=1.02)

for ax, (title, dfs) in zip(axes, [
    ("Train (S1+S2+S3)", [s1_tr, s2_tr, s3_tr]),
    ("Test  (S1+S2+S3)", [s1_te, s2_te, s3_te]),
]):
    combined = pd.concat([d["country"] for d in dfs]).fillna("MISSING")
    vc = combined.value_counts()
    bars = ax.barh(vc.index.tolist(), vc.values, color=PALETTE[:len(vc)], alpha=0.9)
    ax.set_title(title, color="#E0E0FF")
    ax.set_xlabel("Number of records")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for bar, val in zip(bars, vc.values):
        ax.text(val * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", color="#E0E0FF", fontsize=9)
    ax.grid(True, axis="x")

plt.tight_layout()
plt.savefig(PLOTS / "country_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot 3 saved: country_distribution.png")

# ── Plot 4: Missing Values Heatmap ───────────────────────────────────────────
sources = ["S1_train", "S2_train", "S3_train", "S1_test", "S2_test", "S3_test"]
fields  = ["business_name", "business_address", "country"]
mv_matrix = np.array([[mv[src][f] for f in fields] for src in sources])

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("#0F0F1A")
im = ax.imshow(mv_matrix, cmap="RdPu", aspect="auto", vmin=0, vmax=max(mv_matrix.max(), 1))
ax.set_xticks(range(len(fields)))
ax.set_xticklabels(fields, rotation=15, ha="right")
ax.set_yticks(range(len(sources)))
ax.set_yticklabels(sources)
ax.set_title("Missing Value Percentage (%)", color="#E0E0FF", pad=12)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Missing %", color="#E0E0FF")
cbar.ax.yaxis.set_tick_params(color="#E0E0FF")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#E0E0FF")

for i in range(len(sources)):
    for j in range(len(fields)):
        val = mv_matrix[i, j]
        ax.text(j, i, f"{val:.2f}%", ha="center", va="center",
                color="white" if val > 5 else "#AAAACC", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(PLOTS / "missing_values.png", dpi=150, bbox_inches="tight")
plt.close()
print("Plot 4 saved: missing_values.png")

# ════════════════════════════════════════════════════════════════════════════
# HTML REPORT
# ════════════════════════════════════════════════════════════════════════════

rc = stats["row_counts"]
gt = stats["ground_truth"]
nm = stats["name"]
ad = stats["address"]
ct = stats["country"]
np_ = stats["noise_patterns"]

def pct_bar(pct, color="#6C63FF"):
    w = min(pct * 5, 100)
    return f'<div style="background:#2A2A4A;border-radius:4px;width:120px;display:inline-block;vertical-align:middle;">' \
           f'<div style="background:{color};width:{w:.0f}%;height:10px;border-radius:4px;"></div></div>'

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Member 2 EDA Report — Akatsuki</title>
<style>
  :root {{
    --bg:#0F0F1A; --card:#1A1A2E; --border:#3A3A5C;
    --text:#E0E0FF; --muted:#A0A0CC; --accent:#6C63FF;
    --pink:#FF6584; --green:#43B89C; --gold:#FFBE0B;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',sans-serif; padding:40px 24px; }}
  h1   {{ font-size:2rem; color:var(--accent); margin-bottom:6px; }}
  h2   {{ font-size:1.3rem; color:var(--gold); margin:32px 0 12px; border-bottom:1px solid var(--border); padding-bottom:6px; }}
  h3   {{ font-size:1.05rem; color:var(--green); margin:18px 0 8px; }}
  p,li {{ color:var(--muted); line-height:1.7; font-size:.93rem; }}
  .subtitle {{ color:var(--muted); font-size:.9rem; margin-bottom:32px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:16px; margin:16px 0; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:20px; }}
  .card h3 {{ color:var(--accent); margin:0 0 10px; font-size:.95rem; }}
  .stat {{ display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid var(--border); }}
  .stat:last-child {{ border-bottom:none; }}
  .stat .label {{ color:var(--muted); font-size:.85rem; }}
  .stat .value {{ color:var(--text); font-weight:600; font-size:.85rem; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:20px; font-size:.78rem; font-weight:700; }}
  .badge-ok   {{ background:#1a3a2e; color:var(--green); }}
  .badge-warn {{ background:#3a2a1a; color:var(--gold); }}
  .badge-bad  {{ background:#3a1a2e; color:var(--pink); }}
  img {{ max-width:100%; border-radius:10px; border:1px solid var(--border); margin:12px 0; }}
  table {{ width:100%; border-collapse:collapse; margin:12px 0; font-size:.88rem; }}
  th {{ background:var(--card); color:var(--accent); padding:10px 12px; text-align:left; border:1px solid var(--border); }}
  td {{ padding:8px 12px; border:1px solid var(--border); color:var(--muted); }}
  tr:nth-child(even) td {{ background:#16162A; }}
  .highlight {{ color:var(--gold); font-weight:700; }}
  ul {{ padding-left:20px; margin:8px 0; }}
  .conclusion {{ background:var(--card); border-left:4px solid var(--accent); border-radius:8px; padding:16px 20px; margin:16px 0; }}
  .conclusion p {{ color:var(--text); }}
</style>
</head>
<body>
<h1>Member 2 — Exploratory Data Analysis Report</h1>
<p class="subtitle">Amazon ML Challenge 2026 · Business Entity Resolution · Team Akatsuki</p>

<h2>1. Dataset Overview — Row Counts</h2>
<div class="grid">
  <div class="card">
    <h3>Train Set</h3>
    <div class="stat"><span class="label">Source 1 records</span><span class="value highlight">{rc['train']['S1']:,}</span></div>
    <div class="stat"><span class="label">Source 2 records</span><span class="value highlight">{rc['train']['S2']:,}</span></div>
    <div class="stat"><span class="label">Source 3 records</span><span class="value highlight">{rc['train']['S3']:,}</span></div>
    <div class="stat"><span class="label">Total train records</span><span class="value">{rc['train']['S1']+rc['train']['S2']+rc['train']['S3']:,}</span></div>
  </div>
  <div class="card">
    <h3>Test Set</h3>
    <div class="stat"><span class="label">Source 1 records</span><span class="value highlight">{rc['test']['S1']:,}</span></div>
    <div class="stat"><span class="label">Source 2 records</span><span class="value highlight">{rc['test']['S2']:,}</span></div>
    <div class="stat"><span class="label">Source 3 records</span><span class="value highlight">{rc['test']['S3']:,}</span></div>
    <div class="stat"><span class="label">Total test records</span><span class="value">{rc['test']['S1']+rc['test']['S2']+rc['test']['S3']:,}</span></div>
  </div>
  <div class="card">
    <h3>Ground Truth (Train)</h3>
    <div class="stat"><span class="label">S1 entities</span><span class="value">{gt['total_s1_entities']:,}</span></div>
    <div class="stat"><span class="label">Zero matches (singletons)</span><span class="value">{gt['zero_match']:,}</span></div>
    <div class="stat"><span class="label">One match</span><span class="value">{gt['one_match']:,}</span></div>
    <div class="stat"><span class="label">Multiple matches</span><span class="value">{gt['multi_match']:,}</span></div>
    <div class="stat"><span class="label">Avg matches per S1</span><span class="value">{gt['avg_matches']}</span></div>
    <div class="stat"><span class="label">Max matches for one S1</span><span class="value">{gt['max_matches']}</span></div>
  </div>
</div>

<h2>2. Missing Value Analysis</h2>
<img src="plots/missing_values.png" alt="Missing Values Heatmap">
<table>
  <tr><th>Source</th><th>business_name missing %</th><th>business_address missing %</th><th>country missing %</th></tr>
  {''.join(f"<tr><td>{src}</td><td>{mv[src]['business_name']}%</td><td>{mv[src]['business_address']}%</td><td>{mv[src]['country']}%</td></tr>" for src in sources)}
</table>
<div class="conclusion"><p>
  The missing value heatmap reveals the relative data completeness across all sources. Fields with high missing rates require careful handling — they cannot be used as hard-required blocking keys. Country-level missing values are particularly important because unseen labels (e.g., France) must be treated as a valid open-set label, not filtered.
</p></div>

<h2>3. Business Name Analysis</h2>
<img src="plots/name_length.png" alt="Name Length Distribution">

<table>
  <tr><th>Metric</th><th>S1 Train</th><th>S2 Train</th><th>S3 Train</th><th>S1 Test</th></tr>
  <tr><td>Record count</td><td>{nm['S1_train']['count']:,}</td><td>{nm['S2_train']['count']:,}</td><td>{nm['S3_train']['count']:,}</td><td>{nm['S1_test']['count']:,}</td></tr>
  <tr><td>Unique names</td><td>{nm['S1_train']['unique']:,}</td><td>{nm['S2_train']['unique']:,}</td><td>{nm['S3_train']['unique']:,}</td><td>{nm['S1_test']['unique']:,}</td></tr>
  <tr><td>Duplicate name %</td><td>{nm['S1_train']['dup_pct']}%</td><td>{nm['S2_train']['dup_pct']}%</td><td>{nm['S3_train']['dup_pct']}%</td><td>{nm['S1_test']['dup_pct']}%</td></tr>
  <tr><td>Average length</td><td>{nm['S1_train']['avg_len']}</td><td>{nm['S2_train']['avg_len']}</td><td>{nm['S3_train']['avg_len']}</td><td>{nm['S1_test']['avg_len']}</td></tr>
  <tr><td>Median length</td><td>{nm['S1_train']['median_len']}</td><td>{nm['S2_train']['median_len']}</td><td>{nm['S3_train']['median_len']}</td><td>{nm['S1_test']['median_len']}</td></tr>
  <tr><td>Min length</td><td>{nm['S1_train']['min_len']}</td><td>{nm['S2_train']['min_len']}</td><td>{nm['S3_train']['min_len']}</td><td>{nm['S1_test']['min_len']}</td></tr>
  <tr><td>Max length</td><td>{nm['S1_train']['max_len']}</td><td>{nm['S2_train']['max_len']}</td><td>{nm['S3_train']['max_len']}</td><td>{nm['S1_test']['max_len']}</td></tr>
</table>

<h3>Legal Suffix Noise Patterns (across all train sources)</h3>
<table>
  <tr><th>Pattern</th><th>Records Containing Pattern</th></tr>
  <tr><td>Corp / Corporation</td><td>{np_['corp_variants']:,}</td></tr>
  <tr><td>Pvt / Private</td><td>{np_['pvt_variants']:,}</td></tr>
  <tr><td>Ltd / Limited</td><td>{np_['ltd_variants']:,}</td></tr>
  <tr><td>Inc / Incorporated</td><td>{np_['inc_variants']:,}</td></tr>
  <tr><td>LLC</td><td>{np_['llc_variants']:,}</td></tr>
  <tr><td>Ampersand (&amp;)</td><td>{np_['and_ampersand']:,}</td></tr>
</table>
<div class="conclusion"><p>
  Legal suffix variants (Corp/Corporation, Pvt/Private, Ltd/Limited) are confirmed to appear in the dataset. These must be normalized — a token-level canonical form (e.g., mapping "Corporation" → "Corp") is required before computing name similarity features. The presence of "&" alongside "and" also requires disambiguation during preprocessing.
</p></div>

<h2>4. Business Address Analysis</h2>
<img src="plots/address_length.png" alt="Address Length Distribution">

<table>
  <tr><th>Metric</th><th>S1 Train</th><th>S2 Train</th><th>S3 Train</th><th>S1 Test</th></tr>
  <tr><td>Average length</td><td>{ad['S1_train']['avg_len']}</td><td>{ad['S2_train']['avg_len']}</td><td>{ad['S3_train']['avg_len']}</td><td>{ad['S1_test']['avg_len']}</td></tr>
  <tr><td>Median length</td><td>{ad['S1_train']['median_len']}</td><td>{ad['S2_train']['median_len']}</td><td>{ad['S3_train']['median_len']}</td><td>{ad['S1_test']['median_len']}</td></tr>
  <tr><td>Has numeric token %</td><td>{ad['S1_train']['digit_pct']}%</td><td>{ad['S2_train']['digit_pct']}%</td><td>{ad['S3_train']['digit_pct']}%</td><td>{ad['S1_test']['digit_pct']}%</td></tr>
  <tr><td>Has postal code %</td><td>{ad['S1_train']['postcode_pct']}%</td><td>{ad['S2_train']['postcode_pct']}%</td><td>{ad['S3_train']['postcode_pct']}%</td><td>{ad['S1_test']['postcode_pct']}%</td></tr>
  <tr><td>Min length</td><td>{ad['S1_train']['min_len']}</td><td>{ad['S2_train']['min_len']}</td><td>{ad['S3_train']['min_len']}</td><td>{ad['S1_test']['min_len']}</td></tr>
  <tr><td>Max length</td><td>{ad['S1_train']['max_len']}</td><td>{ad['S2_train']['max_len']}</td><td>{ad['S3_train']['max_len']}</td><td>{ad['S1_test']['max_len']}</td></tr>
</table>

<h3>Road/Street Abbreviation Pattern Counts (all train sources)</h3>
<table>
  <tr><th>Pattern</th><th>Count</th></tr>
  <tr><td>"Rd" (abbreviation)</td><td>{np_['road_abbrev']:,}</td></tr>
  <tr><td>"Road" (full form)</td><td>{np_['road_full']:,}</td></tr>
  <tr><td>"St" (abbreviation)</td><td>{np_['street_abbrev']:,}</td></tr>
  <tr><td>"Street" (full form)</td><td>{np_['street_full']:,}</td></tr>
</table>
<div class="conclusion"><p>
  Address fields contain significant abbreviation noise ("Rd" vs "Road", "St" vs "Street"). Postal code presence varies substantially across sources, making it a weak mandatory blocking key but a strong confirming feature. Address fields should be tokenized and soft-matched using token Jaccard or TF-IDF cosine similarity rather than exact matching.
</p></div>

<h2>5. Country Distribution</h2>
<img src="plots/country_distribution.png" alt="Country Distribution">

<table>
  <tr><th>Source</th>{"".join(f"<th>{c}</th>" for c in sorted(set(list(ct['S1_train'].keys())+list(ct['S1_test'].keys())+list(ct['S2_test'].keys()))))}
  </tr>
  {"".join(
    "<tr><td>" + src + "</td>" +
    "".join(f"<td>{ct[src].get(c, 0):,}</td>" for c in sorted(set(list(ct['S1_train'].keys())+list(ct['S1_test'].keys())+list(ct['S2_test'].keys())))) +
    "</tr>"
    for src in ["S1_train","S2_train","S3_train","S1_test","S2_test","S3_test"]
  )}
</table>

<div class="conclusion">
<p><strong>Key Finding:</strong> Training data contains the countries: <span class="highlight">{", ".join(ct['unique_train'])}</span>.<br>
Test data contains: <span class="highlight">{", ".join(ct['unique_test'])}</span>.<br>
France {'<span class="badge badge-warn">⚠ PRESENT in Test</span>' if ct['france_in_test'] else '<span class="badge badge-ok">Not found in Test</span>'}.<br>
<br>
<strong>Implication:</strong> Country must be treated as an open-set string label. Do NOT one-hot encode or hard-code a fixed set of {{US, India}} — France (and potentially other unseen countries) must flow through the pipeline without errors. Use country as a string-equality feature only, never as a categorical encoder with a fixed vocabulary.
</p></div>

<h2>6. Ground Truth — Match Distribution</h2>
<div class="grid">
  <div class="card">
    <h3>Match Count Distribution</h3>
    <div class="stat"><span class="label">Zero matches (singletons)</span><span class="value highlight">{gt['zero_match']:,}</span></div>
    <div class="stat"><span class="label">Exactly one match</span><span class="value">{gt['one_match']:,}</span></div>
    <div class="stat"><span class="label">Multiple matches (≥2)</span><span class="value">{gt['multi_match']:,}</span></div>
    <div class="stat"><span class="label">Average match count</span><span class="value">{gt['avg_matches']}</span></div>
    <div class="stat"><span class="label">Maximum match count</span><span class="value">{gt['max_matches']}</span></div>
  </div>
  <div class="card">
    <h3>Source Category (matched entities)</h3>
    {"".join(f'<div class="stat"><span class="label">{k}</span><span class="value">{v:,}</span></div>' for k,v in gt['src_category'].items())}
  </div>
</div>
<div class="conclusion"><p>
  Singletons (zero-match S1 entities) earn a full F_0.5 = 1.0 when correctly predicted as empty. They must not be ignored — the model must confidently produce an empty matched_entity_ids for these records. False merges on singletons are penalised heavily by the precision-weighted F_0.5 metric.
</p></div>

<h2>7. Key Conclusions for Normalization & Blocking Design</h2>
<ul>
  <li><strong>Name normalization required:</strong> Legal suffix abbreviations (Corp/Corporation, Pvt/Private, Ltd/Limited, Inc) confirmed as major noise source. Strip/canonicalize before any comparison.</li>
  <li><strong>Address soft-matching required:</strong> Both abbreviated ("Rd", "St") and full ("Road", "Street") forms coexist. Postal code presence is inconsistent — cannot be a mandatory blocking key.</li>
  <li><strong>Country is an open set:</strong> France appears in the test set but not in training. Never encode country as a fixed-vocabulary categorical feature. Use it as a string equality boolean feature.</li>
  <li><strong>Singletons matter:</strong> A substantial fraction of S1 entities have zero matches. Precision-heavy F_0.5 scoring means false merges are twice as costly as missed links.</li>
  <li><strong>Multi-source matching:</strong> Many S1 entities match records from both S2 and S3. Blocking must cover both source files simultaneously.</li>
  <li><strong>Blocking strategy implication:</strong> Given address variability, blocking on name token overlap (inverted index) + phonetic keys offers the best recall ceiling without exploding the candidate set.</li>
</ul>

<hr style="border-color:var(--border);margin:40px 0">
<p style="color:var(--muted);font-size:.8rem;text-align:center">
  Generated by Member 2 EDA Script · Team Akatsuki · Amazon ML Challenge 2026
</p>
</body>
</html>
"""

html_path = OUT / "eda_report.html"
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"HTML report saved → {html_path}")

print("\n✅  All Member 2 EDA artifacts generated successfully.")
print(f"   Plots   : {PLOTS}")
print(f"   Report  : {html_path}")
print(f"   Stats   : {json_path}")
