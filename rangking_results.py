import csv
import glob
import os
import re
from datetime import datetime

# Configuration
OUTPUT_DIR = "output"

SUMMARY_PATTERN = os.path.join(OUTPUT_DIR, "evaluation_summary_*.txt")

WEIGHTS = {
    "EX_NO_FALLBACK": 0.40,
    "ESM": 0.25,
    "VX": 0.20,
    "Latency": 0.15,
    "Fallback": 0.00,
}

DIFFICULTY_BUCKETS = [
    ("easy", 1, 19),
    ("medium", 20, 30),
    ("hard", 31, 41),
    ("extra_hard", 42, 45),
]


def init_difficulty_stats():
    return {
        name: {
            "total": 0,
            "esm_count": 0,
            "ex_count": 0,
            "vx_count": 0,
            "esm_rate": 0.0,
            "ex_rate": 0.0,
            "vx_rate": 0.0,
        }
        for name, _, _ in DIFFICULTY_BUCKETS
    }


def extract_run_timestamp(filename):
    """Extract timestamp YYYYMMDD_HHMMSS dari filename summary/detail."""
    m = re.search(r"_(\d{8}_\d{6})\.txt$", filename)
    if m:
        return m.group(1)
    m = re.search(r"_(\d{8}_\d{6})\.csv$", filename)
    if m:
        return m.group(1)
    return ""


def parse_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def question_id_to_bucket(question_id):
    for name, start, end in DIFFICULTY_BUCKETS:
        if start <= question_id <= end:
            return name
    return None


def parse_detail_file(filepath):
    """Parse evaluation_detail_*.csv untuk menghitung rate per difficulty."""
    stats = init_difficulty_stats()

    if not os.path.exists(filepath):
        return stats, False

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid_raw = str(row.get("question_id", "")).strip()
                if not qid_raw.isdigit():
                    continue

                bucket = question_id_to_bucket(int(qid_raw))
                if bucket is None:
                    continue

                s = stats[bucket]
                s["total"] += 1

                if parse_bool(row.get("esm_match", "")):
                    s["esm_count"] += 1
                if parse_bool(row.get("exec_match", "")):
                    s["ex_count"] += 1
                if parse_bool(row.get("pred_syntax_valid", "")):
                    s["vx_count"] += 1

        for name, _, _ in DIFFICULTY_BUCKETS:
            s = stats[name]
            if s["total"] > 0:
                s["esm_rate"] = s["esm_count"] / s["total"] * 100
                s["ex_rate"] = s["ex_count"] / s["total"] * 100
                s["vx_rate"] = s["vx_count"] / s["total"] * 100

        return stats, True

    except Exception as e:
        print(f" [WARNING] Gagal parse detail file {os.path.basename(filepath)}: {e}")
        return stats, False


def parse_summary_datetime(date_str, summary_filename):
    """Parse datetime dari field Date, fallback ke timestamp filename."""
    if date_str:
        try:
            return datetime.fromisoformat(date_str)
        except Exception:
            pass

    m = re.search(r"evaluation_summary_(\d{8}_\d{6})\.txt", summary_filename)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
        except Exception:
            pass

    return datetime.min


def parse_summary_file(filepath):
    """
    Parse evaluation_summary_*.txt dan extract metrics.
    Return dict dengan model info dan metrics.
    """
    data = {
        "file": os.path.basename(filepath),
        "predictions_file": "",
        "model_name": "",
        "date": "",
        "parsed_date": datetime.min,
        "gold_file": "",
        "database": "",
        "total": 0,
        "ESM": 0.0,
        "ESM_count": 0,
        "EX": 0.0,
        "EX_count": 0,
        "EX_no_fallback": 0.0,
        "EX_no_fallback_count": 0,
        "EX_no_fallback_total": 0,
        "VX": 0.0,
        "VX_count": 0,
        "VX_no_fallback": 0.0,
        "VX_no_fallback_count": 0,
        "VX_no_fallback_total": 0,
        "valid_rate": 0.0,
        "select1_fallback": 0,
        "select1_rate": 0.0,
        "both_error": 0,
        "pred_error": 0,
        "gold_error": 0,
        "pred_error_rate": 0.0,
        "avg_latency": 0.0,
        "median_latency": 0.0,
        "p95_latency": 0.0,
        "difficulty_detail_file": "",
        "difficulty_available": False,
        "difficulty_stats": init_difficulty_stats(),
    }

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        pred_match = re.search(r"Predictions:\s*(.+)", content)
        if pred_match:
            data["predictions_file"] = pred_match.group(1).strip()
            fname = os.path.basename(data["predictions_file"])
            model_match = re.search(
                r"prediksi_imron_topk\d+_(.+?)_\d{8}_\d{6}\.txt", fname
            )
            if model_match:
                data["model_name"] = model_match.group(1).replace("_", " ")
            else:
                data["model_name"] = fname

        # Gold file
        gold_match = re.search(r"Gold:\s*(.+)", content)
        if gold_match:
            data["gold_file"] = gold_match.group(1).strip()

        # Database
        db_match = re.search(r"Database:\s*(.+)", content)
        if db_match:
            data["database"] = db_match.group(1).strip()

        # Date
        date_match = re.search(r"Date:\s*(.+)", content)
        if date_match:
            data["date"] = date_match.group(1).strip()

        data["parsed_date"] = parse_summary_datetime(data["date"], data["file"])

        run_ts = extract_run_timestamp(data["file"])
        if run_ts:
            detail_filename = f"evaluation_detail_{run_ts}.csv"
            detail_path = os.path.join(OUTPUT_DIR, detail_filename)
            data["difficulty_detail_file"] = detail_filename
            diff_stats, diff_ok = parse_detail_file(detail_path)
            data["difficulty_stats"] = diff_stats
            data["difficulty_available"] = diff_ok

        # Total questions
        total_match = re.search(r"Total Questions:\s*(\d+)", content)
        if total_match:
            data["total"] = int(total_match.group(1))

        # Exact Set Match (ESM): 15/45 = 33.3%
        esm_match = re.search(
            r"Exact Set Match \(ESM\):\s*(\d+)/(\d+)\s*=\s*([\d.]+)%", content
        )
        if esm_match:
            data["ESM_count"] = int(esm_match.group(1))
            data["ESM"] = float(esm_match.group(3))

        # Execution Accuracy (EX): 20/45 = 44.4%
        ex_match = re.search(
            r"Execution Accuracy \(EX\):\s*(\d+)/(\d+)\s*=\s*([\d.]+)%", content
        )
        if ex_match:
            data["EX_count"] = int(ex_match.group(1))
            data["EX"] = float(ex_match.group(3))

        # EX (excl. SELECT 1): 18/40 = 45.0%
        ex_nf_match = re.search(
            r"EX \(excl\. SELECT 1\):\s*(\d+)/(\d+)\s*=\s*([\d.]+)%", content
        )
        if ex_nf_match:
            data["EX_no_fallback_count"] = int(ex_nf_match.group(1))
            data["EX_no_fallback_total"] = int(ex_nf_match.group(2))
            data["EX_no_fallback"] = float(ex_nf_match.group(3))

        # Syntax Validity (VX): 40/45 = 88.9%
        vx_match = re.search(
            r"Syntax Validity \(VX\):\s*(\d+)/(\d+)\s*=\s*([\d.]+)%", content
        )
        if vx_match:
            data["VX_count"] = int(vx_match.group(1))
            data["VX"] = float(vx_match.group(3))

        # Execution Validity: 38/45 = 84.4%
        ev_match = re.search(
            r"Execution Validity:\s*(\d+)/(\d+)\s*=\s*([\d.]+)%", content
        )
        if ev_match:
            data["valid_rate"] = float(ev_match.group(3))

        # SELECT 1 Fallbacks: 5/45 = 11.1%
        s1_match = re.search(
            r"SELECT 1 Fallbacks:\s*(\d+)/(\d+)\s*=\s*([\d.]+)%", content
        )
        if s1_match:
            data["select1_fallback"] = int(s1_match.group(1))
            data["select1_rate"] = float(s1_match.group(3))

        # Latency: Avg: 2.345ms
        avg_lat_match = re.search(r"Avg:\s*([\d.]+)ms", content)
        if avg_lat_match:
            data["avg_latency"] = float(avg_lat_match.group(1))
        else:
            avg_lat_s = re.search(r"Avg:\s*([\d.]+)s", content)
            if avg_lat_s:
                data["avg_latency"] = float(avg_lat_s.group(1)) * 1000

        # Median
        med_match = re.search(r"Median:\s*([\d.]+)ms", content)
        if med_match:
            data["median_latency"] = float(med_match.group(1))
        else:
            med_s = re.search(r"Median:\s*([\d.]+)s", content)
            if med_s:
                data["median_latency"] = float(med_s.group(1)) * 1000

        # P95
        p95_match = re.search(r"P95:\s*([\d.]+)ms", content)
        if p95_match:
            data["p95_latency"] = float(p95_match.group(1))
        else:
            p95_s = re.search(r"P95:\s*([\d.]+)s", content)
            if p95_s:
                data["p95_latency"] = float(p95_s.group(1)) * 1000

        # Error analysis
        be_match = re.search(r"Both Error:\s*(\d+)", content)
        if be_match:
            data["both_error"] = int(be_match.group(1))

        pe_match = re.search(r"Pred Error Only:\s*(\d+)", content)
        if pe_match:
            data["pred_error"] = int(pe_match.group(1))

        ge_match = re.search(r"Gold Error Only:\s*(\d+)", content)
        if ge_match:
            data["gold_error"] = int(ge_match.group(1))

        # Derived rates
        if data["total"] > 0:
            data["pred_error_rate"] = data["pred_error"] / data["total"] * 100

        # VX tanpa fallback SELECT 1
        non_fallback_total = data["total"] - data["select1_fallback"]
        data["VX_no_fallback_total"] = max(0, non_fallback_total)
        if non_fallback_total > 0:
            vx_nf_count = max(0, data["VX_count"] - data["select1_fallback"])
            data["VX_no_fallback_count"] = vx_nf_count
            data["VX_no_fallback"] = vx_nf_count / non_fallback_total * 100
        else:
            data["VX_no_fallback"] = 0.0

    except Exception as e:
        print(f" Error parsing {filepath}: {e}")

    return data


def deduplicate_latest_run_per_model(models):
    """Ambil run terbaru per model_name."""
    latest = {}
    for m in models:
        key = m["model_name"] or m["file"]
        prev = latest.get(key)
        if prev is None or m["parsed_date"] > prev["parsed_date"]:
            latest[key] = m
    return list(latest.values())


def print_consistency_warnings(models):
    """Peringatan jika eksperimen tidak konsisten antar summary."""
    golds = sorted({m["gold_file"] for m in models if m["gold_file"]})
    dbs = sorted({m["database"] for m in models if m["database"]})
    totals = sorted({m["total"] for m in models if m["total"] > 0})

    warning = False
    if len(golds) > 1:
        warning = True
        print(f"\n[WARNING] Gold file tidak konsisten: {golds}")
    if len(dbs) > 1:
        warning = True
        print(f"[WARNING] Database tidak konsisten: {dbs}")
    if len(totals) > 1:
        warning = True
        print(f"[WARNING] Total questions tidak konsisten: {totals}")

    if warning:
        print(
            "[WARNING] Ranking tetap dilanjutkan, tapi hasil bisa tidak apple-to-apple.\n"
        )


def calculate_weighted_score(model_data, all_models_data):
    """
    Hitung weighted score untuk ranking.

    - ESM, EX_no_fallback, VX: semakin tinggi semakin bagus
    - Latency: semakin rendah semakin bagus (dinormalisasi terbalik)
    - Fallback: semakin kecil select1_rate semakin bagus
    """
    esm = model_data["ESM"]
    vx_for_score = model_data.get("VX_no_fallback", model_data["VX"])

    # EX metric untuk scoring: EX_no_fallback (jika denominator > 0), else 0
    # (artinya jika semua query fallback, model tidak mendapat poin EX utama)
    if model_data["EX_no_fallback_total"] > 0:
        ex_for_score = model_data["EX_no_fallback"]
    else:
        ex_for_score = 0.0

    # Fallback score 0..100 (100 = tanpa fallback)
    fallback_score = max(0.0, 100.0 - model_data["select1_rate"])

    # Latency score 0..100 (100 = tercepat)
    all_latencies = [m["avg_latency"] for m in all_models_data if m["avg_latency"] > 0]
    if all_latencies and model_data["avg_latency"] > 0:
        max_lat = max(all_latencies)
        min_lat = min(all_latencies)
        if max_lat > min_lat:
            latency_score = (
                1 - (model_data["avg_latency"] - min_lat) / (max_lat - min_lat)
            ) * 100
        else:
            latency_score = 100.0
    else:
        latency_score = 0.0

    weighted = (
        WEIGHTS["EX_NO_FALLBACK"] * ex_for_score
        + WEIGHTS["ESM"] * esm
        + WEIGHTS["VX"] * vx_for_score
        + WEIGHTS["Latency"] * latency_score
        + WEIGHTS["Fallback"] * fallback_score
    )

    return weighted, latency_score, ex_for_score, fallback_score


def format_latency_display(ms):
    """Format latency untuk display"""
    if ms <= 0:
        return "N/A"
    if ms < 1:
        return f"{ms:.3f}ms"
    if ms < 1000:
        return f"{ms:.1f}ms"
    return f"{ms / 1000:.2f}s"


def format_rate_display(rate, total):
    if total <= 0:
        return "N/A"
    return f"{rate:.1f}%"


def main():
    print(f"\n{'=' * 120}")
    print("MODEL RANKING: Perbandingan Hasil Evaluasi")
    print(f"{'=' * 120}")
    print(f"Summary Directory: {OUTPUT_DIR}")
    print(
        "Weights: "
        f"EX(-S1)={WEIGHTS['EX_NO_FALLBACK']:.0%}, "
        f"ESM={WEIGHTS['ESM']:.0%}, "
        f"VX(-S1)={WEIGHTS['VX']:.0%}, "
        f"Latency={WEIGHTS['Latency']:.0%}"
    )
    print(f"{'=' * 120}\n")

    # Validate weight sum
    weight_sum = sum(WEIGHTS.values())
    if abs(weight_sum - 1.0) > 1e-9:
        print(f" [WARNING] Total weight = {weight_sum:.4f} (seharusnya 1.0)")

    # Find all summary files
    summary_files = glob.glob(SUMMARY_PATTERN)

    if not summary_files:
        print(
            f"   Tidak ditemukan file evaluation_summary_*.txt di folder {OUTPUT_DIR}/"
        )
        print("   Jalankan evaluation.py terlebih dahulu untuk setiap model.")
        return

    print(f"Found {len(summary_files)} evaluation summaries:\n")

    # Parse all summaries
    all_runs = []
    for sf in sorted(summary_files):
        print(f"  📄 Parsing: {os.path.basename(sf)}")
        data = parse_summary_file(sf)
        if data["total"] > 0:
            all_runs.append(data)
            print(
                "     -> "
                f"Model: {data['model_name']} | "
                f"ESM={data['ESM']:.1f}% | "
                f"EX={data['EX']:.1f}% | "
                f"EX(-S1)={data['EX_no_fallback']:.1f}% | "
                f"VX(-S1)={data['VX_no_fallback']:.1f}%"
            )
        else:
            print("     Skipped (no data parsed)")

    if not all_runs:
        print("\n Tidak ada data model yang valid untuk diranking.")
        return

    # Deduplicate by model: keep latest run
    all_models = deduplicate_latest_run_per_model(all_runs)
    removed = len(all_runs) - len(all_models)
    if removed > 0:
        print(
            f"\n[INFO] Dedup run per model: {len(all_runs)} -> {len(all_models)} "
            f"(menghapus {removed} run lama)."
        )

    # Warn for inconsistent experiment setup
    print_consistency_warnings(all_models)

    # Calculate weighted scores
    for model in all_models:
        weighted, lat_score, ex_for_score, fallback_score = calculate_weighted_score(
            model, all_models
        )
        model["weighted_score"] = weighted
        model["latency_score"] = lat_score
        model["ex_for_score"] = ex_for_score
        model["fallback_score"] = fallback_score

    # Sort by weighted score (highest first)
    ranked = sorted(all_models, key=lambda x: x["weighted_score"], reverse=True)

    # ==================== RANKING TABLE ====================
    print(f"\n{'=' * 138}")
    print(f"{'RANKING':^138}")
    print(f"{'=' * 138}")

    header = (
        f"{'Rank':>4} | "
        f"{'Model':<25} | "
        f"{'ESM':>8} | "
        f"{'EX':>8} | "
        f"{'EX(-S1)':>8} | "
        f"{'VX(-S1)':>8} | "
        f"{'Avg Lat':>10} | "
        f"{'FB%':>7} | "
        f"{'Score':>8}"
    )
    print(header)
    print(f"{'-' * 138}")

    for rank, m in enumerate(ranked, 1):
        if rank == 1:
            medal = "🥇"
        elif rank == 2:
            medal = "🥈"
        elif rank == 3:
            medal = "🥉"
        else:
            medal = "  "

        row = (
            f"{medal}{rank:>2} | "
            f"{m['model_name']:<25} | "
            f"{m['ESM']:>7.1f}% | "
            f"{m['EX']:>7.1f}% | "
            f"{m['EX_no_fallback']:>7.1f}% | "
            f"{m['VX_no_fallback']:>7.1f}% | "
            f"{format_latency_display(m['avg_latency']):>10} | "
            f"{m['select1_rate']:>6.1f}% | "
            f"{m['weighted_score']:>7.1f}"
        )
        print(row)

    print(f"{'=' * 138}")

    # ==================== DETAILED COMPARISON ====================
    print(f"\n{'=' * 100}")
    print("DETAILED COMPARISON")
    print(f"{'=' * 100}")

    best_esm = max(ranked, key=lambda x: x["ESM"])
    best_ex = max(ranked, key=lambda x: x["EX_no_fallback"])
    best_vx = max(ranked, key=lambda x: x["VX_no_fallback"])
    best_lat = min(
        ranked,
        key=lambda x: x["avg_latency"] if x["avg_latency"] > 0 else float("inf"),
    )
    least_fallback = min(
        ranked, key=lambda x: (x["select1_rate"], x["select1_fallback"])
    )
    least_errors = min(ranked, key=lambda x: (x["pred_error_rate"], x["pred_error"]))

    print("\n🏆 Best per Category:")
    print(
        f"  Best Exact Set Match (ESM): {best_esm['model_name']} ({best_esm['ESM']:.1f}%)"
    )
    print(
        f"  Best Execution Acc EX(-S1): {best_ex['model_name']} ({best_ex['EX_no_fallback']:.1f}%)"
    )
    print(
        f"  Best Syntax Validity (VX(-S1)):  {best_vx['model_name']} ({best_vx['VX_no_fallback']:.1f}%)"
    )
    print(
        f"  Fastest (Avg Latency):      {best_lat['model_name']} "
        f"({format_latency_display(best_lat['avg_latency'])})"
    )
    print(
        f"  Least Fallbacks:            {least_fallback['model_name']} "
        f"({least_fallback['select1_rate']:.1f}%)"
    )
    print(
        f"  Least Pred Errors:          {least_errors['model_name']} "
        f"({least_errors['pred_error_rate']:.1f}% / {least_errors['pred_error']} errors)"
    )

    # ==================== ERROR ANALYSIS ====================
    print(f"\n{'=' * 108}")
    print("ERROR ANALYSIS")
    print(f"{'=' * 108}")

    err_header = (
        f"{'Model':<25} | "
        f"{'Pred Err':>8} | "
        f"{'Pred Err%':>9} | "
        f"{'Gold Err':>8} | "
        f"{'Both Err':>8} | "
        f"{'SELECT 1':>8} | "
        f"{'Valid SQL':>8}"
    )
    print(err_header)
    print(f"{'-' * 108}")

    for m in ranked:
        row = (
            f"{m['model_name']:<25} | "
            f"{m['pred_error']:>8} | "
            f"{m['pred_error_rate']:>8.1f}% | "
            f"{m['gold_error']:>8} | "
            f"{m['both_error']:>8} | "
            f"{m['select1_fallback']:>8} | "
            f"{m['valid_rate']:>7.1f}%"
        )
        print(row)

    # ==================== DIFFICULTY BREAKDOWN ====================
    print(f"\n{'=' * 108}")
    print("DIFFICULTY BREAKDOWN (Execution Accuracy / EX)")
    print(f"{'=' * 108}")

    has_any_difficulty_data = any(m.get("difficulty_available") for m in ranked)

    diff_header = (
        f"{'Model':<25} | "
        f"{'Easy':>10} | "
        f"{'Medium':>10} | "
        f"{'Hard':>10} | "
        f"{'XHard':>10}"
    )
    print(diff_header)
    print(f"{'-' * 108}")

    for m in ranked:
        s_easy = m["difficulty_stats"]["easy"]
        s_med = m["difficulty_stats"]["medium"]
        s_hard = m["difficulty_stats"]["hard"]
        s_xhard = m["difficulty_stats"]["extra_hard"]

        row = (
            f"{m['model_name']:<25} | "
            f"{format_rate_display(s_easy['ex_rate'], s_easy['total']):>10} | "
            f"{format_rate_display(s_med['ex_rate'], s_med['total']):>10} | "
            f"{format_rate_display(s_hard['ex_rate'], s_hard['total']):>10} | "
            f"{format_rate_display(s_xhard['ex_rate'], s_xhard['total']):>10}"
        )
        print(row)

    if not has_any_difficulty_data:
        print("\n[INFO] File evaluation_detail_*.csv tidak ditemukan atau tidak terbaca.")
        print("       Rate per difficulty ditampilkan sebagai N/A.")

    def best_model_by_difficulty(bucket_name):
        candidates = [
            m for m in ranked if m["difficulty_stats"][bucket_name]["total"] > 0
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda x: (
                x["difficulty_stats"][bucket_name]["ex_rate"],
                x["difficulty_stats"][bucket_name]["ex_count"],
            ),
        )

    best_easy = best_model_by_difficulty("easy")
    best_medium = best_model_by_difficulty("medium")
    best_hard = best_model_by_difficulty("hard")
    best_xhard = best_model_by_difficulty("extra_hard")

    print("\n🏆 Best EX per Difficulty:")
    if best_easy:
        print(
            f"  Easy:       {best_easy['model_name']} "
            f"({best_easy['difficulty_stats']['easy']['ex_rate']:.1f}%)"
        )
    if best_medium:
        print(
            f"  Medium:     {best_medium['model_name']} "
            f"({best_medium['difficulty_stats']['medium']['ex_rate']:.1f}%)"
        )
    if best_hard:
        print(
            f"  Hard:       {best_hard['model_name']} "
            f"({best_hard['difficulty_stats']['hard']['ex_rate']:.1f}%)"
        )
    if best_xhard:
        print(
            f"  Extra Hard: {best_xhard['model_name']} "
            f"({best_xhard['difficulty_stats']['extra_hard']['ex_rate']:.1f}%)"
        )

    # ==================== WINNER ====================
    winner = ranked[0]
    print(f"\n{'=' * 100}")
    print(f"{'🏆 OVERALL WINNER 🏆':^100}")
    print(f"{'=' * 100}")
    print(f"  Model:              {winner['model_name']}")
    print(f"  Weighted Score:     {winner['weighted_score']:.1f}")
    print(
        f"  Exact Set Match:    {winner['ESM']:.1f}% ({winner['ESM_count']}/{winner['total']})"
    )
    print(
        f"  Execution Acc:      {winner['EX']:.1f}% ({winner['EX_count']}/{winner['total']})"
    )
    print(
        f"  EX (no fallback):   {winner['EX_no_fallback']:.1f}% "
        f"({winner['EX_no_fallback_count']}/{winner['EX_no_fallback_total']})"
    )
    print(
        f"  Syntax Validity:    {winner['VX_no_fallback']:.1f}% ({winner['VX_no_fallback_count']}/{winner['VX_no_fallback_total']})"
    )
    print(f"  Avg Latency:        {format_latency_display(winner['avg_latency'])}")
    print(
        f"  Fallbacks:          {winner['select1_fallback']}/{winner['total']} ({winner['select1_rate']:.1f}%)"
    )
    print(f"{'=' * 100}")

    # ==================== SAVE REPORT ====================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(OUTPUT_DIR, f"ranking_report_{timestamp}.txt")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("MODEL RANKING REPORT\n")
        f.write(f"{'=' * 80}\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Models Compared: {len(ranked)}\n")
        f.write(
            "Weights: "
            f"EX(-S1)={WEIGHTS['EX_NO_FALLBACK']:.0%}, "
            f"ESM={WEIGHTS['ESM']:.0%}, "
            f"VX(-S1)={WEIGHTS['VX']:.0%}, "
            f"Latency={WEIGHTS['Latency']:.0%}\n"
        )
        f.write(f"{'=' * 80}\n\n")

        f.write("RANKING\n")
        f.write(f"{'-' * 98}\n")
        f.write(
            f"{'Rank':>4} | {'Model':<25} | {'ESM':>7} | {'EX':>7} | {'EX(-S1)':>8} | {'VX(-S1)':>7} | {'FB%':>7} | {'Avg Lat':>10} | {'Score':>7}\n"
        )
        f.write(f"{'-' * 98}\n")

        for rank, m in enumerate(ranked, 1):
            f.write(
                f"{rank:>4} | "
                f"{m['model_name']:<25} | "
                f"{m['ESM']:>6.1f}% | "
                f"{m['EX']:>6.1f}% | "
                f"{m['EX_no_fallback']:>7.1f}% | "
                f"{m['VX_no_fallback']:>6.1f}% | "
                f"{m['select1_rate']:>6.1f}% | "
                f"{format_latency_display(m['avg_latency']):>10} | "
                f"{m['weighted_score']:>6.1f}\n"
            )

        f.write("\nDIFFICULTY BREAKDOWN (Execution Accuracy / EX)\n")
        f.write(f"{'-' * 80}\n")
        f.write(
            f"{'Model':<25} | {'Easy':>10} | {'Medium':>10} | {'Hard':>10} | {'XHard':>10}\n"
        )
        f.write(f"{'-' * 80}\n")
        for m in ranked:
            s_easy = m["difficulty_stats"]["easy"]
            s_med = m["difficulty_stats"]["medium"]
            s_hard = m["difficulty_stats"]["hard"]
            s_xhard = m["difficulty_stats"]["extra_hard"]
            f.write(
                f"{m['model_name']:<25} | "
                f"{format_rate_display(s_easy['ex_rate'], s_easy['total']):>10} | "
                f"{format_rate_display(s_med['ex_rate'], s_med['total']):>10} | "
                f"{format_rate_display(s_hard['ex_rate'], s_hard['total']):>10} | "
                f"{format_rate_display(s_xhard['ex_rate'], s_xhard['total']):>10}\n"
            )

        if not has_any_difficulty_data:
            f.write("\n[INFO] evaluation_detail_*.csv tidak ditemukan/tidak terbaca; rate difficulty N/A.\n")

        f.write(f"\n{'=' * 80}\n")
        f.write(
            f"WINNER: {winner['model_name']} (Score: {winner['weighted_score']:.1f})\n"
        )
        f.write(f"{'=' * 80}\n\n")

        f.write("Best per Category:\n")
        f.write(f"  ESM:       {best_esm['model_name']} ({best_esm['ESM']:.1f}%)\n")
        f.write(
            f"  EX(-S1):   {best_ex['model_name']} ({best_ex['EX_no_fallback']:.1f}%)\n"
        )
        f.write(
            f"  VX(-S1):   {best_vx['model_name']} ({best_vx['VX_no_fallback']:.1f}%)\n"
        )
        f.write(
            f"  Latency:   {best_lat['model_name']} ({format_latency_display(best_lat['avg_latency'])})\n"
        )
        f.write(
            f"  Fallback:  {least_fallback['model_name']} ({least_fallback['select1_rate']:.1f}%)\n"
        )

        f.write("\nBest EX per Difficulty:\n")
        if best_easy:
            f.write(
                f"  Easy:       {best_easy['model_name']} ({best_easy['difficulty_stats']['easy']['ex_rate']:.1f}%)\n"
            )
        if best_medium:
            f.write(
                f"  Medium:     {best_medium['model_name']} ({best_medium['difficulty_stats']['medium']['ex_rate']:.1f}%)\n"
            )
        if best_hard:
            f.write(
                f"  Hard:       {best_hard['model_name']} ({best_hard['difficulty_stats']['hard']['ex_rate']:.1f}%)\n"
            )
        if best_xhard:
            f.write(
                f"  Extra Hard: {best_xhard['model_name']} ({best_xhard['difficulty_stats']['extra_hard']['ex_rate']:.1f}%)\n"
            )

    print(f"\n  ✓ Saved ranking report: {report_file}")
    print()


if __name__ == "__main__":
    main()
