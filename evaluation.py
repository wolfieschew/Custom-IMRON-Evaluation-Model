"""
Evaluator: Membandingkan predicted SQL vs gold SQL (answer.sql)
Metrics: Exact Set Match (ESM), Execution Accuracy (EX), Syntax Validity (VX), Latency
"""
import os
import re
import csv
import time
from datetime import datetime
from sqlalchemy import create_engine, text as sa_text

# Configuration
DB_USER = os.getenv('DB_USER', 'root')
DB_PASS = os.getenv('DB_PASS', '')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '3306')
DB_NAME = os.getenv('DB_NAME', 'example_endpoint')

# Path
PRED_FILE = r"output\prediksi_imron_topk3_gemma3_4b_20260227_124328.txt"
GOLD_FILE = r"answer.sql"
OUTPUT_DIR = "output"


def create_engine_connection():
    connection_string = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(
        connection_string,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=3600
    )
    return engine


def load_predictions(path):
    """Load predicted SQL dari file output (format: SQL\\tdb_id)"""
    preds = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            sql = parts[0].rstrip(';').strip()
            db_id = parts[1].strip() if len(parts) > 1 else 'unknown'
            preds.append({'sql': sql, 'db_id': db_id})
    return preds


def load_golds(path):
    """Load gold SQL dari answer.sql (format: satu SQL per baris, dengan ;)"""
    golds = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('--'):
                continue
            sql = line.rstrip(';').strip()
            if sql:
                golds.append({'sql': sql})
    return golds


def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison"""
    sql = sql.lower().strip()
    sql = sql.rstrip(';')
    sql = re.sub(r'\s+', ' ', sql)
    sql = re.sub(r'\s*,\s*', ', ', sql)
    sql = re.sub(r'\(\s+', '(', sql)
    sql = re.sub(r'\s+\)', ')', sql)
    return sql.strip()


# ESM Module

def split_sql_columns(clause: str) -> list:
    """
    Split SQL columns dengan memperhatikan nested parentheses.
    Contoh: "count(*), sum(case when x then 1 end), name" → 3 elemen
    """
    columns = []
    depth = 0
    current = []

    for ch in clause:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            columns.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)

    if current:
        columns.append(''.join(current).strip())

    return [c for c in columns if c]


def extract_sql_components(sql: str) -> dict:
    """
    Extract komponen SQL untuk Exact Set Match (ESM).
    Membandingkan SELECT columns, FROM tables, WHERE conditions,
    GROUP BY, HAVING, ORDER BY sebagai SET (urutan tidak penting).
    """
    sql_norm = normalize_sql(sql)

    components = {
        'select': set(),
        'from': set(),
        'where': set(),
        'group_by': set(),
        'having': set(),
        'order_by': set(),
        'limit': '',
        'keywords': set(),
    }

    try:
        # Extract SELECT clause
        select_match = re.search(r'select\s+(.*?)\s+from\s', sql_norm, re.DOTALL)
        if select_match:
            select_clause = select_match.group(1).strip()
            if select_clause.startswith('distinct'):
                components['keywords'].add('distinct')
                select_clause = select_clause[8:].strip()
            cols = split_sql_columns(select_clause)
            for col in cols:
                col_clean = re.sub(r'\s+as\s+\w+', '', col).strip()
                if col_clean:
                    components['select'].add(col_clean)

        # Extract FROM clause
        from_match = re.search(
            r'\bfrom\s+(.*?)(?:\s+where\s|\s+group\s|\s+having\s|\s+order\s|\s+limit\s|$)',
            sql_norm, re.DOTALL
        )
        if from_match:
            from_clause = from_match.group(1).strip()
            tables = re.split(r'\s*(?:,|join)\s*', from_clause)
            for t in tables:
                t_clean = re.sub(r'\s+(?:as\s+)?\w+\s+on\s+.*', '', t).strip()
                t_clean = re.sub(r'\s+(?:as\s+)?\w+$', '', t_clean).strip()
                t_clean = re.sub(r'^\s*(?:left|right|inner|outer|cross)\s+', '', t_clean).strip()
                if t_clean:
                    components['from'].add(t_clean)

        # Extract WHERE clause
        where_match = re.search(
            r'\bwhere\s+(.*?)(?:\s+group\s|\s+having\s|\s+order\s|\s+limit\s|$)',
            sql_norm, re.DOTALL
        )
        if where_match:
            where_clause = where_match.group(1).strip()
            conditions = re.split(r'\s+(?:and|or)\s+', where_clause)
            for cond in conditions:
                cond_clean = cond.strip()
                if cond_clean:
                    components['where'].add(cond_clean)

        # Extract GROUP BY
        group_match = re.search(
            r'\bgroup\s+by\s+(.*?)(?:\s+having\s|\s+order\s|\s+limit\s|$)',
            sql_norm, re.DOTALL
        )
        if group_match:
            for g in group_match.group(1).strip().split(','):
                g_clean = g.strip()
                if g_clean:
                    components['group_by'].add(g_clean)

        # Extract HAVING
        having_match = re.search(
            r'\bhaving\s+(.*?)(?:\s+order\s|\s+limit\s|$)',
            sql_norm, re.DOTALL
        )
        if having_match:
            components['having'].add(having_match.group(1).strip())

        # Extract ORDER BY
        order_match = re.search(
            r'\border\s+by\s+(.*?)(?:\s+limit\s|$)',
            sql_norm, re.DOTALL
        )
        if order_match:
            for o in order_match.group(1).strip().split(','):
                o_clean = o.strip()
                if o_clean:
                    components['order_by'].add(o_clean)

        # Extract LIMIT
        limit_match = re.search(r'\blimit\s+(\d+)', sql_norm)
        if limit_match:
            components['limit'] = limit_match.group(1)

        # Detect aggregate keywords
        for kw in ['count(', 'sum(', 'avg(', 'min(', 'max(', 'group_concat(']:
            if kw in sql_norm:
                components['keywords'].add(kw.rstrip('('))

    except Exception:
        components['select'] = {sql_norm}

    return components


def exact_set_match(pred_sql: str, gold_sql: str) -> tuple:
    """
    Exact Set Match (ESM): Membandingkan SQL structure sebagai set.

    Return (is_match: bool, detail: dict)
    - Membandingkan: SELECT columns, FROM tables, WHERE conditions,
      GROUP BY, HAVING, ORDER BY sebagai SET.
    - Urutan kolom/kondisi TIDAK penting.
    - Alias TIDAK penting.
    """
    pred_comp = extract_sql_components(pred_sql)
    gold_comp = extract_sql_components(gold_sql)

    detail = {}
    all_match = True

    for key in ['select', 'from', 'where', 'group_by', 'having', 'order_by']:
        pred_set = pred_comp[key]
        gold_set = gold_comp[key]
        match = (pred_set == gold_set)
        detail[key] = {
            'match': match,
            'pred': pred_set,
            'gold': gold_set,
            'missing': gold_set - pred_set,
            'extra': pred_set - gold_set,
        }
        if not match:
            all_match = False

    # LIMIT comparison
    limit_match = pred_comp['limit'] == gold_comp['limit']
    detail['limit'] = {
        'match': limit_match,
        'pred': pred_comp['limit'],
        'gold': gold_comp['limit'],
    }
    if not limit_match:
        all_match = False

    # Keywords comparison (DISTINCT, aggregates)
    kw_match = pred_comp['keywords'] == gold_comp['keywords']
    detail['keywords'] = {
        'match': kw_match,
        'pred': pred_comp['keywords'],
        'gold': gold_comp['keywords'],
    }
    if not kw_match:
        all_match = False

    return all_match, detail


# ==================== SQL EXECUTION ====================

def execute_sql_safe(engine, sql, timeout=30):
    """
    Execute SQL dan return result sebagai sorted tuple list.
    Return (result, error_msg, latency_ms)
    """
    start_time = time.perf_counter()
    try:
        with engine.connect() as conn:
            result = conn.execute(sa_text(sql))
            rows = result.fetchall()
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            normalized = sorted([tuple(str(v) for v in row) for row in rows])
            return normalized, None, elapsed_ms
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return None, str(e), elapsed_ms


def check_syntax_validity(engine, sql):
    """
    Check apakah SQL syntax valid tanpa mengeksekusi secara penuh.
    Return (is_valid: bool, error_msg: str or None)
    """
    try:
        with engine.connect() as conn:
            conn.execute(sa_text(f"EXPLAIN {sql}"))
        return True, None
    except Exception as e:
        error_msg = str(e)
        if "doesn't support EXPLAIN" in error_msg or "EXPLAIN" in error_msg:
            try:
                with engine.connect() as conn:
                    conn.execute(sa_text(f"SELECT * FROM ({sql}) AS __syntax_check LIMIT 0"))
                return True, None
            except Exception as e2:
                return False, str(e2)
        return False, error_msg


def compare_results(pred_result, gold_result):
    """
    Compare execution results.
    Returns: 'match', 'mismatch', 'both_error', 'pred_error', 'gold_error'
    """
    pred_is_none = pred_result is None
    gold_is_none = gold_result is None

    if pred_is_none and gold_is_none:
        return 'both_error'
    if pred_is_none:
        return 'pred_error'
    if gold_is_none:
        return 'gold_error'

    return 'match' if pred_result == gold_result else 'mismatch'


def format_latency(ms):
    """Format latency for display"""
    if ms < 1:
        return f"{ms:.3f}ms"
    elif ms < 1000:
        return f"{ms:.1f}ms"
    else:
        return f"{ms/1000:.2f}s"


# ==================== MAIN ====================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*90}")
    print(f"SQL EVALUATION: Predicted vs Gold (answer.sql)")
    print(f"{'='*90}")
    print(f"Predictions: {PRED_FILE}")
    print(f"Gold:        {GOLD_FILE}")
    print(f"Database:    {DB_HOST}:{DB_PORT}/{DB_NAME}")
    print(f"{'='*90}\n")

    # Load files
    preds = load_predictions(PRED_FILE)
    golds = load_golds(GOLD_FILE)

    print(f"Predictions loaded: {len(preds)}")
    print(f"Golds loaded:       {len(golds)}")

    if len(preds) != len(golds):
        print(f"\n  WARNING: Jumlah baris TIDAK SAMA! pred={len(preds)} vs gold={len(golds)}")
        print(f"    Evaluasi akan dilakukan untuk {min(len(preds), len(golds))} baris pertama.\n")

    min_len = min(len(preds), len(golds))

    # Connect to database
    print(f"\nConnecting to MySQL...")
    engine = create_engine_connection()
    try:
        with engine.connect() as conn:
            conn.execute(sa_text("SELECT 1"))
        print(f"  ✓ Connection successful\n")
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        return

    # ==================== EVALUATION ====================
    results = []

    # Counters
    esm_match_count = 0
    exec_match_count = 0
    valid_pred_count = 0
    valid_gold_count = 0
    both_error_count = 0
    pred_error_count = 0
    gold_error_count = 0
    select1_count = 0

    # Syntax validity counters
    syntax_valid_pred_count = 0
    syntax_valid_gold_count = 0

    # Latency tracking
    pred_latencies = []
    gold_latencies = []

    print(f"{'='*90}")
    print(f"{'#':>3} | {'ESM':>3} | {'EX':>3} | {'VX':>3} | {'Pred Lat':>10} | {'Gold Lat':>10} | Detail")
    print(f"{'-'*90}")

    for i in range(min_len):
        pred_sql = preds[i]['sql']
        gold_sql = golds[i]['sql']
        db_id = preds[i]['db_id']

        # --- Exact Set Match (ESM) ---
        is_esm_match, esm_detail = exact_set_match(pred_sql, gold_sql)

        if is_esm_match:
            esm_match_count += 1

        # --- Check SELECT 1 fallback ---
        is_select1 = pred_sql.strip().upper() in ('SELECT 1', 'SELECT 1 AS PLACEHOLDER')
        if is_select1:
            select1_count += 1

        # --- Syntax Validity (VX) ---
        pred_syntax_valid, pred_syntax_err = check_syntax_validity(engine, pred_sql)
        gold_syntax_valid, gold_syntax_err = check_syntax_validity(engine, gold_sql)

        if pred_syntax_valid:
            syntax_valid_pred_count += 1
        if gold_syntax_valid:
            syntax_valid_gold_count += 1

        # --- Execution Accuracy (EX) with Latency ---
        pred_result, pred_err, pred_latency = execute_sql_safe(engine, pred_sql)
        gold_result, gold_err, gold_latency = execute_sql_safe(engine, gold_sql)

        if pred_err is None:
            valid_pred_count += 1
            pred_latencies.append(pred_latency)
        if gold_err is None:
            valid_gold_count += 1
            gold_latencies.append(gold_latency)

        comparison = compare_results(pred_result, gold_result)

        is_exec_match = False
        if comparison == 'match':
            is_exec_match = True
            exec_match_count += 1
        elif comparison == 'both_error':
            both_error_count += 1
        elif comparison == 'pred_error':
            pred_error_count += 1
        elif comparison == 'gold_error':
            gold_error_count += 1

        esm_icon = "OK" if is_esm_match else "  "
        ex_icon = "OK" if is_exec_match else "False"
        vx_icon = "OK" if pred_syntax_valid else "False"
        pred_lat_str = format_latency(pred_latency)
        gold_lat_str = format_latency(gold_latency)

        # Print detail untuk mismatch
        if not is_exec_match:
            print(f"{i+1:>3} | {esm_icon:>3} | {ex_icon:>3} | {vx_icon:>3} | {pred_lat_str:>10} | {gold_lat_str:>10} | {db_id}")
            print(f"    | GOLD: {gold_sql[:70]}")
            print(f"    | PRED: {pred_sql[:70]}")
            # Show ESM mismatch detail
            if not is_esm_match:
                for comp_name, comp_info in esm_detail.items():
                    if comp_name in ('limit', 'keywords'):
                        continue
                    if not comp_info['match'] and (comp_info.get('missing') or comp_info.get('extra')):
                        missing_str = ', '.join(comp_info['missing']) if comp_info.get('missing') else ''
                        extra_str = ', '.join(comp_info['extra']) if comp_info.get('extra') else ''
                        if missing_str:
                            print(f"    | ESM {comp_name.upper()} missing: {missing_str[:60]}")
                        if extra_str:
                            print(f"    | ESM {comp_name.upper()} extra: {extra_str[:60]}")
            if not pred_syntax_valid:
                print(f"    | SYNTAX ERR: {pred_syntax_err[:60]}")
            if pred_err:
                print(f"    | PRED EXEC ERR: {pred_err[:60]}")
            if gold_err:
                print(f"    | GOLD EXEC ERR: {gold_err[:60]}")
            if pred_result is not None and gold_result is not None:
                print(f"    | PRED rows: {len(pred_result)}, GOLD rows: {len(gold_result)}")
            print()
        else:
            print(f"{i+1:>3} | {esm_icon:>3} | {ex_icon:>3} | {vx_icon:>3} | {pred_lat_str:>10} | {gold_lat_str:>10} | OK")

        results.append({
            'question_id': i + 1,
            'db_id': db_id,
            'gold_sql': gold_sql,
            'predicted_sql': pred_sql,
            'esm_match': is_esm_match,
            'exec_match': is_exec_match,
            'is_select1': is_select1,
            'pred_syntax_valid': pred_syntax_valid,
            'gold_syntax_valid': gold_syntax_valid,
            'pred_valid': pred_err is None,
            'gold_valid': gold_err is None,
            'comparison': comparison,
            'pred_error': pred_err or '',
            'gold_error': gold_err or '',
            'pred_syntax_error': pred_syntax_err or '',
            'pred_latency_ms': round(pred_latency, 3),
            'gold_latency_ms': round(gold_latency, 3),
        })

    # ==================== METRICS CALCULATION ====================

    esm_accuracy = esm_match_count / min_len * 100 if min_len > 0 else 0
    ex_accuracy = exec_match_count / min_len * 100 if min_len > 0 else 0
    valid_rate = valid_pred_count / min_len * 100 if min_len > 0 else 0
    syntax_valid_rate = syntax_valid_pred_count / min_len * 100 if min_len > 0 else 0
    select1_rate = select1_count / min_len * 100 if min_len > 0 else 0

    # EX tanpa SELECT 1
    non_fallback = min_len - select1_count
    ex_no_fallback = sum(
        1 for r in results if r['exec_match'] and not r['is_select1']
    )
    ex_accuracy_no_fallback = ex_no_fallback / non_fallback * 100 if non_fallback > 0 else 0

    # Latency statistics
    def calc_latency_stats(latencies):
        if not latencies:
            return {'avg': 0, 'min': 0, 'max': 0, 'median': 0, 'p95': 0, 'total': 0, 'count': 0}
        sorted_lat = sorted(latencies)
        n = len(sorted_lat)
        return {
            'avg': sum(sorted_lat) / n,
            'min': sorted_lat[0],
            'max': sorted_lat[-1],
            'median': sorted_lat[n // 2],
            'p95': sorted_lat[int(n * 0.95)] if n > 1 else sorted_lat[0],
            'total': sum(sorted_lat),
            'count': n,
        }

    pred_lat_stats = calc_latency_stats(pred_latencies)
    gold_lat_stats = calc_latency_stats(gold_latencies)

    # ==================== PRINT RESULTS ====================

    print(f"\n{'='*90}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*90}")
    print(f"Total Questions:           {min_len}")
    print(f"")
    print(f"--- Accuracy Metrics ---")
    print(f"Exact Set Match (ESM):     {esm_match_count}/{min_len} = {esm_accuracy:.1f}%")
    print(f"Execution Accuracy (EX):   {exec_match_count}/{min_len} = {ex_accuracy:.1f}%")
    print(f"EX (excl. SELECT 1):       {ex_no_fallback}/{non_fallback} = {ex_accuracy_no_fallback:.1f}%")
    print(f"")
    print(f"--- Syntax & Validity ---")
    print(f"Syntax Validity (VX):      {syntax_valid_pred_count}/{min_len} = {syntax_valid_rate:.1f}%")
    print(f"Execution Validity:        {valid_pred_count}/{min_len} = {valid_rate:.1f}%")
    print(f"SELECT 1 Fallbacks:        {select1_count}/{min_len} = {select1_rate:.1f}%")
    print(f"")
    print(f"--- Latency (Predicted SQL, hanya yang valid) ---")
    if pred_latencies:
        print(f"Avg Latency:               {format_latency(pred_lat_stats['avg'])}")
        print(f"Min Latency:               {format_latency(pred_lat_stats['min'])}")
        print(f"Max Latency:               {format_latency(pred_lat_stats['max'])}")
        print(f"Median Latency:            {format_latency(pred_lat_stats['median'])}")
        print(f"P95 Latency:               {format_latency(pred_lat_stats['p95'])}")
        print(f"Total Execution Time:      {format_latency(pred_lat_stats['total'])}")
        print(f"Queries Measured:          {pred_lat_stats['count']}")
    else:
        print(f"  (no valid predictions to measure)")
    print(f"")
    print(f"--- Latency (Gold SQL, reference) ---")
    if gold_latencies:
        print(f"Avg Latency:               {format_latency(gold_lat_stats['avg'])}")
        print(f"Median Latency:            {format_latency(gold_lat_stats['median'])}")
        print(f"P95 Latency:               {format_latency(gold_lat_stats['p95'])}")
    print(f"")
    print(f"--- Error Analysis ---")
    print(f"Both Error:                {both_error_count}")
    print(f"Pred Error Only:           {pred_error_count}")
    print(f"Gold Error Only:           {gold_error_count}")
    print(f"Gold Syntax Valid:         {syntax_valid_gold_count}/{min_len}")
    print(f"{'='*90}")

    # ==================== SAVE DETAILED RESULTS ====================

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_csv = os.path.join(OUTPUT_DIR, f"evaluation_detail_{timestamp}.csv")
    eval_summary = os.path.join(OUTPUT_DIR, f"evaluation_summary_{timestamp}.txt")

    # Save CSV
    with open(eval_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['question_id', 'db_id', 'gold_sql', 'predicted_sql',
                      'esm_match', 'exec_match', 'pred_syntax_valid', 'gold_syntax_valid',
                      'is_select1', 'pred_valid', 'gold_valid', 'comparison',
                      'pred_latency_ms', 'gold_latency_ms',
                      'pred_error', 'gold_error', 'pred_syntax_error']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  ✓ Saved detail: {eval_csv}")

    # Save summary
    with open(eval_summary, 'w', encoding='utf-8') as f:
        f.write(f"EVALUATION SUMMARY\n")
        f.write(f"{'='*60}\n")
        f.write(f"Predictions: {PRED_FILE}\n")
        f.write(f"Gold: {GOLD_FILE}\n")
        f.write(f"Database: {DB_HOST}:{DB_PORT}/{DB_NAME}\n")
        f.write(f"Date: {datetime.now().isoformat()}\n\n")
        f.write(f"Total Questions: {min_len}\n\n")
        f.write(f"--- Accuracy ---\n")
        f.write(f"Exact Set Match (ESM): {esm_match_count}/{min_len} = {esm_accuracy:.1f}%\n")
        f.write(f"Execution Accuracy (EX): {exec_match_count}/{min_len} = {ex_accuracy:.1f}%\n")
        f.write(f"EX (excl. SELECT 1): {ex_no_fallback}/{non_fallback} = {ex_accuracy_no_fallback:.1f}%\n\n")
        f.write(f"--- Syntax & Validity ---\n")
        f.write(f"Syntax Validity (VX): {syntax_valid_pred_count}/{min_len} = {syntax_valid_rate:.1f}%\n")
        f.write(f"Execution Validity: {valid_pred_count}/{min_len} = {valid_rate:.1f}%\n")
        f.write(f"SELECT 1 Fallbacks: {select1_count}/{min_len} = {select1_rate:.1f}%\n\n")
        f.write(f"--- Latency (Predicted SQL) ---\n")
        if pred_latencies:
            f.write(f"Avg: {pred_lat_stats['avg']:.3f}ms\n")
            f.write(f"Min: {pred_lat_stats['min']:.3f}ms\n")
            f.write(f"Max: {pred_lat_stats['max']:.3f}ms\n")
            f.write(f"Median: {pred_lat_stats['median']:.3f}ms\n")
            f.write(f"P95: {pred_lat_stats['p95']:.3f}ms\n")
            f.write(f"Total: {pred_lat_stats['total']:.3f}ms\n")
            f.write(f"Queries Measured: {pred_lat_stats['count']}\n\n")
        f.write(f"--- Error Analysis ---\n")
        f.write(f"Both Error: {both_error_count}\n")
        f.write(f"Pred Error Only: {pred_error_count}\n")
        f.write(f"Gold Error Only: {gold_error_count}\n")
    print(f"  ✓ Saved summary: {eval_summary}")

    # ==================== PER-DOMAIN BREAKDOWN ====================

    print(f"\n{'='*90}")
    print(f"PER-DOMAIN BREAKDOWN")
    print(f"{'='*90}")

    domain_stats = {}
    for r in results:
        db = r['db_id']
        if db not in domain_stats:
            domain_stats[db] = {'total': 0, 'esm': 0, 'ex': 0, 'vx': 0, 'latencies': []}
        domain_stats[db]['total'] += 1
        if r['esm_match']:
            domain_stats[db]['esm'] += 1
        if r['exec_match']:
            domain_stats[db]['ex'] += 1
        if r['pred_syntax_valid']:
            domain_stats[db]['vx'] += 1
        if r['pred_valid']:
            domain_stats[db]['latencies'].append(r['pred_latency_ms'])

    print(f"{'Domain':<40} | {'Total':>5} | {'ESM':>8} | {'EX':>8} | {'VX':>8} | {'Avg Lat':>10}")
    print(f"{'-'*90}")
    for db, stats in sorted(domain_stats.items()):
        t = stats['total']
        esm = stats['esm']
        ex = stats['ex']
        vx = stats['vx']
        avg_lat = sum(stats['latencies']) / len(stats['latencies']) if stats['latencies'] else 0
        print(f"{db:<40} | {t:>5} | {esm}/{t} {esm/t*100:>3.0f}% | {ex}/{t} {ex/t*100:>3.0f}% | {vx}/{t} {vx/t*100:>3.0f}% | {format_latency(avg_lat):>10}")

    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()