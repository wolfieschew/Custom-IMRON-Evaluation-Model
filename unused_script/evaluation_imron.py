"""
Evaluation Script untuk IMRON Domain (MySQL)
Metrics: EX (Execution Accuracy), ESM (Exact Set Match)
Kompatibel dengan format output generate_predictions_llamaindex.py
"""

import os
import sys
import csv
import re
import json
import sqlparse
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from sqlalchemy import create_engine, text as sa_text
from collections import OrderedDict


# ==================== CONFIGURATION ====================

DB_USER = os.getenv('DB_USER', 'root')
DB_PASS = os.getenv('DB_PASS', '')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '3306')
DB_NAME = os.getenv('DB_NAME', 'example_endpoint')

# File paths
GOLD_SQL_FILE = "gold.sql"           # Gold standard SQL answers
QUESTIONS_FILE = "questions.sql"      # Questions with db_id
OUTPUT_DIR = "eval_results"

EXECUTION_TIMEOUT = 30  # seconds


# ==================== DATABASE CONNECTION ====================

def create_mysql_engine():
    connection_string = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(
        connection_string,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=3600,
        connect_args={'connect_timeout': EXECUTION_TIMEOUT}
    )
    return engine


# ==================== SQL NORMALIZATION (for ESM) ====================

def normalize_sql(sql: str) -> str:
    """Normalize SQL for exact set match comparison"""
    if not sql or sql.strip().lower() == 'select 1':
        return ''

    sql = sql.strip().rstrip(';').strip()

    # Lowercase
    sql = sql.lower()

    # Normalize whitespace
    sql = re.sub(r'\s+', ' ', sql)

    # Remove aliases after AS (for comparison fairness)
    # e.g. "count(*) as total" -> "count(*)"
    sql = re.sub(r'\s+as\s+\w+', '', sql)

    # Normalize quotes
    sql = sql.replace('"', "'")

    # Remove LIMIT clause (often differs between gold and pred)
    sql = re.sub(r'\s+limit\s+\d+', '', sql)

    # Normalize spaces around operators
    sql = re.sub(r'\s*=\s*', ' = ', sql)
    sql = re.sub(r'\s*>\s*', ' > ', sql)
    sql = re.sub(r'\s*<\s*', ' < ', sql)
    sql = re.sub(r'\s*!=\s*', ' != ', sql)
    sql = re.sub(r'\s*>=\s*', ' >= ', sql)
    sql = re.sub(r'\s*<=\s*', ' <= ', sql)

    # Final whitespace cleanup
    sql = re.sub(r'\s+', ' ', sql).strip()

    return sql


def tokenize_sql(sql: str) -> List[str]:
    """Tokenize normalized SQL into keyword/value tokens for set comparison"""
    normalized = normalize_sql(sql)
    if not normalized:
        return []

    # Use sqlparse for tokenization
    parsed = sqlparse.parse(normalized)
    if not parsed:
        return normalized.split()

    tokens = []
    for statement in parsed:
        for token in statement.flatten():
            t = str(token).strip()
            if t:
                tokens.append(t)

    return tokens


# ==================== EXECUTION ACCURACY (EX) ====================

def execute_sql_on_mysql(engine, sql: str) -> Tuple[Optional[List], Optional[str]]:
    """Execute SQL query on MySQL and return results or error"""
    if not sql or sql.strip().lower() == 'select 1':
        return None, "FALLBACK_QUERY"

    try:
        with engine.connect() as conn:
            result = conn.execute(sa_text(sql))
            rows = result.fetchall()
            # Convert to list of tuples for comparison
            result_set = [tuple(row) for row in rows]
            return result_set, None
    except Exception as e:
        return None, str(e)


def compare_results(gold_results: Optional[List], pred_results: Optional[List]) -> bool:
    """
    Compare execution results (EX metric).
    Results match if they contain the same set of rows (order-insensitive).
    """
    if gold_results is None and pred_results is None:
        return False  # Both failed = not a match

    if gold_results is None or pred_results is None:
        return False  # One failed = not a match

    # Convert to comparable format (handle Decimal, datetime, etc.)
    def normalize_value(val):
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        try:
            return float(val)
        except (ValueError, TypeError):
            return str(val)

    def normalize_row(row):
        return tuple(normalize_value(v) for v in row)

    gold_normalized = sorted([normalize_row(r) for r in gold_results])
    pred_normalized = sorted([normalize_row(r) for r in pred_results])

    return gold_normalized == pred_normalized


# ==================== EXACT SET MATCH (ESM) ====================

def compute_esm(gold_sql: str, pred_sql: str) -> bool:
    """
    Exact Set Match: compare SQL structure token-by-token (set-based).
    Two SQLs match if they have the same set of tokens after normalization.
    """
    gold_tokens = set(tokenize_sql(gold_sql))
    pred_tokens = set(tokenize_sql(pred_sql))

    if not gold_tokens and not pred_tokens:
        return False

    return gold_tokens == pred_tokens


# ==================== SQL COMPONENT MATCHING (Detailed ESM) ====================

def extract_sql_components(sql: str) -> Dict[str, str]:
    """Extract SQL components (SELECT, FROM, WHERE, GROUP BY, ORDER BY, etc.)"""
    normalized = normalize_sql(sql)
    if not normalized:
        return {}

    components = {}

    # Extract SELECT clause
    select_match = re.search(r'select\s+(.*?)(?:\s+from\s+|$)', normalized)
    if select_match:
        components['select'] = select_match.group(1).strip()

    # Extract FROM clause
    from_match = re.search(r'from\s+(.*?)(?:\s+where\s+|\s+group\s+|\s+order\s+|\s+limit\s+|\s+having\s+|$)', normalized)
    if from_match:
        components['from'] = from_match.group(1).strip()

    # Extract WHERE clause
    where_match = re.search(r'where\s+(.*?)(?:\s+group\s+|\s+order\s+|\s+limit\s+|\s+having\s+|$)', normalized)
    if where_match:
        components['where'] = where_match.group(1).strip()

    # Extract GROUP BY clause
    group_match = re.search(r'group\s+by\s+(.*?)(?:\s+having\s+|\s+order\s+|\s+limit\s+|$)', normalized)
    if group_match:
        components['group_by'] = group_match.group(1).strip()

    # Extract HAVING clause
    having_match = re.search(r'having\s+(.*?)(?:\s+order\s+|\s+limit\s+|$)', normalized)
    if having_match:
        components['having'] = having_match.group(1).strip()

    # Extract ORDER BY clause
    order_match = re.search(r'order\s+by\s+(.*?)(?:\s+limit\s+|$)', normalized)
    if order_match:
        components['order_by'] = order_match.group(1).strip()

    return components


def compute_component_match(gold_sql: str, pred_sql: str) -> Dict[str, bool]:
    """Compare SQL component by component"""
    gold_comp = extract_sql_components(gold_sql)
    pred_comp = extract_sql_components(pred_sql)

    all_keys = set(list(gold_comp.keys()) + list(pred_comp.keys()))
    matches = {}

    for key in all_keys:
        gold_val = gold_comp.get(key, '')
        pred_val = pred_comp.get(key, '')
        matches[key] = gold_val == pred_val

    return matches


# ==================== DIFFICULTY CLASSIFICATION ====================

def classify_difficulty(sql: str) -> str:
    """Classify SQL difficulty: easy, medium, hard, extra"""
    sql_lower = sql.lower()

    has_subquery = 'select' in sql_lower[sql_lower.find('from'):] if 'from' in sql_lower else False
    has_join = 'join' in sql_lower
    has_union = 'union' in sql_lower
    has_group = 'group by' in sql_lower
    has_having = 'having' in sql_lower
    has_nested_json = sql_lower.count('json_extract') > 1
    has_case = 'case' in sql_lower
    has_multiple_conditions = sql_lower.count(' and ') + sql_lower.count(' or ') >= 2

    # Score complexity
    complexity = 0
    if has_subquery: complexity += 2
    if has_join: complexity += 2
    if has_union: complexity += 2
    if has_group: complexity += 1
    if has_having: complexity += 1
    if has_nested_json: complexity += 1
    if has_case: complexity += 1
    if has_multiple_conditions: complexity += 1

    if complexity >= 4:
        return 'extra'
    elif complexity >= 3:
        return 'hard'
    elif complexity >= 1:
        return 'medium'
    else:
        return 'easy'


# ==================== FILE PARSING ====================

def parse_predictions_file(pred_file: str) -> List[Dict]:
    """Parse prediction file (format: SQL\tdb_id per line)"""
    predictions = []
    with open(pred_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                predictions.append({
                    'id': i,
                    'sql': parts[0].strip(),
                    'db_id': parts[1].strip()
                })
            else:
                predictions.append({
                    'id': i,
                    'sql': line.strip(),
                    'db_id': 'unknown'
                })
    return predictions


def parse_gold_file(gold_file: str) -> List[Dict]:
    """Parse gold SQL file (format: SQL\tdb_id per line)"""
    golds = []
    with open(gold_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('--'):
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                golds.append({
                    'id': i,
                    'sql': parts[0].strip(),
                    'db_id': parts[1].strip()
                })
            else:
                golds.append({
                    'id': i,
                    'sql': line.strip(),
                    'db_id': 'unknown'
                })
    return golds


def parse_questions_file(questions_file: str) -> List[Dict]:
    """Parse questions.sql file"""
    questions = []
    with open(questions_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('--'):
                continue
            if line.startswith('Question') and '|||' in line:
                try:
                    parts = line.split('|||')
                    question_part = parts[0].strip()
                    db_id = parts[1].strip()
                    colon_idx = question_part.find(':')
                    q_num = int(question_part[:colon_idx].replace('Question', '').strip())
                    question_text = question_part[colon_idx + 1:].strip()
                    questions.append({
                        'id': q_num,
                        'question': question_text,
                        'db_id': db_id
                    })
                except:
                    continue
    return questions


# ==================== MAIN EVALUATION ====================

def evaluate(pred_file: str, gold_file: str, questions_file: str = None):
    """Run full evaluation: EX + ESM"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract model name from prediction filename
    pred_basename = os.path.basename(pred_file)
    model_tag = pred_basename.replace('prediksi_imron_', '').replace('.txt', '')

    print(f"\n{'='*80}")
    print(f"IMRON DOMAIN EVALUATION (MySQL)")
    print(f"{'='*80}")
    print(f"Prediction file: {pred_file}")
    print(f"Gold file:       {gold_file}")
    print(f"Questions file:  {questions_file or 'N/A'}")
    print(f"Database:        {DB_HOST}:{DB_PORT}/{DB_NAME}")
    print(f"{'='*80}\n")

    # Parse files
    predictions = parse_predictions_file(pred_file)
    golds = parse_gold_file(gold_file)
    questions = parse_questions_file(questions_file) if questions_file and os.path.exists(questions_file) else None

    if len(predictions) != len(golds):
        print(f"[ERROR] Prediction count ({len(predictions)}) != Gold count ({len(golds)})")
        print(f"  Proceeding with min({len(predictions)}, {len(golds)}) pairs...")

    n = min(len(predictions), len(golds))

    # Connect to MySQL
    print("Connecting to MySQL for execution accuracy...")
    try:
        engine = create_mysql_engine()
        with engine.connect() as conn:
            conn.execute(sa_text("SELECT 1"))
        print("  ✓ MySQL connection successful\n")
    except Exception as e:
        print(f"  [ERROR] MySQL connection failed: {e}")
        print(f"  Will skip EX metric, only compute ESM.\n")
        engine = None

    # ==================== EVALUATE EACH PAIR ====================

    results = []
    ex_correct = 0
    esm_correct = 0
    exec_errors_gold = 0
    exec_errors_pred = 0
    fallback_count = 0

    # Per-difficulty counters
    difficulty_stats = {
        'easy': {'total': 0, 'ex': 0, 'esm': 0},
        'medium': {'total': 0, 'ex': 0, 'esm': 0},
        'hard': {'total': 0, 'ex': 0, 'esm': 0},
        'extra': {'total': 0, 'ex': 0, 'esm': 0},
    }

    # Per-domain counters
    domain_stats = {}

    for i in range(n):
        gold = golds[i]
        pred = predictions[i]
        question_text = questions[i]['question'] if questions and i < len(questions) else 'N/A'

        gold_sql = gold['sql']
        pred_sql = pred['sql']
        db_id = gold['db_id']

        difficulty = classify_difficulty(gold_sql)

        # Track per-domain
        if db_id not in domain_stats:
            domain_stats[db_id] = {'total': 0, 'ex': 0, 'esm': 0}
        domain_stats[db_id]['total'] += 1
        difficulty_stats[difficulty]['total'] += 1

        # Check fallback
        is_fallback = pred_sql.strip().lower() == 'select 1'
        if is_fallback:
            fallback_count += 1

        # ESM (Exact Set Match)
        esm_match = compute_esm(gold_sql, pred_sql)
        if esm_match:
            esm_correct += 1
            domain_stats[db_id]['esm'] += 1
            difficulty_stats[difficulty]['esm'] += 1

        # EX (Execution Accuracy)
        ex_match = False
        gold_error = None
        pred_error = None

        if engine is not None:
            gold_results, gold_error = execute_sql_on_mysql(engine, gold_sql)
            pred_results, pred_error = execute_sql_on_mysql(engine, pred_sql)

            if gold_error:
                exec_errors_gold += 1
            if pred_error:
                exec_errors_pred += 1

            ex_match = compare_results(gold_results, pred_results)
            if ex_match:
                ex_correct += 1
                domain_stats[db_id]['ex'] += 1
                difficulty_stats[difficulty]['ex'] += 1

        # Component match
        comp_match = compute_component_match(gold_sql, pred_sql)

        result = {
            'question_id': i + 1,
            'db_id': db_id,
            'question': question_text,
            'difficulty': difficulty,
            'gold_sql': gold_sql,
            'pred_sql': pred_sql,
            'ex_match': 1 if ex_match else 0,
            'esm_match': 1 if esm_match else 0,
            'is_fallback': 1 if is_fallback else 0,
            'gold_exec_error': gold_error or '',
            'pred_exec_error': pred_error or '',
            'select_match': 1 if comp_match.get('select', False) else 0,
            'from_match': 1 if comp_match.get('from', False) else 0,
            'where_match': 1 if comp_match.get('where', False) else 0,
            'group_by_match': 1 if comp_match.get('group_by', False) else 0,
            'order_by_match': 1 if comp_match.get('order_by', False) else 0,
        }
        results.append(result)

        # Print progress
        status = ''
        if ex_match and esm_match:
            status = '✓✓ (EX+ESM)'
        elif ex_match:
            status = '✓  (EX only)'
        elif esm_match:
            status = '~  (ESM only)'
        else:
            status = '✗  (miss)'

        if is_fallback:
            status = '⚠  (FALLBACK)'

        print(f"  [{i+1}/{n}] Q{i+1} [{difficulty:6s}] {status}")

    # ==================== COMPUTE FINAL METRICS ====================

    ex_accuracy = (ex_correct / n * 100) if n > 0 else 0
    esm_accuracy = (esm_correct / n * 100) if n > 0 else 0

    # ==================== SAVE DETAILED RESULTS ====================

    detail_file = os.path.join(OUTPUT_DIR, f"eval_detail_{model_tag}_{timestamp}.csv")
    with open(detail_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['question_id', 'db_id', 'question', 'difficulty',
                      'gold_sql', 'pred_sql', 'ex_match', 'esm_match',
                      'is_fallback', 'gold_exec_error', 'pred_exec_error',
                      'select_match', 'from_match', 'where_match',
                      'group_by_match', 'order_by_match']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # ==================== SAVE SUMMARY ====================

    summary_file = os.path.join(OUTPUT_DIR, f"eval_summary_{model_tag}_{timestamp}.txt")
    summary_lines = []

    def log(msg):
        print(msg)
        summary_lines.append(msg)

    log(f"\n{'='*80}")
    log(f"EVALUATION RESULTS")
    log(f"{'='*80}")
    log(f"Model: {model_tag}")
    log(f"Total Questions: {n}")
    log(f"Fallbacks (SELECT 1): {fallback_count}")
    log(f"")
    log(f"{'='*60}")
    log(f"  METRIC              CORRECT    TOTAL    ACCURACY")
    log(f"{'='*60}")
    log(f"  EX  (Execution)     {ex_correct:>5}      {n:>5}    {ex_accuracy:>6.2f}%")
    log(f"  ESM (Exact Set)     {esm_correct:>5}      {n:>5}    {esm_accuracy:>6.2f}%")
    log(f"{'='*60}")

    if engine is not None:
        log(f"\n  Execution Errors:")
        log(f"    Gold SQL errors:  {exec_errors_gold}")
        log(f"    Pred SQL errors:  {exec_errors_pred}")

    # Per-difficulty breakdown
    log(f"\n{'='*60}")
    log(f"  DIFFICULTY BREAKDOWN")
    log(f"{'='*60}")
    log(f"  {'Difficulty':<10} {'Total':>6} {'EX':>6} {'EX%':>8} {'ESM':>6} {'ESM%':>8}")
    log(f"  {'-'*50}")
    for diff in ['easy', 'medium', 'hard', 'extra']:
        stats = difficulty_stats[diff]
        if stats['total'] > 0:
            ex_pct = stats['ex'] / stats['total'] * 100
            esm_pct = stats['esm'] / stats['total'] * 100
            log(f"  {diff:<10} {stats['total']:>6} {stats['ex']:>6} {ex_pct:>7.1f}% {stats['esm']:>6} {esm_pct:>7.1f}%")
        else:
            log(f"  {diff:<10} {stats['total']:>6}      -        -      -        -")

    # Per-domain breakdown
    log(f"\n{'='*60}")
    log(f"  DOMAIN BREAKDOWN")
    log(f"{'='*60}")
    log(f"  {'Domain':<45} {'Total':>5} {'EX%':>7} {'ESM%':>7}")
    log(f"  {'-'*65}")
    for db_id, stats in domain_stats.items():
        if stats['total'] > 0:
            ex_pct = stats['ex'] / stats['total'] * 100
            esm_pct = stats['esm'] / stats['total'] * 100
            log(f"  {db_id:<45} {stats['total']:>5} {ex_pct:>6.1f}% {esm_pct:>6.1f}%")

    log(f"\n{'='*60}")
    log(f"  Output Files:")
    log(f"    Detail: {detail_file}")
    log(f"    Summary: {summary_file}")
    log(f"{'='*60}\n")

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))

    return {
        'ex_accuracy': ex_accuracy,
        'esm_accuracy': esm_accuracy,
        'total': n,
        'ex_correct': ex_correct,
        'esm_correct': esm_correct,
    }


# ==================== CLI ====================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='IMRON Domain SQL Evaluation (EX + ESM)')
    parser.add_argument('--pred', '-p', required=True, help='Prediction file path')
    parser.add_argument('--gold', '-g', required=True, help='Gold SQL file path')
    parser.add_argument('--questions', '-q', default='questions.sql', help='Questions file path')

    args = parser.parse_args()

    evaluate(
        pred_file=args.pred,
        gold_file=args.gold,
        questions_file=args.questions
    )


if __name__ == "__main__":
    main()