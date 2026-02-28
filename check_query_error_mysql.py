"""
Script untuk testing apakah semua SQL di answer.sql valid dan bisa dieksekusi di MySQL.
Mengecek: syntax validity, execution, dan jumlah rows yang dikembalikan.
"""
import os
from sqlalchemy import create_engine, text as sa_text

# ==================== CONFIGURATION ====================
DB_USER = os.getenv('DB_USER', 'root')
DB_PASS = os.getenv('DB_PASS', '')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '3306')
DB_NAME = os.getenv('DB_NAME', 'example_endpoint')

ANSWER_FILE = "answer.sql"


def load_queries(path):
    """Load SQL queries dari file (satu SQL per baris, strip komentar)"""
    queries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('--'):
                continue
            sql = line.rstrip(';').strip()
            if sql:
                queries.append({'line': line_num, 'sql': sql})
    return queries


def test_query(engine, sql):
    """
    Test satu query. Return dict dengan:
    - is_valid: bool (bisa dieksekusi tanpa error)
    - rows: jumlah rows yang dikembalikan
    - columns: list nama kolom
    - sample: 3 baris pertama (preview)
    - error: pesan error jika gagal
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(sa_text(sql))
            rows = result.fetchall()
            columns = list(result.keys()) if result.keys() else []
            sample = rows[:3]
            return {
                'is_valid': True,
                'rows': len(rows),
                'columns': columns,
                'sample': sample,
                'error': None
            }
    except Exception as e:
        return {
            'is_valid': False,
            'rows': 0,
            'columns': [],
            'sample': [],
            'error': str(e)
        }


def main():
    print(f"\n{'='*80}")
    print(f"ANSWER.SQL VALIDATION TEST")
    print(f"{'='*80}")
    print(f"File:     {ANSWER_FILE}")
    print(f"Database: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    print(f"{'='*80}\n")

    # Load queries
    queries = load_queries(ANSWER_FILE)
    print(f"Total queries loaded: {len(queries)}\n")

    # Connect
    engine = create_engine(
        f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    try:
        with engine.connect() as conn:
            conn.execute(sa_text("SELECT 1"))
        print(f"✓ Database connection OK\n")
    except Exception as e:
        print(f"✗ Database connection FAILED: {e}")
        return

    # Test each query
    valid_count = 0
    invalid_count = 0
    empty_count = 0
    results = []

    print(f"{'#':>3} | {'Status':>6} | {'Rows':>6} | SQL (preview)")
    print(f"{'-'*80}")

    for i, q in enumerate(queries):
        result = test_query(engine, q['sql'])
        results.append({**q, **result})

        if result['is_valid']:
            valid_count += 1
            if result['rows'] == 0:
                empty_count += 1
                status = "0row"
                print(f"{i+1:>3} | {status:>6} | {result['rows']:>6} | {q['sql'][:60]}")
            else:
                status = "OK"
                print(f"{i+1:>3} | {status:>6} | {result['rows']:>6} | {q['sql'][:60]}")
        else:
            invalid_count += 1
            status = "ERR"
            print(f"{i+1:>3} | {status:>6} | {'---':>6} | {q['sql'][:60]}")
            print(f"    |        |        | ERROR: {result['error'][:70]}")

    # ==================== SUMMARY ====================
    print(f"\n{'='*80}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total Queries:    {len(queries)}")
    print(f"Valid:          {valid_count}/{len(queries)} ({valid_count/len(queries)*100:.1f}%)")
    print(f"Invalid:        {invalid_count}/{len(queries)} ({invalid_count/len(queries)*100:.1f}%)")
    print(f"Empty (0 rows): {empty_count}/{len(queries)} ({empty_count/len(queries)*100:.1f}%)")
    print(f"{'='*80}")

    # Detail invalid queries
    if invalid_count > 0:
        print(f"\n{'='*80}")
        print(f"DETAIL: INVALID QUERIES")
        print(f"{'='*80}")
        for r in results:
            if not r['is_valid']:
                print(f"\n  Line {r['line']} (Q#{results.index(r)+1}):")
                print(f"  SQL:   {r['sql']}")
                print(f"  ERROR: {r['error']}")

    # Detail empty queries
    if empty_count > 0:
        print(f"\n{'='*80}")
        print(f"DETAIL: EMPTY RESULT QUERIES (0 rows)")
        print(f"{'='*80}")
        for r in results:
            if r['is_valid'] and r['rows'] == 0:
                print(f"\n  Line {r['line']} (Q#{results.index(r)+1}):")
                print(f"  SQL: {r['sql']}")
                print(f"  → Query valid tapi return 0 rows. Cek apakah data ada di DB.")

    # Final verdict
    print(f"\n{'='*80}")
    if invalid_count == 0 and empty_count == 0:
        print(f"SEMUA QUERY VALID DAN MENGEMBALIKAN DATA")
    elif invalid_count == 0:
        print(f"SEMUA QUERY VALID, tapi {empty_count} query return 0 rows")
    else:
        print(f"ADA {invalid_count} QUERY YANG ERROR — PERLU DIPERBAIKI SEBELUM EVALUASI")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()