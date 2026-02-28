import os
import time
import csv
import json
import re
from datetime import datetime
from typing import List, Dict, Optional

from llama_index.core.indices.struct_store.sql_query import SQLTableRetrieverQueryEngine
from llama_index.core.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema
from llama_index.core import SQLDatabase, Settings, VectorStoreIndex
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from sqlalchemy import create_engine, MetaData, text as sa_text


# Configuration

EMBEDDING_MODEL = "mxbai-embed-large"
LLM_MODEL = "gemma3:4b"
OLLAMA_BASE_URL = "http://localhost:11434"

# MySQL Database Configuration
DB_USER = os.getenv('DB_USER', 'root')
DB_PASS = os.getenv('DB_PASS', '')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '3306')
DB_NAME = os.getenv('DB_NAME', 'example_endpoint')

# File paths
GOLD_FILE = "questions.sql"
OUTPUT_DIR = "output"

# Generation settings
USE_FEW_SHOT = True
INCLUDE_SAMPLE_ROWS = False

LLM_TEMPERATURE = 0.3
LLM_TOP_P = 0.95
LLM_NUM_PREDICT = 300
TOP_K_SCHEMA_FRAGMENTS = 3

# Domain tables mapping
DOMAIN_TABLES = {
    'transaction_enose': ['transaction_enose'],
    'transaction_object_detection': ['transaction_object_detection'],
    'transaction_enose + transaction_object_detection': ['transaction_enose', 'transaction_object_detection'],
}

ALL_TABLES = ['transaction_enose', 'transaction_object_detection']


# Few Shot

FEW_SHOT_EXAMPLES = """
### RULES:
- MySQL syntax ONLY (NEVER use strftime, date('now'), or SQLite functions)
- Use YEAR(), MONTH(), DATE(), TIME() for date operations
- Use JSON_EXTRACT() / JSON_UNQUOTE() for JSON columns
- transaction_object_detection.value is INTEGER (not JSON)
- transaction_enose.value and data_send are JSON
- Output single-line SQL only, no explanation

# EXAMPLE 1 - COUNT:
Question: Berapa jumlah record di tabel produk?
SQL: SELECT COUNT(*) AS total FROM produk

# EXAMPLE 2 - FILTER + ORDER:
Question: Tampilkan data sensor dari perangkat X001 urutkan terbaru
SQL: SELECT * FROM sensor_log WHERE device_id = 'X001' ORDER BY created_at DESC LIMIT 50

# EXAMPLE 3 - JSON EXTRACT:
Question: Ambil label kategori dari kolom JSON metadata
SQL: SELECT id, JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.category')) AS kategori FROM items

# EXAMPLE 4 - DATE FILTER (MySQL):
Question: Tampilkan log pada bulan Maret 2025
SQL: SELECT * FROM activity_log WHERE YEAR(timestamp_col) = 2025 AND MONTH(timestamp_col) = 3 ORDER BY timestamp_col DESC LIMIT 50

# EXAMPLE 5 - GROUP BY + AGGREGATE:
Question: Berapa total penjualan per kategori?
SQL: SELECT category, SUM(amount) AS total FROM sales GROUP BY category ORDER BY total DESC

# EXAMPLE 6 - DATE RANGE:
Question: Tampilkan data antara 1 Januari dan 31 Januari 2025
SQL: SELECT * FROM events WHERE DATE(event_time) BETWEEN '2025-01-01' AND '2025-01-31' ORDER BY event_time DESC LIMIT 50

# EXAMPLE 7 - SUBQUERY:
Question: Tampilkan data yang nilainya di atas rata-rata
SQL: SELECT * FROM measurements WHERE score > (SELECT AVG(score) FROM measurements) ORDER BY score DESC LIMIT 50

# EXAMPLE 8 - CROSS TABLE (UNION):
Question: Berapa total record dari tabel A dan tabel B?
SQL: SELECT 'table_a' AS source, COUNT(*) AS total FROM table_a UNION ALL SELECT 'table_b' AS source, COUNT(*) AS total FROM table_b
"""


def get_safe_filename(model_name):
    """Convert model name to safe filename"""
    return model_name.replace(":", "_").replace("/", "_").replace("\\", "_")


def parse_gold_file(gold_file_path):
    """Parse gold SQL file (questions.sql format for IMRON domain)"""
    questions = []

    with open(gold_file_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    for line in lines:
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
            except Exception as e:
                print(f"  [WARN] Failed to parse line: {line} -> {e}")
                continue

    return questions


def setup_llamaindex():
    """Setup LlamaIndex with embedding and LLM models"""

    print(f"\n{'='*80}")
    print("SETTING UP LLAMAINDEX TEXT-TO-SQL FOR IMRON DOMAIN (MySQL)")
    print(f"{'='*80}")

    embed_model = OllamaEmbedding(
        model_name=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=LLM_TEMPERATURE,
        top_p=LLM_TOP_P,
        request_timeout=120.0,
        context_window=4096,
        additional_kwargs={
            'num_predict': LLM_NUM_PREDICT,
            'num_ctx': 4096,
            'stop': ['\n\n', 'Question:', '# EXAMPLE']
        }
    )

    Settings.embed_model = embed_model
    Settings.llm = llm

    print(f"  ✓ Embedding Model: {EMBEDDING_MODEL}")
    print(f"  ✓ LLM Model: {LLM_MODEL}")
    print(f"  ✓ Temperature: {LLM_TEMPERATURE}")
    print(f"  ✓ Top-P: {LLM_TOP_P}")
    print(f"  ✓ Few-Shot Learning: {USE_FEW_SHOT} (8 generic examples)")
    print(f"  ✓ Sample Rows: {INCLUDE_SAMPLE_ROWS}")
    print(f"  ✓ Top-K Schema Fragments: {TOP_K_SCHEMA_FRAGMENTS}")
    print(f"  ✓ Database: MySQL ({DB_HOST}:{DB_PORT}/{DB_NAME})")
    print(f"{'='*80}\n")

    return embed_model, llm


def create_mysql_engine():
    """Create MySQL database engine"""
    connection_string = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(
        connection_string,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600
    )
    return engine


def create_sql_database_for_domain(engine, domain_id: str):
    """Create SQLDatabase object for a specific domain (table set)"""
    try:
        if domain_id in DOMAIN_TABLES:
            table_names = DOMAIN_TABLES[domain_id]
        else:
            table_names = ALL_TABLES

        sql_database = SQLDatabase(
            engine,
            include_tables=table_names,
            sample_rows_in_table_info=3 if INCLUDE_SAMPLE_ROWS else 0
        )

        return sql_database, table_names

    except Exception as e:
        print(f"  [ERROR] Failed to create SQL database for domain '{domain_id}': {e}")
        return None, []


def create_custom_text_to_sql_prompt() -> PromptTemplate:
    """Custom Text-to-SQL prompt — generic few-shot, no domain-specific leakage"""

    template = f"""You are a MySQL query generator. Given a database schema and a natural language question, generate the correct SQL query.

{FEW_SHOT_EXAMPLES}

Given the database schema below, write a MySQL query that answers the question.

Schema:
{{schema}}

Question: {{query_str}}

SQL:"""

    return PromptTemplate(template)


def create_table_schema_objs(sql_database, table_names: List[str]) -> List[SQLTableSchema]:
    """Create SQLTableSchema objects — minimal context, let LLM figure it out from schema"""

    # Hanya hint minimal tentang tipe kolom kritis, BUKAN jawaban lengkap
    IMRON_TABLE_CONTEXT = {
        'transaction_enose': (
            "E-nose IoT sensor data. "
            "Note: data_send is JSON array, value is JSON object. "
            "Use JSON_EXTRACT / JSON_UNQUOTE for JSON columns."
        ),
        'transaction_object_detection': (
            "Object detection results from edge devices. "
            "Note: value column is INTEGER (count of detected objects), not JSON."
        ),
    }

    table_schema_objs = []

    for table_name in table_names:
        context_str = IMRON_TABLE_CONTEXT.get(table_name, f"Table {table_name}")
        table_schema_objs.append(
            SQLTableSchema(
                table_name=table_name,
                context_str=context_str
            )
        )

    return table_schema_objs


def create_query_engine_with_retrieval(sql_database, table_names: List[str], embed_model):
    """Create query engine with ObjectIndex for top-K schema retrieval"""

    table_node_mapping = SQLTableNodeMapping(sql_database)
    table_schema_objs = create_table_schema_objs(sql_database, table_names)

    obj_index = ObjectIndex.from_objects(
        table_schema_objs,
        table_node_mapping,
        VectorStoreIndex,
    )

    text_to_sql_prompt = create_custom_text_to_sql_prompt()

    query_engine = SQLTableRetrieverQueryEngine(
        sql_database=sql_database,
        table_retriever=obj_index.as_retriever(similarity_top_k=TOP_K_SCHEMA_FRAGMENTS),
        text_to_sql_prompt=text_to_sql_prompt,
        synthesize_response=False,
        streaming=False,
    )

    return query_engine, obj_index


def create_direct_query_engine(sql_database, table_names: List[str]):
    """Create a direct NLSQLTableQueryEngine (tables already known)"""
    text_to_sql_prompt = create_custom_text_to_sql_prompt()

    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=table_names,
        text_to_sql_prompt=text_to_sql_prompt,
        synthesize_response=False,
        streaming=False,
    )

    return query_engine


def get_retrieved_tables(obj_index, question: str, top_k: int = TOP_K_SCHEMA_FRAGMENTS) -> List[str]:
    """Get the tables that would be retrieved for a given question"""
    retriever = obj_index.as_retriever(similarity_top_k=top_k)
    retrieved_nodes = retriever.retrieve(question)

    retrieved_tables = []
    for node in retrieved_nodes:
        if hasattr(node, 'node') and hasattr(node.node, 'metadata'):
            table_name = node.node.metadata.get('table_name', 'unknown')
        elif hasattr(node, 'metadata'):
            table_name = node.metadata.get('table_name', 'unknown')
        else:
            text = str(node)
            if 'Table ' in text:
                table_name = text.split('Table ')[1].split()[0].strip(':')
            else:
                table_name = 'unknown'
        retrieved_tables.append(table_name)

    return retrieved_tables


def fix_sqlite_to_mysql(sql: str) -> str:
    """Post-process: fix any SQLite syntax leaks to MySQL syntax"""
    if not sql:
        return sql

    sql = re.sub(r"strftime\s*\(\s*'%Y'\s*,\s*(\w+(?:\.\w+)?)\s*\)", r"YEAR(\1)", sql)
    sql = re.sub(r"strftime\s*\(\s*'%m'\s*,\s*(\w+(?:\.\w+)?)\s*\)", r"MONTH(\1)", sql)
    sql = re.sub(r"strftime\s*\(\s*'%d'\s*,\s*(\w+(?:\.\w+)?)\s*\)", r"DAY(\1)", sql)
    sql = re.sub(r"strftime\s*\(\s*'%H'\s*,\s*(\w+(?:\.\w+)?)\s*\)", r"HOUR(\1)", sql)
    sql = re.sub(r"strftime\s*\(\s*'%Y-%m'\s*,\s*(\w+(?:\.\w+)?)\s*\)", r"DATE_FORMAT(\1, '%Y-%m')", sql)
    sql = re.sub(r"strftime\s*\(\s*'%Y-%m-%d'\s*,\s*(\w+(?:\.\w+)?)\s*\)", r"DATE(\1)", sql)
    sql = re.sub(r"date\s*\(\s*'now'\s*\)", "CURDATE()", sql, flags=re.IGNORECASE)
    sql = re.sub(r"datetime\s*\(\s*'now'\s*\)", "NOW()", sql, flags=re.IGNORECASE)

    if '||' in sql and 'SELECT' in sql.upper():
        sql = re.sub(r"(\S+)\s*\|\|\s*(\S+)", r"CONCAT(\1, \2)", sql)

    return sql


def extract_sql_from_response(response) -> str:
    """Extract and normalize SQL query from LlamaIndex response"""

    if hasattr(response, 'metadata') and 'sql_query' in response.metadata:
        raw_sql = response.metadata['sql_query'].strip()
    else:
        raw_sql = str(response)

    sql = raw_sql.strip()
    sql = sql.replace('```sql', '').replace('```', '').strip()

    lines = []
    for line in sql.split('\n'):
        line = line.strip()
        if not line or line.startswith('--') or line.startswith('#'):
            continue
        if '--' in line:
            line = line.split('--')[0].strip()
        if line:
            lines.append(line)

    sql = ' '.join(lines)
    sql = sql.rstrip(';').strip()
    sql = fix_sqlite_to_mysql(sql)

    sql = re.sub(r'\s+', ' ', sql)
    sql = re.sub(r'\s*,\s*', ', ', sql)
    sql = re.sub(r'\(\s+', '(', sql)
    sql = re.sub(r'\s+\)', ')', sql)
    sql = re.sub(r'\s+', ' ', sql).strip()

    if len(sql) < 10:
        return "SELECT 1"

    sql_lower = sql.lower()
    if not any(kw in sql_lower for kw in ['select', 'insert', 'update', 'delete']):
        return "SELECT 1"

    return sql


def validate_prediction_file(predictions_file):
    """Validate predictions file"""

    with open(predictions_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    valid_count = 0
    invalid_count = 0
    warning_count = 0

    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        parts = line.split('\t')
        if len(parts) != 2:
            invalid_count += 1
        else:
            sql, db_id = parts
            if not sql or not db_id:
                invalid_count += 1
            elif sql.lower() == "select 1":
                warning_count += 1
                valid_count += 1
            else:
                valid_count += 1

    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"[OK] Valid: {valid_count}")
    print(f"[!] Warnings (SELECT 1): {warning_count}")
    print(f"[X] Invalid: {invalid_count}")
    print(f"{'='*60}\n")

    return valid_count, invalid_count


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_llm_name = get_safe_filename(LLM_MODEL)

    predictions_file = os.path.join(OUTPUT_DIR, f"prediksi_imron_topk{TOP_K_SCHEMA_FRAGMENTS}_{safe_llm_name}_{timestamp}.txt")
    metrics_file = os.path.join(OUTPUT_DIR, f"metric_imron_topk{TOP_K_SCHEMA_FRAGMENTS}_{safe_llm_name}_{timestamp}.csv")

    print(f"\n{'='*80}")
    print(f"SQL GENERATION: LLAMAINDEX + IMRON DOMAIN (MySQL)")
    print(f"{'='*80}")
    print(f"Method: NLSQLTableQueryEngine / SQLTableRetrieverQueryEngine")
    print(f"LLM Model: {LLM_MODEL}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"Database: MySQL ({DB_HOST}:{DB_PORT}/{DB_NAME})")
    print(f"Tables: {', '.join(ALL_TABLES)}")
    print(f"Top-K Schema Retrieval: {TOP_K_SCHEMA_FRAGMENTS} tables per query")
    print(f"Few-Shot Examples: 8 generic patterns (no domain leakage)")
    print(f"Temperature: {LLM_TEMPERATURE}")
    print(f"\nOutput files:")
    print(f"  1. {predictions_file}")
    print(f"  2. {metrics_file}")
    print(f"{'='*80}\n")

    # Setup LlamaIndex
    embed_model, llm = setup_llamaindex()

    # Parse questions
    if not os.path.exists(GOLD_FILE):
        print(f"ERROR: {GOLD_FILE} not found!")
        return

    questions = parse_gold_file(GOLD_FILE)
    print(f"Loaded {len(questions)} questions from {GOLD_FILE}\n")

    # Create MySQL engine
    print(f"Connecting to MySQL: {DB_HOST}:{DB_PORT}/{DB_NAME}...")
    try:
        mysql_engine = create_mysql_engine()
        with mysql_engine.connect() as conn:
            conn.execute(sa_text("SELECT 1"))
        print(f"  ✓ MySQL connection successful\n")
    except Exception as e:
        print(f"  [ERROR] MySQL connection failed: {e}")
        print(f"  Please check DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME settings.")
        return

    # Build query engines per domain (cache)
    metrics_data = []
    predictions = []

    print(f"{'='*80}")
    print(f"STARTING SQL GENERATION FOR IMRON DOMAIN")
    print(f"{'='*80}\n")

    total_start_time = time.time()
    error_count = 0
    fallback_count = 0

    domain_cache = {}

    for i, q in enumerate(questions):
        q_id = q['id']
        question = q['question']
        db_id = q['db_id']

        print(f"[{i+1}/{len(questions)}] Q{q_id} ({db_id})")
        print(f"  Question: {question[:80]}...")

        start_time = time.time()
        retrieved_tables_str = 'N/A'

        try:
            if db_id not in domain_cache:
                print(f"  [INFO] Setting up domain: {db_id}")
                sql_database, table_names = create_sql_database_for_domain(mysql_engine, db_id)

                if sql_database is None:
                    raise Exception(f"Failed to create SQL database for domain '{db_id}'")

                print(f"  [INFO] Tables: {', '.join(table_names)}")

                if len(table_names) <= 2:
                    print(f"  [INFO] Using direct NLSQLTableQueryEngine (tables known)")
                    query_engine = create_direct_query_engine(sql_database, table_names)
                    obj_index = None
                else:
                    print(f"  [INFO] Building ObjectIndex for top-{TOP_K_SCHEMA_FRAGMENTS} schema retrieval...")
                    query_engine, obj_index = create_query_engine_with_retrieval(
                        sql_database, table_names, embed_model
                    )

                domain_cache[db_id] = {
                    'sql_database': sql_database,
                    'query_engine': query_engine,
                    'obj_index': obj_index,
                    'tables': table_names
                }
            else:
                query_engine = domain_cache[db_id]['query_engine']
                obj_index = domain_cache[db_id]['obj_index']
                table_names = domain_cache[db_id]['tables']

            if obj_index is not None:
                try:
                    retrieved_tables = get_retrieved_tables(obj_index, question, TOP_K_SCHEMA_FRAGMENTS)
                    actual_retrieved = min(len(retrieved_tables), TOP_K_SCHEMA_FRAGMENTS)
                    retrieved_tables_str = ', '.join(retrieved_tables[:TOP_K_SCHEMA_FRAGMENTS])
                    print(f"  [RETR] Top-{actual_retrieved} tables retrieved: {retrieved_tables_str}")
                except Exception as e:
                    print(f"  [WARN] Could not log retrieved tables: {e}")
            else:
                retrieved_tables_str = ', '.join(table_names)
                print(f"  [DIRECT] Using tables: {retrieved_tables_str}")

            print(f"  [PRCS] Generating SQL...")
            response = query_engine.query(question)
            pred_sql = extract_sql_from_response(response)

            if not pred_sql or len(pred_sql) < 10:
                print(f"  [WARNING] Empty/short SQL, using fallback")
                pred_sql = "SELECT 1"
                fallback_count += 1
            elif not any(kw in pred_sql.lower() for kw in ['select', 'insert', 'update', 'delete']):
                print(f"  [WARNING] No SQL keyword, using fallback")
                pred_sql = "SELECT 1"
                fallback_count += 1

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            pred_sql = "SELECT 1"
            error_count += 1

        end_time = time.time()
        latency = end_time - start_time

        predictions.append(f"{pred_sql}\t{db_id}")

        metrics_data.append({
            'question_id': q_id,
            'db_id': db_id,
            'question': question,
            'predicted_sql': pred_sql,
            'retrieved_tables': retrieved_tables_str,
            'latency_seconds': round(latency, 4),
            'llm_model': LLM_MODEL,
            'embedding_model': EMBEDDING_MODEL,
            'top_k': TOP_K_SCHEMA_FRAGMENTS,
            'method': 'IMRON_MySQL_NLSQLTableQueryEngine',
            'timestamp': datetime.now().isoformat()
        })

        display_sql = pred_sql[:80] + "..." if len(pred_sql) > 83 else pred_sql
        print(f"  [OK] {latency:.2f}s: {display_sql}\n")

    total_time = time.time() - total_start_time
    avg_latency = total_time / len(questions) if questions else 0

    # ==================== LATENCY SUMMARY ====================
    all_latencies = [m['latency_seconds'] for m in metrics_data]
    successful_latencies = [m['latency_seconds'] for m in metrics_data if m['predicted_sql'] != 'SELECT 1']
    fallback_latencies = [m['latency_seconds'] for m in metrics_data if m['predicted_sql'] == 'SELECT 1']

    def calc_avg(latencies):
        return sum(latencies) / len(latencies) if latencies else 0

    def calc_median(latencies):
        if not latencies:
            return 0
        s = sorted(latencies)
        return s[len(s) // 2]

    avg_all = calc_avg(all_latencies)
    median_all = calc_median(all_latencies)
    min_all = min(all_latencies) if all_latencies else 0
    max_all = max(all_latencies) if all_latencies else 0

    avg_success = calc_avg(successful_latencies)
    avg_fallback = calc_avg(fallback_latencies)

    # Per-domain avg latency
    domain_latencies = {}
    for m in metrics_data:
        db = m['db_id']
        if db not in domain_latencies:
            domain_latencies[db] = []
        domain_latencies[db].append(m['latency_seconds'])

    # ==================== SAVE RESULTS ====================

    print(f"{'='*60}")
    print(f"SAVING RESULTS")
    print(f"{'='*60}")

    with open(predictions_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(pred + '\n')
    print(f"  [OK] Saved predictions: {predictions_file}")

    with open(metrics_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['question_id', 'db_id', 'question', 'predicted_sql',
                      'retrieved_tables', 'latency_seconds', 'llm_model',
                      'embedding_model', 'top_k', 'method', 'timestamp']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_data)
    print(f"  [OK] Saved metrics: {metrics_file}\n")

    valid_count, invalid_count = validate_prediction_file(predictions_file)

    # ==================== SUMMARY ====================

    print(f"{'='*80}")
    print(f"GENERATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Domain: IMRON (MySQL)")
    print(f"Method: NLSQLTableQueryEngine (direct) / SQLTableRetrieverQueryEngine")
    print(f"LLM: {LLM_MODEL} (temp={LLM_TEMPERATURE})")
    print(f"Embedding: {EMBEDDING_MODEL}")
    print(f"Database: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    print(f"Tables: {', '.join(ALL_TABLES)}")
    print(f"Few-Shot Examples: 8 generic patterns (no domain leakage)")
    print(f"\nResults:")
    print(f"  Total: {len(questions)}")
    print(f"  Successful: {len(questions) - error_count}")
    print(f"  Errors: {error_count}")
    print(f"  Fallbacks (SELECT 1): {fallback_count}")
    print(f"  Valid: {valid_count}")
    print(f"  Invalid: {invalid_count}")
    print(f"  Success Rate: {((len(questions)-fallback_count)/len(questions)*100):.1f}%" if questions else "  N/A")
    print(f"\nLatency (per query):")
    print(f"  Avg (all):        {avg_all:.2f}s")
    print(f"  Avg (successful): {avg_success:.2f}s")
    if fallback_latencies:
        print(f"  Avg (fallback):   {avg_fallback:.2f}s")
    print(f"  Median:           {median_all:.2f}s")
    print(f"  Min:              {min_all:.2f}s")
    print(f"  Max:              {max_all:.2f}s")
    print(f"  Total Time:       {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"\nLatency per Domain:")
    for db, lats in sorted(domain_latencies.items()):
        print(f"  {db}: avg {calc_avg(lats):.2f}s ({len(lats)} queries)")
    print(f"\nOutput Files:")
    print(f"  1. {predictions_file}")
    print(f"  2. {metrics_file}")

    # ==================== DOMAIN BREAKDOWN ====================

    print(f"\n{'='*60}")
    print(f"DOMAIN BREAKDOWN")
    print(f"{'='*60}")
    domain_counts = {}
    for q in questions:
        db = q['db_id']
        domain_counts[db] = domain_counts.get(db, 0) + 1
    for db, count in domain_counts.items():
        print(f"  {db}: {count} questions")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()