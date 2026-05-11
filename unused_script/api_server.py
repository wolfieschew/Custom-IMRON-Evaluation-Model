from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_index.core import SQLDatabase, Settings
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.core.embeddings import BaseEmbedding
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
import mysql.connector
from mysql.connector import Error as MySQLError
import sys
import os
import re
import json
import time
import logging
from logging.handlers import RotatingFileHandler
from collections import defaultdict
from typing import List, Dict, Optional
from datetime import datetime
from decimal import Decimal
from sentence_transformers import SentenceTransformer
import ollama as ollama_client
from rag_system import RAGSystem

# ==================== CONFIGURATION ====================

log_file = 'api_chat.log'
max_bytes = 10 * 1024 * 1024
backup_count = 3

DB_USER = os.getenv('DB_USER', 'root')
DB_PASS = os.getenv('DB_PASS', '')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '3306')
DB_NAME = os.getenv('DB_NAME', 'example_endpoint')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'masimron:v5')

# ==================== LOGGING SETUP ====================

log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
file_handler.setFormatter(log_format)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_format)
console_handler.setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.propagate = False

# ==================== FLASK APP ====================

app = Flask(__name__)
CORS(app)

# ==================== RATE LIMITER ====================

class RateLimiter:
    def __init__(self, max_requests=20, window_seconds=60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests = defaultdict(list)

    def is_allowed(self, user_id):
        now = time.time()
        self.requests[user_id] = [r for r in self.requests[user_id] if r > now - self.window]
        if len(self.requests[user_id]) >= self.max_requests:
            return False
        self.requests[user_id].append(now)
        return True

rate_limiter = RateLimiter(max_requests=20, window_seconds=60)

# ==================== DATABASE SETUP ====================

db_engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600
)

metadata = MetaData()
metadata.reflect(bind=db_engine)

# Database object per-domain (schema dump lebih kecil per engine)
sql_database_enose = SQLDatabase(
    db_engine,
    include_tables=["transaction_enose"]
)

sql_database_objdet = SQLDatabase(
    db_engine,
    include_tables=["transaction_object_detection"]
)

sql_database_general = SQLDatabase(
    db_engine,
    include_tables=["device", "user", "device_user_mapping", "user_detail"]
)

# ==================== LLM & EMBEDDINGS SETUP ====================

llm = Ollama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    request_timeout=60.0,
    num_ctx=4096
)

llm_agent = ollama_client.Client(host=OLLAMA_BASE_URL)
embedder = SentenceTransformer('BAAI/bge-m3')

class CustomEmbedding(BaseEmbedding):
    def __init__(self, model):
        super().__init__()
        self._model = model

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._model.encode(query).tolist()

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._model.encode(text).tolist()

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

Settings.llm = llm
Settings.embed_model = CustomEmbedding(embedder)

# ==================== CUSTOM PROMPT (OVERRIDE LLAMAINDEX DEFAULT) ====================

CUSTOM_TEXT_TO_SQL_PROMPT = PromptTemplate(
    "Given an input question, create a syntactically correct {dialect} query to run. "
    "IMPORTANT: This is MySQL/MariaDB. NEVER use SQLite functions like strftime(). "
    "Use YEAR(), MONTH(), DATE(), CURDATE() for date operations.\n\n"
    "Only use the following tables:\n"
    "{schema}\n\n"
    "RULES:\n"
    "1. ONLY output a single SELECT statement\n"
    "2. Use MySQL syntax ONLY — NO strftime, NO date('now')\n"
    "3. Do NOT explain, do NOT wrap in markdown\n"
    "4. Always use LIMIT to avoid returning too many rows\n\n"
    "Question: {query_str}\n"
    "SQLQuery: "
)

CUSTOM_RESPONSE_SYNTHESIS_PROMPT = PromptTemplate(
    "Given an input question, synthesize a response from the query results.\n"
    "Respond in Bahasa Indonesia, be concise (max 3 sentences).\n\n"
    "Query: {query_str}\n"
    "SQL Query: {sql_query}\n"
    "SQL Response: {sql_response_str}\n"
    "Response: "
)

# ==================== QUERY ENGINES (DOMAIN-SPECIFIC) ====================

def build_engine(sql_db, tables):
    """Build NLSQLTableQueryEngine dengan custom MySQL prompt."""
    return NLSQLTableQueryEngine(
        sql_database=sql_db,
        synthesize_response=False,
        tables=tables,
        text_to_sql_prompt=CUSTOM_TEXT_TO_SQL_PROMPT,
    )

engine_enose = build_engine(sql_database_enose, ["transaction_enose"])
engine_objdet = build_engine(sql_database_objdet, ["transaction_object_detection"])
engine_general = build_engine(sql_database_general, ["device", "user", "device_user_mapping", "user_detail"])

# ==================== RAG SYSTEM SETUP ====================

rag_system = None

def initialize_rag():
    global rag_system
    try:
        html_path = os.getenv('HTML_PATH', '')
        logger.info("Menginisialisasi RAG System...")
        rag_system = RAGSystem(html_path)
        if rag_system.initialize():
            logger.info("[OK] RAG System berhasil diinisialisasi")
            if rag_system.kb_retriever:
                logger.info(f"[KB] ChromaDB siap: {rag_system.kb_retriever.get_count()} dokumen")
            return True
        else:
            logger.error("[ERROR] Gagal menginisialisasi RAG System")
            return False
    except Exception as e:
        logger.error(f"[ERROR] Error inisialisasi RAG: {e}")
        import traceback
        traceback.print_exc()
        return False

# ==================== FEW-SHOT EXAMPLES (COMPACT) ====================

_FS_SCHEMA = """### MySQL ONLY — JANGAN SQLite!
-- transaction_enose: device_id, date_time(DATETIME), type(VARCHAR), data_send(JSON), value(JSON)
--   data_send: [{"MQ3":float,"TGS822":float,"TGS2602":float,"MQ5":float,"MQ138":float,"TGS2620":float}]
--   value: {"0":{"Score":[float],"Class":"str","Multiclass":["str"]}}
-- transaction_object_detection: detection_time(DATETIME), type(VARCHAR), value(INT), device_id(VARCHAR)
-- DATE: YEAR(), MONTH(), DATE(), CURDATE() — BUKAN strftime!
"""

_FS_SCORE = """### SCORE (transaction_enose)
Q: "score terbaik"
SQL: SELECT device_id, date_time, type, CAST(REPLACE(REPLACE(SUBSTRING_INDEX(SUBSTRING_INDEX(value,'"Score":[',-1),']',1),'[',''),']','') AS DECIMAL(10,2)) AS actual_score, JSON_UNQUOTE(JSON_EXTRACT(value,'$.0.Class')) AS class_label FROM transaction_enose WHERE CAST(REPLACE(REPLACE(SUBSTRING_INDEX(SUBSTRING_INDEX(value,'"Score":[',-1),']',1),'[',''),']','') AS DECIMAL(10,2)) > 0 ORDER BY actual_score DESC LIMIT 1

Q: "rata-rata score"
SQL: SELECT AVG(CAST(REPLACE(REPLACE(SUBSTRING_INDEX(SUBSTRING_INDEX(value,'"Score":[',-1),']',1),'[',''),']','') AS DECIMAL(10,2))) AS avg_score FROM transaction_enose

Q: "score terburuk"
SQL: SELECT device_id, date_time, type, CAST(REPLACE(REPLACE(SUBSTRING_INDEX(SUBSTRING_INDEX(value,'"Score":[',-1),']',1),'[',''),']','') AS DECIMAL(10,2)) AS actual_score FROM transaction_enose WHERE CAST(REPLACE(REPLACE(SUBSTRING_INDEX(SUBSTRING_INDEX(value,'"Score":[',-1),']',1),'[',''),']','') AS DECIMAL(10,2)) > 0 ORDER BY actual_score ASC LIMIT 1
"""

_FS_SENSOR = """### SENSOR E-NOSE
Q: "nilai MQ3"
SQL: SELECT device_id, date_time, type, CAST(JSON_UNQUOTE(JSON_EXTRACT(data_send,'$[0].MQ3')) AS DECIMAL(10,4)) AS MQ3 FROM transaction_enose ORDER BY MQ3 DESC LIMIT 5

Q: "rata-rata TGS822"
SQL: SELECT AVG(CAST(JSON_UNQUOTE(JSON_EXTRACT(data_send,'$[0].TGS822')) AS DECIMAL(10,4))) AS avg_TGS822 FROM transaction_enose
"""

_FS_OBJECT_DETECTION = """### OBJECT DETECTION (value=INTEGER, BUKAN JSON!)
Q: "total mobil" → SELECT SUM(value) AS total_car FROM transaction_object_detection WHERE type='car'
Q: "jenis terbanyak" → SELECT type, SUM(value) AS total FROM transaction_object_detection GROUP BY type ORDER BY total DESC LIMIT 1
Q: "deteksi hari ini" → SELECT type, SUM(value) AS total FROM transaction_object_detection WHERE DATE(detection_time)=CURDATE() GROUP BY type
Q: "total dosen" → SELECT SUM(value) FROM transaction_object_detection WHERE type='dosen'
Q: "total per jenis" → SELECT type, SUM(value) AS total FROM transaction_object_detection GROUP BY type ORDER BY total DESC
"""

_FS_FILTER = """### FILTER
Q: "klasifikasi Baik" → SELECT COUNT(*) FROM transaction_enose WHERE value LIKE '%"Class":"Baik"%'
Q: "cacat mutu" → SELECT COUNT(*) FROM transaction_enose WHERE value LIKE '%"Class":"Cacat Mutu"%'
Q: "transaksi greentea" → SELECT * FROM transaction_enose WHERE type='greentea' LIMIT 10
Q: "dosen terdeteksi" → SELECT SUM(value) FROM transaction_object_detection WHERE type='dosen'
"""

_FS_EXPLANATION = """### EXPLANATION (ambil SEMUA sensor untuk analisis)
Q: "kenapa score 51.9"
SQL: SELECT device_id, date_time, type, CAST(JSON_UNQUOTE(JSON_EXTRACT(data_send,'$[0].MQ3')) AS DECIMAL(10,4)) AS MQ3, CAST(JSON_UNQUOTE(JSON_EXTRACT(data_send,'$[0].TGS822')) AS DECIMAL(10,4)) AS TGS822, CAST(JSON_UNQUOTE(JSON_EXTRACT(data_send,'$[0].TGS2602')) AS DECIMAL(10,4)) AS TGS2602, CAST(JSON_UNQUOTE(JSON_EXTRACT(data_send,'$[0].MQ5')) AS DECIMAL(10,4)) AS MQ5, CAST(JSON_UNQUOTE(JSON_EXTRACT(data_send,'$[0].MQ138')) AS DECIMAL(10,4)) AS MQ138, CAST(JSON_UNQUOTE(JSON_EXTRACT(data_send,'$[0].TGS2620')) AS DECIMAL(10,4)) AS TGS2620, CAST(REPLACE(REPLACE(SUBSTRING_INDEX(SUBSTRING_INDEX(value,'"Score":[',-1),']',1),'[',''),']','') AS DECIMAL(10,2)) AS actual_score FROM transaction_enose WHERE CAST(REPLACE(REPLACE(SUBSTRING_INDEX(SUBSTRING_INDEX(value,'"Score":[',-1),']',1),'[',''),']','') AS DECIMAL(10,2))=51.9 ORDER BY date_time DESC LIMIT 1
"""

_FS_SIMPLE = """### BASIC
Q: "total user" → SELECT COUNT(*) FROM user
Q: "semua device" → SELECT * FROM device
Q: "total data enose" → SELECT COUNT(*) FROM transaction_enose
"""

# ==================== HELPER FUNCTIONS ====================

def get_greeting() -> str:
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Selamat pagi"
    elif 12 <= hour < 15:
        return "Selamat siang"
    elif 15 <= hour < 18:
        return "Selamat sore"
    else:
        return "Selamat malam"

def is_object_detection_query(query: str) -> bool:
    od_keywords = ['objek', 'object', 'deteksi', 'detection', 'person', 'car',
                   'confidence', 'gambar', 'image', 'mahasiswa', 'dosen', 'simon-ai',
                   'edge', 'edge_1', 'terdeteksi', 'mobil', 'bus', 'sepeda', 'orang', 'people', 'bicycle']
    return any(kw in query.lower() for kw in od_keywords)

def is_score_query(query: str) -> bool:
    score_keywords = [
        'score', 'nilai', 'skor',
        'terbaik', 'tertinggi', 'terbesar',
        'terburuk', 'terendah', 'terkecil',
        'terbawah', 'paling rendah', 'paling buruk',
        'rata-rata', 'average'
    ]
    return any(kw in query.lower() for kw in score_keywords)

def is_sensor_query(query: str) -> bool:
    sensor_keywords = ['mq3', 'mq5', 'mq138', 'tgs822', 'tgs2602', 'tgs2620',
                       'sensor', 'nilai sensor', 'data sensor', 'pembacaan sensor']
    return any(kw in query.lower() for kw in sensor_keywords)

def is_filter_query(question: str) -> bool:
    filter_keywords = [
        'baik', 'cacat mutu', 'cacat', 'mutu',
        'multiclass', 'grade', 'kelas', 'klasifikasi',
        'kurang dari', 'lebih dari', 'antara', 'between',
        'greentea', 'kopi'
    ]
    return any(kw in question.lower() for kw in filter_keywords)

def is_explanation_query(query: str) -> bool:
    explain_keywords = [
        'kenapa', 'mengapa', 'alasan', 'why', 'sebab',
        'bagaimana bisa', 'apa yang membuat', 'jelaskan kenapa',
        'faktor', 'penyebab', 'karena apa'
    ]
    return any(kw in query.lower() for kw in explain_keywords)

def is_database_query(query: str) -> bool:
    query_lower = query.lower()
    db_patterns = [
        r'\bberapa\s+(total|jumlah)',
        r'\btotal\s+\w+',
        r'\btampilkan\s+',
        r'\bsebutkan\s+\d+',
        r'\brata-rata\s+score',
        r'\bscore\s+(terbaik|tertinggi|terendah)',
        r'\bdata\s+(transaksi|sensor|device|user)',
        r'\bjumlah\s+(user|device|transaksi)',
    ]
    return any(re.search(pattern, query_lower) for pattern in db_patterns)

def route_query_engine(question: str):
    """Pilih engine yang tepat berdasarkan question — mengurangi schema dump."""
    if is_object_detection_query(question):
        logger.info("[Router] -> Object Detection engine")
        return engine_objdet, "transaction_object_detection"
    elif is_sensor_query(question) or is_score_query(question) or is_explanation_query(question):
        logger.info("[Router] -> E-Nose engine")
        return engine_enose, "transaction_enose"
    elif is_filter_query(question):
        q = question.lower()
        if any(kw in q for kw in ['dosen', 'mahasiswa', 'mobil', 'car', 'bus', 'orang', 'people']):
            logger.info("[Router] → Object Detection engine (filter)")
            return engine_objdet, "transaction_object_detection"
        logger.info("[Router] → E-Nose engine (filter)")
        return engine_enose, "transaction_enose"
    else:
        logger.info("[Router] -> General engine")
        return engine_general, "general"

def get_few_shot(question: str) -> str:
    """Pilih HANYA few-shot yang relevan — hemat token."""
    parts = [_FS_SCHEMA]

    if is_explanation_query(question):
        parts.append(_FS_EXPLANATION)
    elif is_object_detection_query(question):
        parts.append(_FS_OBJECT_DETECTION)
    elif is_filter_query(question):
        parts.append(_FS_FILTER)
    elif is_sensor_query(question):
        parts.append(_FS_SENSOR)
    elif is_score_query(question):
        parts.append(_FS_SCORE)
    else:
        parts.append(_FS_SIMPLE)

    return "\n".join(parts)

def get_mysql_date_filter(question):
    """Auto-generate MySQL date filter dari natural language."""
    months = {
        'januari':1, 'februari':2, 'maret':3, 'april':4, 'mei':5, 'juni':6,
        'juli':7, 'agustus':8, 'september':9, 'oktober':10, 'november':11, 'desember':12,
        'jan':1, 'feb':2, 'mar':3, 'apr':4, 'jun':6, 'jul':7, 'agu':8, 'sep':9, 'okt':10, 'nov':11, 'des':12
    }

    q_lower = question.lower()

    # Pattern 1: "antara DD bulan YYYY dan DD bulan YYYY"
    range_pattern = r'antara\s+(\d+)\s*([a-z]+)\s*(\d{4})?\s*(?:dan|sampai|hingga)\s+(\d+)\s*([a-z]+)\s*(\d{4})?'
    range_match = re.search(range_pattern, q_lower)

    if range_match:
        day1, month1_str, year1, day2, month2_str, year2 = range_match.groups()
        year1 = int(year1 or 2025)
        year2 = int(year2 or 2025)
        month1 = months.get(month1_str)
        month2 = months.get(month2_str)

        if month1 and month2:
            start_date = f"'{year1}-{month1:02d}-{int(day1):02d} 00:00:00'"
            end_date = f"'{year2}-{month2:02d}-{int(day2):02d} 23:59:59'"
            return f"date_time BETWEEN {start_date} AND {end_date}"

    # Pattern 2: bulan + tahun
    year_match = re.search(r'(\d{4})', q_lower)
    year = int(year_match.group(1)) if year_match else 2025

    for month_name, month_num in months.items():
        if month_name in q_lower:
            return f"YEAR(date_time)={year} AND MONTH(date_time)={month_num}"

    return None

def fix_sqlite_to_mysql(sql: str) -> str:
    """Post-process: auto-fix SQLite syntax yang lolos ke MySQL."""
    if not sql:
        return sql

    original = sql

    # strftime('%Y-%m-%d', col) → DATE(col)
    sql = re.sub(
        r"strftime\s*\(\s*'%Y-%m-%d'\s*,\s*(\w+)\s*\)",
        r"DATE(\1)", sql, flags=re.IGNORECASE
    )
    # strftime('%Y', col) → YEAR(col)
    sql = re.sub(
        r"strftime\s*\(\s*'%Y'\s*,\s*(\w+)\s*\)",
        r"YEAR(\1)", sql, flags=re.IGNORECASE
    )
    # strftime('%m', col) → MONTH(col)
    sql = re.sub(
        r"strftime\s*\(\s*'%m'\s*,\s*(\w+)\s*\)",
        r"MONTH(\1)", sql, flags=re.IGNORECASE
    )
    # strftime('%d', col) → DAY(col)
    sql = re.sub(
        r"strftime\s*\(\s*'%d'\s*,\s*(\w+)\s*\)",
        r"DAY(\1)", sql, flags=re.IGNORECASE
    )
    # strftime('%H', col) → HOUR(col)
    sql = re.sub(
        r"strftime\s*\(\s*'%H'\s*,\s*(\w+)\s*\)",
        r"HOUR(\1)", sql, flags=re.IGNORECASE
    )
    # Catch-all: any remaining strftime with 2 args
    sql = re.sub(
        r"strftime\s*\(\s*'[^']*'\s*,\s*(\w+)\s*\)",
        r"DATE(\1)", sql, flags=re.IGNORECASE
    )
    # date('now') → CURDATE()
    sql = re.sub(
        r"date\s*\(\s*'now'\s*\)",
        "CURDATE()", sql, flags=re.IGNORECASE
    )
    # datetime('now') → NOW()
    sql = re.sub(
        r"datetime\s*\(\s*'now'\s*\)",
        "NOW()", sql, flags=re.IGNORECASE
    )
    # date('now', '-7 days') → DATE_SUB(CURDATE(), INTERVAL 7 DAY)
    sql = re.sub(
        r"date\s*\(\s*'now'\s*,\s*'-(\d+)\s*days?'\s*\)",
        r"DATE_SUB(CURDATE(), INTERVAL \1 DAY)", sql, flags=re.IGNORECASE
    )

    if sql != original:
        logger.warning(f"[SQL Fix] SQLite→MySQL: {original}")
        logger.warning(f"[SQL Fix] Fixed to    : {sql}")

    return sql

def extract_sql(text_input: str) -> Optional[str]:
    if not text_input:
        return None

    text_clean = re.sub(r'```sql\s*', '', text_input, flags=re.IGNORECASE)
    text_clean = re.sub(r'```\s*', '', text_clean)
    text_clean = re.sub(r'^sql\s*:\s*', '', text_clean, flags=re.IGNORECASE)
    text_clean = text_clean.strip().strip('"').strip("'")

    # Langsung SQL multiline
    if re.match(r'^\s*(SELECT|INSERT|UPDATE|DELETE)', text_clean, re.IGNORECASE):
        return ' '.join(text_clean.split()).rstrip(';')

    # Fallback: cari SQL di dalam teks
    text_oneline = ' '.join(text_clean.split())
    sql_pattern = r'(SELECT\s+.*?)(?:;|$)'
    matches = re.findall(sql_pattern, text_oneline, re.IGNORECASE | re.DOTALL)
    if matches:
        return matches[0].strip().rstrip(';')

    return None

def is_simple_result(data_result: list) -> bool:
    if not data_result or len(data_result) != 1:
        return False
    row = data_result[0]
    if isinstance(row, dict):
        keys = list(row.keys())
        if len(keys) == 1 and ('count' in str(keys[0]).lower()):
            return True
    elif isinstance(row, tuple):
        if len(row) == 1 and isinstance(row[0], int):
            return True
    return False

def format_simple_result(data_result: list, question: str) -> str:
    if len(data_result) == 1:
        row = data_result[0]
        if isinstance(row, dict):
            count_value = list(row.values())[0]
        elif isinstance(row, tuple):
            count_value = row[0]
        else:
            count_value = str(row)

        prompt = f"""Kamu adalah asisten AI untuk sistem IMRON yang ramah dan informatif.
Awali jawaban dengan salam "{get_greeting()}" yang singkat.
Jawab pertanyaan berikut secara natural dalam bahasa Indonesia.
Berikan jawaban yang SINGKAT, maksimal 2-3 kalimat saja.
Sebutkan angka/hasil dengan jelas, tambahkan 1 kalimat konteks singkat yang relevan.
JANGAN bertele-tele, JANGAN tawarkan bantuan lebih lanjut.

Pertanyaan: {question}
Jawaban dari database: {count_value}

Jawab singkat dan natural:"""

        try:
            response = llm_agent.chat(
                model=OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': prompt}]
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"format_simple_result LLM failed: {e}")
            return f"Hasilnya adalah {count_value}"

    return f"Ditemukan {len(data_result)} data"

def format_row_for_llm(row) -> str:
    SENSOR_KEYS = ['MQ3', 'TGS822', 'TGS2602', 'MQ5', 'MQ138', 'TGS2620']

    def format_value(val):
        if isinstance(val, datetime):
            return val.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(val, Decimal):
            return f"{float(val):.4f}".rstrip('0').rstrip('.')
        elif isinstance(val, (bytes, bytearray)):
            try:
                return val.decode('utf-8')
            except:
                return str(val)
        elif val is None:
            return "N/A"
        return str(val)

    def parse_data_send(raw: str) -> str:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and len(parsed) > 0:
                sensors = parsed[0]
                parts = [f"{k}: {sensors[k]}" for k in SENSOR_KEYS if k in sensors]
                return "Sensor [" + ", ".join(parts) + "]"
        except:
            pass
        return "(data_send tidak dapat diparsing)"

    def parse_value_json(raw: str) -> str:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                score = None
                class_name = None
                if '0' in parsed:
                    score = parsed['0'].get('Score', [None])[0]
                    class_name = parsed['0'].get('Class', 'N/A')
                elif 'Score' in parsed:
                    score = parsed['Score'][0] if isinstance(parsed['Score'], list) else parsed['Score']
                    class_name = parsed.get('Class', 'N/A')
                if score is not None:
                    return f"Score: {score}, Class: {class_name}"
        except:
            pass
        return None

    if isinstance(row, dict):
        formatted_parts = []
        for key, value in row.items():
            if key in ['user_key', 'embedding']:
                continue
            if key == 'data_send' and isinstance(value, str):
                formatted_parts.append(parse_data_send(value))
                continue
            if key == 'value' and isinstance(value, str):
                parsed_val = parse_value_json(value)
                if parsed_val:
                    formatted_parts.append(parsed_val)
                    continue
                if len(value) > 200:
                    continue
            if isinstance(value, str) and len(value) > 200:
                continue
            formatted_parts.append(f"{key}: {format_value(value)}")
        return " | ".join(formatted_parts)

    elif isinstance(row, tuple):
        formatted_values = []
        for val in row:
            if isinstance(val, str) and len(val) > 200:
                parsed_val = parse_value_json(val)
                if parsed_val:
                    formatted_values.append(parsed_val)
                    continue
                sensor_parsed = parse_data_send(val)
                if 'Sensor [' in sensor_parsed:
                    formatted_values.append(sensor_parsed)
                    continue
                formatted_values.append("(data terlalu panjang)")
                continue
            formatted_values.append(format_value(val))
        return " | ".join(formatted_values)

    return str(row)

def synthesize_answer(question: str, sql_query: str, data_result: list, table_used: str = 'unknown') -> str:
    if isinstance(data_result, list) and len(data_result) > 0:
        if len(data_result) == 1:
            result_summary = format_row_for_llm(data_result[0])
        else:
            result_summary = f"Total {len(data_result)} records found:\n\n"
            for idx, row in enumerate(data_result[:10], 1):
                result_summary += f"{idx}. {format_row_for_llm(row)}\n"
    else:
        result_summary = "Tidak ada data."

    prompt = f"""Kamu adalah asisten AI untuk sistem IMRON yang ramah dan informatif.
Awali jawaban dengan salam "{get_greeting()}" yang singkat.
Jawab pertanyaan berdasarkan HASIL QUERY DATABASE.
Gunakan bahasa Indonesia natural, RINGKAS (max 3 kalimat).
JANGAN tampilkan SQL, JANGAN bertele-tele.

KONTEKS TABEL: {table_used.upper()}
- TRANSACTION_OBJECT_DETECTION: type=nama objek (car/bus/people/mahasiswa/dosen), value=jumlah deteksi (integer)
- TRANSACTION_ENOSE: type=sampel (greentea), ada sensor MQ3/TGS822/score

PENTING - ANTI HALUSINASI:
- Jawab HANYA berdasarkan DATA di bawah
- Jika data 'mahasiswa: 3968' → katakan "mahasiswa (3968)"
- JANGAN tukar jadi 'greentea', 'car' kecuali ada di data itu sendiri

Keterangan field:
- actual_score: skor klasifikasi e-nose
- type: objek/sampel sesuai tabel di atas
- value: jumlah deteksi (OD) atau JSON score (e-nose)
- MQ3,TGS822,...: sensor gas e-nose

Pertanyaan: {question}

Data dari Database:
{result_summary}

Jawab ringkas dan natural:"""

    try:
        response = llm_agent.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    except Exception as e:
        logger.error(f"Synthesize LLM call failed: {str(e)}")
        return format_simple_result(data_result, question)

def classify_intent(question: str) -> str:
    prompt = f"""Klasifikasikan pertanyaan ke salah satu kategori:

- "database": QUERY DATA REAL-TIME dari database (butuh SELECT, COUNT, SUM)
  Contoh: "total mobil", "berapa orang terdeteksi", "score terbaik", "nilai MQ3"

- "explanation": minta alasan/analisis DARI DATA
  Contoh: "kenapa", "mengapa", "alasan", "jelaskan kenapa", "faktor apa"

- "knowledge": UMUM tentang sistem/fitur (tidak butuh data real-time)
  Contoh: "cara kerja sensor", "apa itu e-nose", "fitur sistem IMRON"

- "greeting": sapaan "halo", "hai", "selamat pagi"

Pertanyaan: {question}

Jawab HANYA satu kata (database/explanation/knowledge/greeting):"""

    try:
        response = llm_agent.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.1, 'num_predict': 10}
        )
        intent = response['message']['content'].strip().lower()
        intent = re.sub(r'[^a-z]', '', intent)
        if intent in ['database', 'explanation', 'knowledge', 'greeting']:
            logger.info(f"[Intent] Classified as: {intent}")
            return intent
        else:
            logger.warning(f"[Intent] Unexpected '{intent}', defaulting to 'database'")
            return 'database'
    except Exception as e:
        logger.error(f"[Intent] Classifier error: {e}, defaulting to 'database'")
        return 'database'

def preprocess_sensor_for_explanation(row: dict) -> str:
    SENSOR_RANGES = {
        'MQ3'    : (100, 300, 400),
        'TGS822' : (30,  80,  100),
        'TGS2602': (50,  100, 150),
        'MQ5'    : (100, 250, 300),
        'MQ138'  : (200, 450, 500),
        'TGS2620': (50,  120, 150),
    }

    lines = []
    for sensor, (low, mid, high) in SENSOR_RANGES.items():
        if sensor in row:
            try:
                val = float(row[sensor])
                if val > high:
                    status = "[TINGGI]"
                elif val < low:
                    status = "[RENDAH]"
                else:
                    status = "[NORMAL]"
                lines.append(f"  {sensor:<10}: {val:<10} -> {status}")
            except:
                pass

    return "\n".join(lines)

def synthesize_explanation(question: str, data_result: list, col_keys: list = None, kb_context: str = "") -> str:
    if not data_result:
        return "Tidak ada data yang dapat dijelaskan."

    def format_row(row):
        if isinstance(row, dict):
            return {k: v for k, v in row.items() if k not in ['user_key', 'embedding']}
        elif isinstance(row, tuple) and col_keys:
            return dict(zip(col_keys, [str(v) for v in row]))
        else:
            return {'data': format_row_for_llm(row)}

    formatted_rows = [format_row(r) for r in data_result[:3]]
    sensor_info = json.dumps(formatted_rows, indent=2, default=str)

    logger.info(f"[Explain] col_keys  : {col_keys}")
    logger.info(f"[Explain] formatted : {sensor_info}")

    sensor_analysis = preprocess_sensor_for_explanation(formatted_rows[0])
    if not sensor_analysis.strip():
        logger.warning("[Explain] sensor_analysis kosong! Data sensor tidak tersedia.")
        return (
            f"{get_greeting()}! Maaf, data sensor detail tidak tersedia untuk analisis ini. "
            f"Coba tanyakan: 'tampilkan nilai sensor dari score 51.9' terlebih dahulu."
        )
    logger.info(f"[Explain] sensor_analysis:\n{sensor_analysis}")

    kb_section = f"\nKonteks dari Knowledge Base:\n{kb_context}\n" if kb_context else ""

    prompt = f"""Kamu adalah analis sistem E-Nose IMRON yang ahli dalam interpretasi data sensor gas.
Awali jawaban dengan salam "{get_greeting()}" yang singkat.

Berikut adalah data hasil pengukuran sensor E-Nose:
{sensor_info}

Status tiap sensor berdasarkan rentang referensi:
{sensor_analysis}

Keterangan sensor:
- MQ3    : mendeteksi alkohol dan benzena
- TGS822 : mendeteksi pelarut organik
- TGS2602: mendeteksi VOC (Volatile Organic Compounds)
- MQ5    : mendeteksi LPG dan gas alam
- MQ138  : mendeteksi hidrokarbon organik
- TGS2620: mendeteksi alkohol dan pelarut
- actual_score: score kecocokan pola sensor terhadap kelas tertentu
{kb_section}
Pertanyaan: {question}

PENTING:
- Gunakan status TINGGI/NORMAL/RENDAH di atas — JANGAN ubah interpretasinya
- Analisis SEMUA sensor, jangan hanya fokus pada satu sensor
- Bandingkan sensor mana yang menonjol dan bagaimana kombinasinya membentuk klasifikasi

Jelaskan secara singkat (3-4 kalimat) dalam bahasa Indonesia:"""

    try:
        response = llm_agent.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        result = response['message']['content']
        logger.info(f"[Explain] LLM response: {result[:200]}")
        return result
    except Exception as e:
        logger.error(f"[Explain] LLM failed: {e}")
        return synthesize_answer(question, "", data_result)

# ==================== API ENDPOINTS ====================

@app.route('/api/query', methods=['POST'])
def query_single():
    user_ip = request.remote_addr
    if not rate_limiter.is_allowed(user_ip):
        return jsonify({'error': 'Rate limit exceeded. Mohon tunggu sebentar.', 'status': 'error'}), 429

    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'Parameter "question" diperlukan', 'status': 'error'}), 400

    question = data['question'].strip()
    if not question:
        return jsonify({'error': 'Question tidak boleh kosong', 'status': 'error'}), 400

    logger.info(f"\n{'='*80}")
    logger.info(f"USER QUERY: {question}")
    logger.info(f"{'='*80}")

    start_time = time.time()

    greetings = ["halo", "hallo", "hai", "hello", "selamat pagi", "selamat siang",
                 "selamat sore", "selamat malam", "wilujeng", "kumaha", "damang"]

    # Level 1: Greeting (keyword check — instant)
    if any(g in question.lower() for g in greetings):
        return jsonify({
            'answer': f'{get_greeting()}! Wilujeng sumping, kumaha damang? Abdi tiasa ngabantosan naon?',
            'source': 'greeting',
            'status': 'success',
            'query_time_ms': 0
        })

    # Level 2: Intent Classifier via LLM
    logger.info("[Intent] Classifying query intent...")
    intent = classify_intent(question)

    if intent == 'greeting':
        logger.info("Response: Greeting detected via intent classifier")
        return jsonify({
            'answer': 'Wilujeng sumping! Kumaha damang? Abdi tiasa ngabantosan naon?',
            'source': 'greeting',
            'status': 'success',
            'query_time_ms': 0
        })

    elif intent == 'knowledge':
        # Level 3: RAG ChromaDB
        logger.info("[Intent] Knowledge query - routing ke RAG")
        if rag_system and rag_system.kb_retriever:
            kb_docs = rag_system.kb_retriever.search(question, n_results=1)
            if kb_docs and kb_docs[0]['distance'] < rag_system.kb_retriever.semantic_threshold:
                logger.info(f"[RAG] Relevant doc found (distance: {kb_docs[0]['distance']:.4f})")
                result = rag_system.ask(question)
                kb_relevant = result.get('relevance', {}).get('kb_relevant', False)
                if kb_relevant:
                    query_time = (time.time() - start_time) * 1000
                    logger.info("Response: RAG answered the query")
                    return jsonify({
                        'question': result['question'],
                        'answer': result['answer'],
                        'sources': result.get('sources', []),
                        'relevance': result.get('relevance', {}),
                        'source': 'rag',
                        'status': 'success',
                        'query_time_ms': round(query_time, 2)
                    })
            logger.info("[RAG] No relevant doc found, falling back to Text-to-SQL...")

    else:
        # intent == 'database' or 'explanation'
        logger.info(f"[Intent] '{intent}' query - routing ke Text-to-SQL")

    # Level 4: Text-to-SQL (LlamaIndex NLSQLTableQueryEngine — CORE)
    try:
        logger.info("Triggering Text-to-SQL...")

        # Step 1: Route ke engine yang tepat (schema kecil per engine)
        selected_engine, table_used = route_query_engine(question)
        logger.info(f"[Router] Selected engine for: {table_used}")

        # Step 2: Build enhanced question dengan few-shot COMPACT
        date_filter = get_mysql_date_filter(question)
        few_shot = get_few_shot(question)

        date_hint = ""
        if date_filter:
            date_hint = f"\nMYSQL DATE FILTER: {date_filter}\nGunakan filter ini di WHERE clause!\n"
            logger.info(f"Date filter detected: {date_filter}")

        enhanced_question = f"""{few_shot}{date_hint}
Question: {question}
SQL:"""

        logger.info(f"[Prompt] Enhanced question length: {len(enhanced_question)} chars")

        # Step 3: Query via LlamaIndex (CORE — NLSQLTableQueryEngine)
        response = selected_engine.query(enhanced_question)

        logger.info(f"RAW RESPONSE TEXT: {str(response)[:500]}")
        logger.info(f"RESPONSE METADATA: {getattr(response, 'metadata', None)}")

        sql_query_raw = ""
        data_result = []
        col_keys = []

        if hasattr(response, 'metadata') and response.metadata:
            sql_query_raw = response.metadata.get('sql_query', '')
            data_result = response.metadata.get('result', [])
            col_keys = response.metadata.get('col_keys', [])

        # Step 4: Extract + Auto-fix SQLite→MySQL
        sql_query = extract_sql(sql_query_raw)
        if sql_query:
            sql_query = fix_sqlite_to_mysql(sql_query)

        logger.info(f"{'='*60}")
        logger.info(f"RAW SQL   : {sql_query_raw}")
        logger.info(f"CLEANED   : {sql_query}")
        logger.info(f"ROWS      : {len(data_result) if isinstance(data_result, list) else 'N/A'}")
        logger.info(f"TABLE     : {table_used}")
        logger.info(f"{'='*60}")

        # Step 4b: Jika fix_sqlite_to_mysql mengubah SQL, re-execute
        cleaned_from_raw = extract_sql(sql_query_raw)
        if sql_query and cleaned_from_raw and sql_query != cleaned_from_raw:
            logger.info("[SQL Fix] SQL was modified, re-executing fixed query...")
            try:
                with db_engine.connect() as conn:
                    result_proxy = conn.execute(text(sql_query))
                    col_keys = list(result_proxy.keys())
                    data_result = [dict(row._mapping) for row in result_proxy.fetchall()]
                    logger.info(f"[SQL Fix] Re-execute OK — {len(data_result)} rows")
            except Exception as e:
                logger.error(f"[SQL Fix] Re-execute failed: {e}")
                # Keep original data_result from LlamaIndex

        # Step 5: Validate
        if not sql_query or not data_result:
            logger.info("SQL generation/execution failed - returning fallback")
            return jsonify({
                'answer': 'Mohon Maaf, saya hanya dapat membantu pertanyaan terkait sistem IMRON, sensor enose, atau data object detection.',
                'source': 'fallback',
                'status': 'success'
            })

        # Step 6: Score filter
        if data_result and is_score_query(question):
            cleaned = []
            for row in data_result:
                if isinstance(row, tuple):
                    score_val = next((v for v in row if isinstance(v, Decimal)), None)
                elif isinstance(row, dict):
                    score_val = row.get('actual_score')
                else:
                    score_val = None
                if score_val is None or float(score_val) > 0:
                    cleaned.append(row)
            if cleaned:
                data_result = cleaned
            logger.info(f"[Filter] Score filter applied, {len(data_result)} rows remaining")

        # Step 7: Synthesize answer
        if is_explanation_query(question):
            logger.info("Explanation query - using synthesize_explanation()")
            answer = synthesize_explanation(question, data_result, col_keys=col_keys)
        elif is_simple_result(data_result):
            logger.info("Simple result - using format_simple_result()")
            answer = format_simple_result(data_result, question)
        else:
            logger.info("Complex result - using synthesize_answer()")
            answer = synthesize_answer(question, sql_query, data_result, table_used=table_used)

        query_time = (time.time() - start_time) * 1000
        return jsonify({
            'answer': answer,
            'source': 'text_to_sql',
            'sql_query': sql_query,
            'table_used': table_used,
            'status': 'success',
            'query_time_ms': round(query_time, 2)
        })

    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        return jsonify({'error': 'Database error occurred', 'status': 'error'}), 500
    except Exception as e:
        logger.error(f"ERROR in text-to-SQL: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'answer': 'Mohon Maaf, saya hanya dapat membantu pertanyaan terkait sistem IMRON, sensor enose, atau data object detection.',
            'source': 'fallback',
            'status': 'success'
        })

@app.route('/api/reload-kb', methods=['POST'])
def reload_kb():
    user_ip = request.remote_addr
    if not rate_limiter.is_allowed(user_ip):
        return jsonify({'error': 'Rate limit exceeded', 'status': 'error'}), 429

    if not rag_system or not rag_system.kb_retriever:
        return jsonify({'error': 'Knowledge base tidak tersedia', 'status': 'error'}), 500

    logger.info("[API] Manual reload KB requested")
    success = rag_system.kb_retriever.reload()
    if success:
        return jsonify({
            'status': 'success',
            'message': 'Knowledge base berhasil di-reload',
            'document_count': rag_system.kb_retriever.get_count(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
    return jsonify({'status': 'error', 'message': 'Gagal reload knowledge base'}), 500

@app.route('/api/kb-status', methods=['GET'])
def kb_status():
    if not rag_system or not rag_system.kb_retriever:
        return jsonify({'status': 'not_available', 'kb_available': False, 'document_count': 0})
    count = rag_system.kb_retriever.get_fresh_count()
    metrics = rag_system.kb_retriever.get_metrics()
    return jsonify({
        'status': 'ready',
        'kb_available': True,
        'document_count': count,
        'collection_name': rag_system.kb_retriever.collection_name,
        'chroma_path': rag_system.kb_retriever.chroma_db_path,
        'metrics': metrics
    })

@app.route('/api/kb-list', methods=['GET'])
def kb_list():
    if not rag_system or not rag_system.kb_retriever:
        return jsonify({'error': 'KB tidak tersedia', 'status': 'error'}), 500

    rag_system.kb_retriever.reload_if_needed()
    collection = rag_system.kb_retriever.collection
    limit = int(request.args.get('limit', 100))
    offset = int(request.args.get('offset', 0))

    all_results = collection.get(limit=limit, offset=offset, include=["documents", "metadatas"])
    documents = []
    if all_results and all_results['ids']:
        for i in range(len(all_results['ids'])):
            documents.append({
                'id': all_results['ids'][i],
                'content_preview': (all_results['documents'][i][:150] if all_results['documents'][i] else 'No content'),
                'metadata': all_results['metadatas'][i] if all_results['metadatas'] else {}
            })

    return jsonify({
        'status': 'success',
        'total': rag_system.kb_retriever.get_count(),
        'returned': len(documents),
        'limit': limit,
        'offset': offset,
        'documents': documents
    })

@app.route('/api/kb-health', methods=['GET'])
def kb_health():
    if not rag_system or not rag_system.kb_retriever:
        return jsonify({'status': 'unavailable', 'healthy': False}), 503

    count = rag_system.kb_retriever.get_count()
    seconds_since_reload = time.time() - rag_system.kb_retriever._last_reload_time
    healthy = count > 0 and seconds_since_reload < 300
    return jsonify({
        'status': 'healthy' if healthy else 'degraded',
        'healthy': healthy,
        'document_count': count,
        'last_reload': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(rag_system.kb_retriever._last_reload_time)),
        'seconds_since_reload': round(seconds_since_reload, 2)
    })

@app.route('/health', methods=['GET'])
def health():
    kb_status_val = 'healthy' if (rag_system and rag_system.kb_retriever) else 'unavailable'
    kb_count = rag_system.kb_retriever.get_count() if (rag_system and rag_system.kb_retriever) else 0
    return jsonify({
        'status': 'healthy',
        'service': 'IMRON Hybrid Chatbot API',
        'version': '3.0',
        'features': ['RAG-ChromaDB', 'Text-to-SQL', 'Greeting', 'Rate-Limiter'],
        'kb_status': kb_status_val,
        'kb_documents': kb_count,
        'model': OLLAMA_MODEL,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint tidak ditemukan', 'status': 'error'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

# ==================== MAIN ====================

if __name__ == '__main__':
    logger.info("="*80)
    logger.info("IMRON Hybrid Chatbot API - RAG + Text-to-SQL")
    logger.info("="*80)

    logger.info("Menginisialisasi RAG system...")
    if initialize_rag():
        logger.info("\nEndpoints tersedia:")
        logger.info("  POST /api/query      - Chat utama (RAG + Text-to-SQL)")
        logger.info("  GET  /api/kb-status  - Status knowledge base")
        logger.info("  GET  /api/kb-list    - List dokumen KB")
        logger.info("  GET  /api/kb-health  - Health check KB")
        logger.info("  POST /api/reload-kb  - Manual reload KB")
        logger.info("  GET  /health         - Health check server")
        logger.info("\nServer ready on port 5001")
        logger.info("="*80)
        app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
    else:
        logger.error("[ERROR] Gagal menginisialisasi RAG system.")
        sys.exit(1)