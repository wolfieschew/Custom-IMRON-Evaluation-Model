# Deskripsi

Project ini adalah sistem evaluasi model bahasa (LLM) untuk tugas Text-to-SQL menggunakan benchmark case IMRON Dataset. Tujuan Project ini diperlukan untuk Pemilihan Model LLM yang paling optimal untuk nanti nya digunakan pada Web IMRON.

## Evaluasi Model LLM
10 Model yang digunakan dalam Evaluation merupakan Low Tier model dengan rentang 1B - 3B parameter
| Model         | Ukuran | 
|---------------|--------|
| qwen2.5       | 3B   | 
| qwen2        | 1.5B   |
| gemma3    | 1B     | 
| Llama3.2       | 1B     | 
| granite3.3     | 2B     | 
| falcon3      | 1B     | 
| smollm2       | 1.7B     | 
| deepseek-coder        | 1.3B     |
| phi      | 2.7B     | 
| InternLM2   | 1.8B   |

## Alur Evaluasi (Evaluation Flow)

Berikut adalah diagram alur proses evaluasi model LLM pada alur kasus IMRON:

![Evaluation of LLM Models in the IMRON Case Flow](https://i.postimg.cc/6Qzc1FBb/Eval-Model-IMRON.png)

### Penjelasan Singkat Alur Evaluasi:

1. **Generasi SQL Prediksi (`generate_predictions_llamaindex.py`)**
   * Menguji **10 model LLM** lokal (ukuran **1 - 3B parameter**) melalui kerangka kerja **LlamaIndex** dan **Ollama**.
   * Model menerima pertanyaan dari dataset `questions.sql` (total **45 pertanyaan** dengan tingkat kesulitan bertingkat) lalu men-generate query SQL prediksi berdasarkan skema tabel database IMRON yang relevan.
   * `questions.sql` berisi daftar pertanyaan, sementara `answer.sql` menyimpan query kunci jawaban (*gold standard*). Keduanya digunakan bersama untuk mencocokkan hasil generasi model.

2. **Evaluasi Query (`evaluation.py`)**
   * Query SQL hasil prediksi model dieksekusi dan dievaluasi dengan membandingkannya terhadap query kunci jawaban dari `answer.sql`.
   * Evaluasi dilakukan berdasarkan **4 Metrik Evaluasi**:
     * **Execution Accuracy (EX)**: Membandingkan keakuratan baris data hasil eksekusi query prediksi dengan kunci jawaban langsung di database (di luar query fallback `SELECT 1`).
     * **Exact Set Match (ESM)**: Mengurai struktur SQL (Select, From, Where, dll.) dan membandingkannya sebagai himpunan (Set) sehingga perbedaan urutan kolom atau kondisi tidak disalahkan secara keliru.
     * **Syntax Validity (VX)**: Memeriksa keabsahan sintaks SQL menggunakan `EXPLAIN` atau *subquery* `LIMIT 0` tanpa membebani pemrosesan database.
     * **Latency**: Mengukur kecepatan respons model dalam milidetik (ms) saat men-generate query.

3. **Perankingan Model (`rangking_results.py`)**
   * Mengumpulkan hasil ringkasan evaluasi dari semua model untuk dihitung menggunakan sistem skor berbobot (*Weighted Score*).
   * **Bobot Penilaian Akhir**:
     * **Execution Accuracy (EX) (Tanpa Fallback)**: **40%**
     * **Exact Set Match (ESM)**: **25%**
     * **Syntax Validity (VX)**: **20%**
     * **Latency**: **15%**
   * Hasil kalkulasi digunakan untuk menentukan peringkat model dari yang terbaik hingga terendah guna mendapatkan **1 Model Terbaik ("The Chosen One")**.

---

## Struktur Script & File Utama

*  **`generate_predictions_llamaindex.py`**
  Script untuk memproses pertanyaan menggunakan LlamaIndex dan Ollama, melakukan pembersihan sintaks SQLite ke MySQL, serta menerapkan RAG (Retrieval-Augmented Generation) pada skema tabel database.
*  **`evaluation.py`**
  Script evaluasi query SQL prediksi terhadap kunci jawaban menggunakan metrik EX, ESM, VX, dan Latency.
*  **`rangking_results.py`**
  Script kalkulasi skor berbobot (*Weighted Score*), analisis performa berdasarkan tingkat kesulitan query (*difficulty breakdown*), dan visualisasi perankingan.
*  **`questions.sql`**
  Menyimpan 45 pertanyaan uji yang dibagi menjadi beberapa level kesulitan (Easy, Medium, Hard, Extra Hard).
*  **`answer.sql`**
  Menyimpan query SQL kunci jawaban (*gold standard*) sebagai acuan evaluasi.
*  **`document/`**
  * `query_difficulty_explanation.md`: Penjelasan rinci mengenai kriteria dan pemetaan tingkat kesulitan pertanyaan uji (Easy, Medium, Hard, Extra Hard).
  * `evaluation_flow.png`: Gambar diagram alur proses evaluasi model.

---

## Setup & Instalasi

1. Pastikan Anda memiliki **Ollama** terinstal dan berjalan secara lokal.
2. Install dependensi Python yang dibutuhkan:
   ```bash
   pip install -r requirements.txt
   ```
3. Konfigurasikan file `.env` di direktori utama dengan detail koneksi database MySQL Anda:
   ```env
   DB_HOST=localhost
   DB_USER=root
   DB_PASSWORD=your_password
   DB_NAME=your_database_name
   DB_PORT=3306
   ```
4. Jalankan script prediksi dan evaluasi sesuai petunjuk penggunaan di masing-masing file.

## Acknowledgements & Citation

Skrip evaluasi (`evaluation.py`) dalam proyek ini diadaptasi dari repositori [Spider](https://github.com/taoyds/spider) yang dikembangkan oleh Yu et al. untuk metric **Exact Set Match (ESM)** dan **Execution Accuracy (EX)**.

```bibtex
@inproceedings{Yu&al.18c,
  title     = {Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task},
  author    = {Tao Yu and Rui Zhang and Kai Yang and Michihiro Yasunaga and Dongxu Wang and Zifan Li and James Ma and Irene Li and Qingning Yao and Shanelle Roman and Zilin Zhang and Dragomir Radev},
  booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
  address   = "Brussels, Belgium",
  publisher = "Association for Computational Linguistics",
  year      = 2018
}
```

Repositori asli: [https://github.com/taoyds/spider](https://github.com/taoyds/spider)
