## Deskripsi
Project ini adalah sistem evaluasi model bahasa (LLM) untuk tugas Text-to-SQL menggunakan benchmark case IMRON Dataset. Tujuan Project ini diperlukan untuk Pemilihan Model LLM yang paling optimal untuk nanti nya digunakan pada Web IMRON.

## Evaluasi Model LLM
10 Model yang digunakan dalam Evaluation merupakan Low Tier model dengan rentang 1B - 4B parameter
| Model         | Ukuran | Pendekatan              |
|---------------|--------|-------------------------|
| qwen2.5:3B       | 3B   | Manual & LlamaIndex     |
| qwen3:1.7B        | 1.7B   | Manual & LlamaIndex     |
| gemma3:1b    | 1B     | Manual & LlamaIndex     |
| Llama3.2:1B       | 1B     | Manual & LlamaIndex     |
| granite3.3:2b     | 2B     | Manual & LlamaIndex     |
| falcon3:1b      | 1B     | Manual & LlamaIndex     |
| smollm2:1.7b       | 1.7B     | Manual & LlamaIndex     |
| deepseek-r1:1.5b        | 4B     | Manual & LlamaIndex     |
| phi4-mini:3.8b       | 3.8B     | Manual & LlamaIndex     |
| InternLM2:1.8B   | 1.8B   | Manual & LlamaIndex     |
