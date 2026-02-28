import os

def split_dev_sql_by_database():
    """
    Memisahkan dev.sql berdasarkan database yang dipilih
    Database: concert_singer, pets_1, employee_hire_evaluation, orchestra, singer
    """
    
    dev_file = "scripts/spider/evaluation_examples/dev.sql"
    output_dir = "database"
    
    target_databases = ['concert_singer', 'pets_1', 'employee_hire_evaluation', 'orchestra', 'singer']
    
    os.makedirs(output_dir, exist_ok=True)
    
    database_data = {db: {'questions': [], 'sqls': []} for db in target_databases}
    
    current_question = None
    current_db = None
    
    print(f"Reading {dev_file}...")
    with open(dev_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if not line:
                continue
            
            if line.startswith('Question'):
                parts = line.split('|||')
                if len(parts) == 2:
                    question_part = parts[0].strip()
                    db_id = parts[1].strip()
                    

                    if db_id in target_databases:
                        current_question = question_part
                        current_db = db_id
                    else:
                        current_question = None
                        current_db = None
            
            elif line.startswith('SQL:') and current_question and current_db:
                sql = line.replace('SQL:', '').strip()
                
                database_data[current_db]['questions'].append(current_question)
                database_data[current_db]['sqls'].append(sql)
    
    total_questions = 0
    
    for db_name in target_databases:
        if database_data[db_name]['questions']:

            questions_file = os.path.join(output_dir, f"{db_name}_questions.txt")
            with open(questions_file, 'w', encoding='utf-8') as f:
                for i, q in enumerate(database_data[db_name]['questions'], 1):
                    f.write(f"{q} ||| {db_name}\n")
            
            gold_file = os.path.join(output_dir, f"{db_name}_gold.sql")
            with open(gold_file, 'w', encoding='utf-8') as f:
                for sql in database_data[db_name]['sqls']:
                    f.write(f"{sql}\t{db_name}\n")
            
            combined_file = os.path.join(output_dir, f"{db_name}_dev.sql")
            with open(combined_file, 'w', encoding='utf-8') as f:
                for i, (q, sql) in enumerate(zip(database_data[db_name]['questions'], 
                                                   database_data[db_name]['sqls']), 1):
                    f.write(f"{q} ||| {db_name}\n")
                    f.write(f"SQL:  {sql}\n\n")
            
            count = len(database_data[db_name]['questions'])
            total_questions += count
            print(f"\n{db_name}:")
            print(f"  - Questions file: {questions_file} ({count} questions)")
            print(f"  - Gold SQL file: {gold_file} ({count} queries)")
            print(f"  - Combined file: {combined_file}")
    
    all_questions_file = os.path.join(output_dir, "gold_195.sql")
    all_gold_file = os.path.join(output_dir, "gold_195_formatted.sql")
    
    question_num = 1
    with open(all_questions_file, 'w', encoding='utf-8') as fq:
        for db_name in target_databases:
            for q in database_data[db_name]['questions']:
                fq.write(f"Question {question_num}:  {q.split(':')[1].strip() if ':' in q else q} ||| {db_name}\n")
                question_num += 1
    
    with open(all_gold_file, 'w', encoding='utf-8') as fg:
        for db_name in target_databases:
            for sql in database_data[db_name]['sqls']:
                fg.write(f"{sql}\t{db_name}\n")
    
    print(f"\n{'='*60}")
    print("SPLIT SELESAI!")
    print(f"{'='*60}")
    print(f"Total questions dari 5 database: {total_questions}")
    print(f"\nFile gabungan:")
    print(f"  - All questions: {all_questions_file}")
    print(f"  - All gold SQL: {all_gold_file}")
    print(f"\nDatabase breakdown:")
    for db_name in target_databases:
        count = len(database_data[db_name]['questions'])
        print(f"  - {db_name}: {count} questions")
    print(f"{'='*60}")

if __name__ == "__main__":
    split_dev_sql_by_database()