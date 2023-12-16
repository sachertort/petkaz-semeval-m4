import spacy
import json
import textstat
import pandas as pd
from tqdm import tqdm
import json

def jsonl_read(file_path: str) -> list:
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def jsonl_write(data: list, file_path: str) -> None:
    with open(file_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")



nlp = spacy.load('en_core_web_sm')
data_train = jsonl_read("SubtaskA/subtaskA_train_monolingual.jsonl")
data_dev = jsonl_read("SubtaskA/subtaskA_dev_monolingual.jsonl")


for sample in tqdm(data_train):
    sent_count = textstat.sentence_count(sample["text"])
    sample["sentence_count"] = sent_count
    flesch_reading_ease_score = textstat.flesch_reading_ease(sample["text"])
    sample["flesch_reading_ease"] = flesch_reading_ease_score
    flesch_kincaid_grade_score = textstat.flesch_kincaid_grade(sample["text"])
    sample["flesch_kincaid_grade"] = flesch_kincaid_grade_score
    linsear_write_formula_score = textstat.linsear_write_formula(sample["text"])
    sample["linsear_write_formula"] = linsear_write_formula_score
    difficult_words_score = textstat.difficult_words(sample["text"])
    sample["difficult_words"] = difficult_words_score / sent_count
    lex_count = textstat.lexicon_count(sample["text"], removepunct=True)
    sample["lexicon_count"] = lex_count / sent_count


for sample in tqdm(data_dev):
    sent_count = textstat.sentence_count(sample["text"])
    sample["sentence_count"] = sent_count
    flesch_reading_ease_score = textstat.flesch_reading_ease(sample["text"])
    sample["flesch_reading_ease"] = flesch_reading_ease_score
    flesch_kincaid_grade_score = textstat.flesch_kincaid_grade(sample["text"])
    sample["flesch_kincaid_grade"] = flesch_kincaid_grade_score
    linsear_write_formula_score = textstat.linsear_write_formula(sample["text"])
    sample["linsear_write_formula"] = linsear_write_formula_score
    difficult_words_score = textstat.difficult_words(sample["text"])
    sample["difficult_words"] = difficult_words_score / sent_count
    lex_count = textstat.lexicon_count(sample["text"], removepunct=True)
    sample["lexicon_count"] = lex_count / sent_count


jsonl_write(data_train, "/processed_data/readibility_train.jsonl")
jsonl_write(data_train, "/processed_data/readibility_dev.jsonl")
