from sgnlp.models.rst_pointer import (
    RstPointerParserConfig,
    RstPointerParserModel,
    RstPointerSegmenterConfig,
    RstPointerSegmenterModel,
    RstPreprocessor,
    RstPostprocessor
)

import spacy
from tqdm import tqdm
import pandas as pd
import utils as utils
import multiprocessing
import json
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

nlp = spacy.load("en_core_web_sm")

# Load processors and models
preprocessor = RstPreprocessor()
postprocessor = RstPostprocessor()

segmenter_config = RstPointerSegmenterConfig.from_pretrained(
    'https://storage.googleapis.com/sgnlp-models/models/rst_pointer/segmenter/config.json')
segmenter = RstPointerSegmenterModel.from_pretrained(
    'https://storage.googleapis.com/sgnlp-models/models/rst_pointer/segmenter/pytorch_model.bin',
    config=segmenter_config)
segmenter.eval()

parser_config = RstPointerParserConfig.from_pretrained(
    'https://storage.googleapis.com/sgnlp-models/models/rst_pointer/parser/config.json')
parser = RstPointerParserModel.from_pretrained(
    'https://storage.googleapis.com/sgnlp-models/models/rst_pointer/parser/pytorch_model.bin',
    config=parser_config)
parser.eval()

unique_relations = list()


def jsonl_read(file_path: str) -> list:
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data


def iter_children(root, relations_count):
    for child in root["children"]:
        if child["attributes"][0] in relations_count:
            relations_count[child["attributes"][0]] += 1
        else:
            relations_count[child["attributes"][0]] = 1
        
        if relations_count[child["attributes"][0]] not in unique_relations:
            unique_relations.append(relations_count[child["attributes"][0]])

        if "children" in child:
            iter_children(child, relations_count)
    
    return relations_count


def split_lists(list_a, chunk_size):
    for i in range(0, len(list_a), chunk_size):
        yield list_a[i:i + chunk_size]


def rst(file_path):
    data = utils.jsonl_read(file_path)
    for sample in tqdm(data):
        sentences = [str(i).replace("\n", "") for i in nlp(sample["text"]).sents]
        chunk_size = len(sentences) // 4
        if chunk_size != 0:
            batched_sents = list(split_lists(sentences, chunk_size))
        else:
            batched_sents = sentences
        trees_list = list()
        for batch in batched_sents:
            try:
                sentences = [str(i) for i in nlp(sample["text"]).sents]
                # print(sentences[0])
                tokenized_sentences_ids, tokenized_sentences, lengths = preprocessor(sentences)

                segmenter_output = segmenter(tokenized_sentences_ids, lengths)
                end_boundaries = segmenter_output.end_boundaries

                parser_output = parser(tokenized_sentences_ids, end_boundaries, lengths)

                trees = postprocessor(sentences=sentences, tokenized_sentences=tokenized_sentences,
                                    end_boundaries=end_boundaries,
                                    discourse_tree_splits=parser_output.splits)
                
                relations_count = dict()
                sample["rst"] = trees
                for tree in trees:
                    try:
                        if tree["root"]["attributes"][0] in relations_count:
                            relations_count[tree["root"]["attributes"][0]] += 1
                        else:
                            relations_count[tree["root"]["attributes"][0]] = 1
                        
                        if "children" in tree["root"]:
                            relations_count = iter_children(tree["root"], relations_count)
                    except:
                        print(tree)
                
                for key, value in relations_count.items():
                    sample[key] = value / len(sentences)

            except: 
                pass

        del sample["text"]
        del sample["model"]
        del sample["source"]
        del sample["label"]
        sample["rst"] = trees_list

    return data

train_rst = rst("SubtaskA/subtaskA_train_monolingual.jsonl")
dev_rst = rst("SubtaskA/subtaskA_dev_monolingual.jsonl")

for sample in tqdm(train_rst):
    try:
        sentences = [str(i) for i in nlp(sample["text"]).sents]
        relations_count = dict()
        trees = sample["rst"]
        for tree in trees:
            try:
                if tree["root"]["attributes"][0] in relations_count:
                    relations_count[tree["root"]["attributes"][0]] += 1
                else:
                    relations_count[tree["root"]["attributes"][0]] = 1

                if "children" in tree["root"]:
                    relations_count = iter_children(tree["root"], relations_count)
            except:
                print(tree)

        for key, value in relations_count.items():
            sample[key] = value / len(sentences)
    except:
        pass


for sample in tqdm(dev_rst):
    try:
        sentences = [str(i) for i in nlp(sample["text"]).sents]
        relations_count = dict()
        trees = sample["rst"]
        for tree in trees:
            try:
                if tree["root"]["attributes"][0] in relations_count:
                    relations_count[tree["root"]["attributes"][0]] += 1
                else:
                    relations_count[tree["root"]["attributes"][0]] = 1

                if "children" in tree["root"]:
                    relations_count = iter_children(tree["root"], relations_count)
            except:
                print(tree)

        for key, value in relations_count.items():
            sample[key] = value / len(sentences)
    except:
        pass


df_train = pd.DataFrame(train_rst)
imp=SimpleImputer(missing_values=np.NaN)
idf_train=pd.DataFrame(imp.fit_transform(df_train))
idf_train.columns=df_train.columns
idf_train.index=df_train.index

idf_train.to_csv("/processed_data/rst_train.csv")


df_dev = pd.DataFrame(dev_rst)
idf_dev=pd.DataFrame(imp.fit_transform(df_dev))
idf_dev.columns=df_dev.columns
idf_dev.index=df_dev.index

idf_dev.to_csv("/processed_data/rst_dev.csv")