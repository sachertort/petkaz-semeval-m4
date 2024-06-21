# PetKaz at SemEval-2024 Task 8: Can Linguistics Capture the Specifics of LLM-generated Text?

[Paper link](https://arxiv.org/abs/2404.05483)

**SemEval 2024**

**Title:** PetKaz at SemEval-2024 Task 8: Can Linguistics Capture the Specifics of LLM-generated Text?

**Authors:** Kseniia Petukhova, Roman Kazakov, Ekaterina Kochmar

**Main idea:** For our submission to SemEval-2024 Task 8, the monolingual track of Subtask A, which focuses on MGT detection in English across a variety of do- mains and generative models, we have developed a system that can be categorized as a black-box detec- tor and is based on a combination of embeddings, measures of lexical diversity, and careful selection of the training data (see Figure below). We also present and discuss an extended set of linguistic features, including discourse and stylistic features, that we have experimented with during the development phase of the competition. The main motivation for using such a feature-based approach is that it helps us to focus on the fundamental differences between MGTs and human-written texts (HWTs) rather than capture the specifics of particular models.


## Replication

To begin with, install all the necessary packages:
```
pip install -r requirements.txt
```
To upload the data, run the following:
```
./get_data.sh
```
This script downloads:
* Subtask A `train` and `dev` data to the folder `SubtaskA`;
* files with extracted features, saved models for the extraction (e.g., vectorizers), the best combination model (`best_model_combination.pt`) to the folder `processed_data`;
* fine-tuned `roberta-base` to the folder `best_roberta`.

### Baseline model
This script is borrowed from another [repository](https://github.com/mbzuai-nlp/SemEval2024-task8). To run training of a model, run the following:
```
python transformer_baseline.py --train_file_path SubtaskA/subtaskA_train_monolingual.jsonl --test_file_path SubtaskA/subtaskA_dev_monolingual.jsonl --prediction_file_path results/baseline_predictions.jsonl --subtask B --model roberta-base
```

### Features extraction
**Text statistics and readability**

To obtain text statistics and readability features, run the following:
```
python features_extraction/readability_textstat.py
```

**Stylometry and lexical diversity**

To run extract stylometry and lexical diversity features, run the following:
```
python features_extraction/diversity_stylometry.py
```

**RST**

To run RST parsing, run the following:
```
python features_extraction/rst.py
```

**Entity grid**

To run entity grid, you need Python 3.7. Install requirements:
```
pip install -r requirements_entity_grid.txt
```

Then run:
```
python features_extraction/entity_grid.py
```

### Feed-forward neural network
To work with different configurations of features, stylometry dense vectors and embeddings, use the notebook `combination.ipynb`.

# Citation

If you find our work useful, please consider citing it!

```
@article{petukhova2024petkaz,
  title={PetKaz at SemEval-2024 Task 8: Can Linguistics Capture the Specifics of LLM-generated Text?},
  author={Petukhova, Kseniia and Kazakov, Roman and Kochmar, Ekaterina},
  journal={arXiv preprint arXiv:2404.05483},
  year={2024}
}
```
