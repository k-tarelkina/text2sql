# text2sql

## Installation & Setup

To access Hugging Face models, create a `.env` file in the root folder of this project and paste your Hugging Face access token there.

```
// .env
HF_TOKEN={your access token}
```

Install necessary dependencies:

```bash
pip install transformers bitsandbytes accelerate datasets outlines scikit-learn python-dotenv
```

## Evaluation

To run the evaluation script, you first need to generate files with predictions.

To do that, run the following command:

```bash
python generate_predictions.py
```

The files will be generated in the `results` folder.

The evaluation script is based on that from https://github.com/taoyds/spider/tree/master.

```bash
python evaluation.py --gold [gold file] --pred [predicted file] --etype [evaluation type] --db [database dir] --table [table file]

arguments:
  [gold file]        gold.sql file where each line is `a gold SQL \t db_id`
  [predicted file]   predicted sql file where each line is a predicted SQL
  [evaluation type]  "match" for exact set matching score, "exec" for execution score, "time" for execution time, and "all" for all
  [database dir]     directory which contains sub-directories where each SQLite3 database is stored
  [table file]       table.json file which includes foreign key info of each database
```
