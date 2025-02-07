# text2sql

## Installation & Setup

To access Hugging Face models, create a `.env` file in the root folder of this project and paste your Hugging Face access token there.

```
// .env
HF_TOKEN={your access token}
```

Install necessary dependencies:

```bash
pip install transformers bitsandbytes accelerate datasets outlines scikit-learn python-dotenv nltk gdown
```

## Evaluation

To run the evaluation script, you first need to generate files with predictions.

To do that, run the following command:

```bash
python main.py predict

# or

python main.py predict --params_path params_llama.yaml
```


The files will be generated in the `results` folder.

The evaluation script is based on that from https://github.com/taoyds/spider/tree/master.

Before running the script, please make sure to download the databases from the test suite and place them in the root directory of this project:

```bash
gdown 1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w
unzip testsuitedatabases.zip -d text2sql
```

```bash
python main.py evaluate

# or

python main.py evaluate --params_path params_llama.yaml
```
