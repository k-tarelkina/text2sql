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
