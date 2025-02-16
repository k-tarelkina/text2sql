from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset


def train_lora(params, logger):
    # get params
    model_name = params.get("llm")
    checkpoint_dir = params.get("checkpoint_dir")
    rank = params.get("rank", 64)
    lora_alpha = params.get("lora_alpha", 16)
    lora_dropout = params.get("lora_dropout", 0.1)
    output_dir = params.get("output_dir", f"./qlora_spider_{model_name}")
    max_steps = params.get("max_steps", 1000)
    batch_size = params.get("batch_size", 8)
    learning_rate = params.get("learning_rate", 1e-4)
    save_steps = params.get("save_steps", 500)

    # init tokenizer from base llm
    logger.write("Start init tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.write("End init tokenizer")

    # load llm from checkpoint or hg
    if checkpoint_dir:
        logger.write(f"Start loading model from checkpoint: '{checkpoint_dir}'")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir, load_in_4bit=True, device_map="auto"
        )
    else:
        logger.write(f"Start loading model from HG: '{model_name}'")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, load_in_4bit=True, device_map="auto"
        )

    model = prepare_model_for_kbit_training(model)
    logger.write("End loading model from checkpoint")

    # init LoRA
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    logger.write("LoRa with LLM are init")

    # pre process dataset
    logger.write("Start loading and preprocessing dataset")
    dataset = load_dataset("spider")

    # utility function
    def preprocess_function(examples):
        inputs = [f"Generate SQL: {q}" for q in examples["question"]]
        outputs = examples["query"]
        tokenized_inputs = tokenizer(
            inputs,
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        tokenized_outputs = tokenizer(
            outputs, truncation=True, max_length=512, padding="max_length"
        )
        tokenized_inputs["labels"] = tokenized_outputs["input_ids"]
        return tokenized_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    logger.write("End loading and preprocessing dataset")

    # prepare args for training
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=max_steps,
        learning_rate=learning_rate,
        save_steps=save_steps,
        logging_steps=50,
        evaluation_strategy="steps",
        save_strategy="steps",
        bf16=True,
        report_to="none",
        load_best_model_at_end=True,
        auto_find_batch_size=True,
        metric_for_best_model="eval_samples_per_second",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )
    logger.write("Trainer is init")

    # run training
    if checkpoint_dir:
        logger.write(f"Resuming training from checkpoint: '{checkpoint_dir}'")
        trainer.train(resume_from_checkpoint=checkpoint_dir)
    else:
        logger.write(f"Start training")
        trainer.train()

    tokenizer.save_pretrained(training_args.output_dir)
    logger.write(f"End training")
