import warnings
from multiprocessing import Pool
from typing import Any, Dict, List, Union

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling

model_name = "meta-llama/Llama-2-7b-hf"
dataset_name = "devxpy/therapychat"
max_seq_length = 1024


tokenizer = AutoTokenizer.from_pretrained(model_name)
# https://github.com/huggingface/transformers/issues/22312#issuecomment-1610558831
tokenizer.pad_token = "[PAD]"

tokenizer.add_special_tokens(
    {
        # "pad_token": "<pad>",
        # "bos_token": "<s>",
        # "eos_token": "</s>",
        # "unk_token": "<unk>",
    }
)

mapping = {
    "system": "### Human:",
    "human": "### Human:",
    "gpt": "### Assistant:",
}


class MyDataCollator(DataCollatorForLanguageModeling):
    ignore_index = -100.0

    human_token_ids = tokenizer.encode("\n### Human:", add_special_tokens=False)[2:]
    response_token_ids = tokenizer.encode("\n### Assistant:", add_special_tokens=False)[
        2:
    ]

    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        for i in range(len(examples)):
            response_token_ids_idxs = []
            human_token_ids_idxs = []

            for assistant_idx in np.where(
                batch["labels"][i] == self.response_token_ids[0]
            )[0]:
                # find the indexes of the start of a response.
                if (
                    self.response_token_ids
                    == batch["labels"][i][
                        assistant_idx : assistant_idx + len(self.response_token_ids)
                    ].tolist()
                ):
                    response_token_ids_idxs.append(
                        assistant_idx + len(self.response_token_ids)
                    )

            if len(response_token_ids_idxs) == 0:
                warnings.warn(
                    f"Could not find response key `#### Assistant:` in the "
                    f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                    f"This instance will be ignored in loss calculation. "
                    f"Note, if this happens often, consider increasing the `max_seq_length`."
                )
                batch["labels"][i, :] = self.ignore_index

            human_token_ids = self.human_token_ids
            for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
                # find the indexes of the start of a human answer.
                if (
                    human_token_ids
                    == batch["labels"][i][
                        human_idx : human_idx + len(human_token_ids)
                    ].tolist()
                ):
                    human_token_ids_idxs.append(human_idx)

            if len(human_token_ids_idxs) == 0:
                warnings.warn(
                    f"Could not find instruction key `### Human:` in the "
                    f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                    f"This instance will be ignored in loss calculation. "
                    f"Note, if this happens often, consider increasing the `max_seq_length`."
                )
                batch["labels"][i, :] = self.ignore_index

            for idx, (start, end) in enumerate(
                zip(human_token_ids_idxs, response_token_ids_idxs)
            ):
                # Make pytorch loss function ignore all non response tokens
                if idx != 0:
                    batch["labels"][i, start:end] = self.ignore_index
                else:
                    batch["labels"][i, :end] = self.ignore_index

            if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index

            # print(response_token_ids_idxs, human_token_ids_idxs)

        return batch


def tokenize_lines(lines):
    return tokenizer("\n".join(lines), add_special_tokens=False, truncation=False)[
        "input_ids"
    ]


bos = tokenizer.bos_token
eos = tokenizer.eos_token


def examples_gen(convo):
    if convo[0]["from"] == "system":
        system_msg = "\n" + convo[0]["value"] + "\n"
    else:
        system_msg = ""
    system_msg = (
        mapping["system"] + " " + bos + "<<SYS>>" + system_msg + "<</SYS>>" + eos
    )

    examples = []
    lines = [system_msg]

    for msg in convo:
        line = mapping[msg["from"]] + " " + bos + msg["value"] + eos

        next_seq_length = len(tokenize_lines(lines + [line]))
        if next_seq_length > max_seq_length:
            input_ids = tokenize_lines(lines)
            if len(input_ids) <= max_seq_length:
                examples.append(input_ids)
            lines = [system_msg]

        lines.append(line)

    if len(lines) > 1:
        input_ids = tokenize_lines(lines)
        if len(input_ids) <= max_seq_length:
            examples.append(input_ids)

    return examples


dataset = load_dataset(dataset_name)
print(dataset["train"])


def flatten(l1):
    return [it for l2 in l1 for it in l2]


with Pool() as pool:
    train_dataset = flatten(pool.map(examples_gen, dataset["train"]["conversations"]))


class MyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return len(train_dataset)

    def __getitem__(self, i):
        return train_dataset[i]


# Step 3: Define the training arguments
training_args = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=0.0002,
    optim="adamw_bnb_8bit",
    lr_scheduler_type="cosine",
    warmup_steps=10,
    logging_first_step=True,
    logging_steps=1,
    num_train_epochs=3,
    report_to="wandb",
    save_steps=100,
    save_total_limit=10,
    log_level="debug",
    bf16=True,
    # fp16=True,
)

peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=BitsAndBytesConfig(
        load_in_8bit=False,
        load_in_4bit=True,
    ),
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    # torch_dtype = torch.float16,
    trust_remote_code=True,
)
model.config.use_cache = (
    False  # Gradient checkpointing is used by default but not compatible with caching
)
model = get_peft_model(model, peft_config)

# Step 5: Define the Trainer
trainer = Trainer(
    model,
    args=training_args,
    data_collator=MyDataCollator(tokenizer=tokenizer, mlm=False),
    train_dataset=MyDataset(),
    # train_dataset=list(dataset_gen()),
    tokenizer=tokenizer,
)

try:
    trainer.train()
except KeyboardInterrupt:
    input("Press enter to save")

    # Step 6: Save the model
    trainer.save_model("output")
