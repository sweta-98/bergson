import csv
import json
import random
from typing import Dict
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import load_dataset, Dataset
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

random.seed(0)


class CircuitBreakerDatasetBio(TorchDataset):

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        num_examples,
        lorra_args,
        model_name_or_path,
        bio_forget_path="/home/luciarosequirke/bio_tmp/bio_forget_ds",
    ):
        super(CircuitBreakerDatasetBio, self).__init__()

        self.model_name_or_path = model_name_or_path.lower()
        self.max_length = 1024
        self.bio_forget_path = bio_forget_path

        one_shot_template = (
            "{user_tag}{instruction}{assistant_tag}<SEPARATOR>{response}"
        )

        # ================ Model and Template Config  ================
        # Default configs
        sep_token = ""
        switch_select = [0]
        use_refusal_retain = False
        user_tag, assistant_tag = None, None
        if "llama-3" in self.model_name_or_path:
            print("USING LLAMA TEMPLATE")
            user_tag = "<|start_header_id|>user<|end_header_id|>\n\n"
            assistant_tag = (
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
            switch_select = [0, 1]
            use_refusal_retain = True
        elif "mistral" in self.model_name_or_path:
            print("USING MISTRAL TEMPLATE")
            # fix spacing issue in template
            tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
            user_tag = "[INST] "
            assistant_tag = " [/INST]"
            sep_token = " "
        elif "eleutherai" in self.model_name_or_path:
            print("USING DEEP IGNORANCE (GPTNeoX) TEMPLATE")
            # Set a simple chat template for GPTNeoX
            tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}Human: {{ message['content'] }}\n\nAssistant: {% elif message['role'] == 'assistant' %}{{ message['content'] }}{% endif %}{% endfor %}"
            # Simple user/assistant format for GPTNeoX
            user_tag = "Human: "
            assistant_tag = "\n\nAssistant: "
            sep_token = "\n\n"
            switch_select = [0, 1]
        else:
            raise NotImplementedError(f"Config {self.model_name_or_path} not found")

        assert user_tag and assistant_tag, "user_tag/assistant_tag not defined"

        self.user_tag = user_tag
        self.assistant_tag = assistant_tag
        self.sep_token = sep_token

        # ======================= Retain ======================= #
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
        orig_s = []
        for example in ds:
            messages = example["messages"]
            if len(messages) < 2:
                continue

            switch = np.random.choice(switch_select)
            if switch == 0:
                chat_template_result = tokenizer.apply_chat_template(
                    messages, tokenize=False
                )
                # Handle case where tokenizer.bos_token might be None
                bos_token = tokenizer.bos_token or ""
                formatted_input = str(chat_template_result).replace(bos_token, "")
            else:  # switch == 1
                formatted_input = one_shot_template.format(
                    user_tag=user_tag,
                    assistant_tag=assistant_tag,
                    instruction="",
                    response=messages[1]["content"],
                )

            orig_s.append(formatted_input)

            if len(orig_s) > num_examples:
                break
        self.orig_s_retain = orig_s
        random.shuffle(self.orig_s_retain)
        print("orig_s_retain[0]", orig_s[0])
        print("Orig s length:", len(self.orig_s_retain))

        # ======================= Borderline Retain ======================= #
        # from https://github.com/paul-rottger/exaggerated-safety
        with open("data/xstest_v2_completions_gpt4_gpteval.csv", newline="") as f:
            data = [dict(row) for row in csv.DictReader(f)]
            data = [row for row in data if row["final_label"] == "1_full_compliance"]

        borderline_orig_s = []
        for i, d in enumerate(data * 50):
            switch = np.random.choice(switch_select)
            if switch == 0:
                formatted_input = one_shot_template.format(
                    user_tag=user_tag,
                    assistant_tag=assistant_tag,
                    instruction=d["prompt"],
                    response=d["completion"],
                )
            else:  # switch == 1
                formatted_input = one_shot_template.format(
                    user_tag=user_tag,
                    assistant_tag=assistant_tag,
                    instruction="",
                    response=d["completion"],
                )

            borderline_orig_s.append(formatted_input)

        self.orig_s_retain += borderline_orig_s
        random.shuffle(self.orig_s_retain)
        print("borderline_orig_s[0]", borderline_orig_s[0])
        print("Orig s length:", len(self.orig_s_retain))

        # ======================= Refusal Retain ======================= #
        if use_refusal_retain:
            with open("data/circuit_breakers_train.json") as file:
                dataset = json.load(file)

            random.shuffle(dataset)
            dataset = dataset[:2000]
            refusal_retain_orig = []
            for i, d in tqdm(enumerate(dataset * 2)):
                switch = np.random.choice(switch_select)
                if switch == 0:
                    formatted_input = one_shot_template.format(
                        user_tag=user_tag,
                        assistant_tag=assistant_tag,
                        instruction=d["prompt"],
                        response=d["llama3_output"],
                    )
                else:  # switch == 1
                    formatted_input = one_shot_template.format(
                        user_tag=user_tag,
                        assistant_tag=assistant_tag,
                        instruction="",
                        response=d["llama3_output"],
                    )

                refusal_retain_orig.append(formatted_input)

            self.orig_s_retain += refusal_retain_orig
            random.shuffle(self.orig_s_retain)
            print("refusal_orig_s[0]", refusal_retain_orig[0])
            print("Orig s length:", len(self.orig_s_retain))

        # ======================= Circuit Breaker (Bio Forget) ======================= #
        print(f"Loading bio_forget dataset from {self.bio_forget_path}")
        bio_forget_dataset = Dataset.load_from_disk(self.bio_forget_path)
        circuit_breaker_orig = []

        # Create prompts to make the model generate biomedical content we want to forget
        bio_prompts = [
            "Tell me about {title}",
            "Provide information about {title}",
            "Explain the research on {title}",
            "What can you tell me about {title}",
            "Summarize the work on {title}",
        ]

        for i in tqdm(range(min(10000, len(bio_forget_dataset)))):
            try:
                example = bio_forget_dataset[i]
                # Skip if text is missing
                text_content = example.get("text", "") if isinstance(example, dict) else ""
                title_content = example.get("title", "") if isinstance(example, dict) else ""

                if not text_content or not title_content:
                    continue

                # Create a prompt asking about the research
                prompt = random.choice(bio_prompts).format(title=title_content)

                # Use the text content as what we want to "forget"
                bio_output = text_content

                # Truncate if too long to avoid memory issues
                if len(bio_output) > 3000:
                    bio_output = bio_output[:3000] + "..."

                switch = np.random.choice(switch_select)
                if switch == 0:
                    formatted_input = one_shot_template.format(
                        user_tag=user_tag,
                        assistant_tag=assistant_tag,
                        instruction=prompt,
                        response=bio_output,
                    )
                else:  # switch == 1
                    formatted_input = one_shot_template.format(
                        user_tag=user_tag,
                        assistant_tag=assistant_tag,
                        instruction="",
                        response=bio_output,
                    )

                circuit_breaker_orig.append(formatted_input)
            except Exception as e:
                print(f"Error processing bio example {i}: {e}")
                continue

        self.circuit_breaker_orig = circuit_breaker_orig
        random.shuffle(self.circuit_breaker_orig)
        print("bio circuit_breaker_orig[0]", circuit_breaker_orig[0] if circuit_breaker_orig else "No examples")
        print("Bio circuit length:", len(self.circuit_breaker_orig))

        # ======================= Val ======================= #
        with open("data/circuit_breakers_val.json") as file:
            dataset = json.load(file)
        val_orig = []
        for i, d in tqdm(enumerate(dataset)):
            val_orig.append(
                one_shot_template.format(
                    user_tag=user_tag,
                    assistant_tag=assistant_tag,
                    instruction=d["prompt"],
                    response=d["output"],
                )
            )

        self.val_orig = val_orig
        self.tokenizer = tokenizer

    def __len__(self):
        return min(len(self.orig_s_retain), len(self.circuit_breaker_orig))

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        orig_s_retain = self.orig_s_retain[i]
        circuit_breaker_orig = self.circuit_breaker_orig[i]
        val_orig = self.val_orig[i % len(self.val_orig)]

        cb_tokenized_kwargs = dict(
            max_length=512, padding="max_length", truncation=True, return_tensors="pt"
        )
        tokenize_kwargs = dict(
            max_length=1024, padding="max_length", truncation=True, return_tensors="pt"
        )

        # =========== Circuit Breaker Inputs ===========
        # === split to [request, response] shape [512,512] to support different mask configs ===
        cb_request, cb_response = circuit_breaker_orig.split("<SEPARATOR>")
        self.tokenizer.padding_side = "left"
        tokenized_request_circuit_breaker = self.tokenizer(
            cb_request, **cb_tokenized_kwargs
        )
        self.tokenizer.padding_side = "right"
        response_tokenized_circuit_breaker = self.tokenizer(
            cb_response, add_special_tokens=False, **cb_tokenized_kwargs
        )
        self.tokenizer.padding_side = "left"

        combined_input_ids_circuit_breaker = torch.cat(
            [
                tokenized_request_circuit_breaker["input_ids"],
                response_tokenized_circuit_breaker["input_ids"],
            ],
            dim=1,
        )
        combined_attention_mask_circuit_breaker = torch.cat(
            [
                tokenized_request_circuit_breaker["attention_mask"],
                response_tokenized_circuit_breaker["attention_mask"],
            ],
            dim=1,
        )

        # ========== Retain Inputs ===========
        tokenized_inputs_retain = self.tokenizer(
            orig_s_retain.replace("<SEPARATOR>", self.sep_token), **tokenize_kwargs
        )

        # =========== Val Inputs ===========
        tokenized_inputs_val = self.tokenizer(
            val_orig.replace("<SEPARATOR>", self.sep_token), **tokenize_kwargs
        )

        return dict(
            input_ids_circuit_breaker=combined_input_ids_circuit_breaker,
            attention_mask_circuit_breaker=combined_attention_mask_circuit_breaker,
            input_ids=tokenized_inputs_retain["input_ids"],
            attention_mask=tokenized_inputs_retain["attention_mask"],
            input_ids_val=tokenized_inputs_val["input_ids"],
            attention_mask_val=tokenized_inputs_val["attention_mask"],
        )