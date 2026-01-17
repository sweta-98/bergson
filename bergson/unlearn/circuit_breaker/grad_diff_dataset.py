import random
from typing import Dict

import torch
import transformers
from datasets import load_dataset
from torch.utils.data import Dataset

random.seed(0)


class GradDiffDataset(Dataset):
    """Dataset for gradient difference unlearning.

    Loads WMDP bio (forget) and MMLU STEM (retain) data formatted for
    next-token prediction. The training loop will:
    - Minimize loss on retain data (preserve STEM capability)
    - Maximize loss on forget data (unlearn bio hazard knowledge)
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        model_name_or_path: str,
        max_length: int = 1024,
        num_forget_examples: int = 5000,
        num_retain_examples: int = 5000,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_name_or_path = model_name_or_path.lower()

        # Model-specific templates
        self._setup_templates()

        # Load forget data (WMDP bio)
        self.forget_data = self._load_wmdp_bio(num_forget_examples)
        print(f"Loaded {len(self.forget_data)} forget examples (WMDP bio)")

        # Load retain data (MMLU STEM)
        self.retain_data = self._load_mmlu_stem(num_retain_examples)
        print(f"Loaded {len(self.retain_data)} retain examples (MMLU STEM)")

        # Shuffle both
        random.shuffle(self.forget_data)
        random.shuffle(self.retain_data)

    def _setup_templates(self):
        """Setup model-specific chat templates."""
        if "llama-3" in self.model_name_or_path:
            self.user_tag = "<|start_header_id|>user<|end_header_id|>\n\n"
            self.assistant_tag = (
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        elif "mistral" in self.model_name_or_path:
            self.user_tag = "[INST] "
            self.assistant_tag = " [/INST] "
        elif "eleutherai" in self.model_name_or_path:
            self.user_tag = "Human: "
            self.assistant_tag = "\n\nAssistant: "
        else:
            # Default format
            self.user_tag = "Question: "
            self.assistant_tag = "\nAnswer: "

    def _format_mcqa(self, question: str, choices: list, answer_idx: int) -> str:
        """Format a multiple choice question with the correct answer."""
        choice_letters = ["A", "B", "C", "D"]
        choices_text = "\n".join(
            f"{letter}. {choice}"
            for letter, choice in zip(choice_letters, choices)
        )
        prompt = f"{question.strip()}\n{choices_text}"
        answer = choice_letters[answer_idx]

        return f"{self.user_tag}{prompt}{self.assistant_tag}{answer}"

    def _load_wmdp_bio(self, num_examples: int) -> list:
        """Load WMDP bio dataset for forget set."""
        formatted_data = []

        # Load all WMDP bio subcategories
        subcategories = [
            "bioweapons_and_bioterrorism",
            "dual_use_virology",
            "enhanced_potential_pandemic_pathogens",
            "expanding_access_to_threat_vectors",
            "reverse_genetics_and_easy_editing",
            "viral_vector_research",
        ]

        for subcat in subcategories:
            try:
                # EleutherAI dataset uses "robust" split not "test"
                ds = load_dataset(
                    "EleutherAI/wmdp_bio_robust_mcqa",
                    subcat,
                    split="robust",
                )
                for example in ds:
                    formatted = self._format_mcqa(
                        example["question"],
                        example["choices"],
                        example["answer"],
                    )
                    formatted_data.append(formatted)
            except Exception as e:
                print(f"Warning: Could not load {subcat}: {e}")
                continue

        if len(formatted_data) == 0:
            # Fallback: try loading the main dataset
            try:
                ds = load_dataset(
                    "cais/wmdp",
                    "wmdp-bio",
                    split="test",
                )
                for example in ds:
                    formatted = self._format_mcqa(
                        example["question"],
                        example["choices"],
                        example["answer"],
                    )
                    formatted_data.append(formatted)
            except Exception as e:
                print(f"Warning: Could not load cais/wmdp: {e}")

        # Repeat if needed to reach num_examples
        if len(formatted_data) > 0:
            while len(formatted_data) < num_examples:
                formatted_data = formatted_data * 2
            formatted_data = formatted_data[:num_examples]

        return formatted_data

    def _load_mmlu_stem(self, num_examples: int) -> list:
        """Load MMLU STEM subjects for retain set."""
        formatted_data = []

        # STEM subjects in MMLU
        stem_subjects = [
            "abstract_algebra",
            "anatomy",
            "astronomy",
            "college_biology",
            "college_chemistry",
            "college_computer_science",
            "college_mathematics",
            "college_physics",
            "computer_security",
            "conceptual_physics",
            "electrical_engineering",
            "elementary_mathematics",
            "high_school_biology",
            "high_school_chemistry",
            "high_school_computer_science",
            "high_school_mathematics",
            "high_school_physics",
            "high_school_statistics",
            "machine_learning",
        ]

        for subject in stem_subjects:
            try:
                # Try auxiliary_train split first (larger), then test
                for split in ["auxiliary_train", "test", "validation"]:
                    try:
                        ds = load_dataset("cais/mmlu", subject, split=split)
                        for example in ds:
                            formatted = self._format_mcqa(
                                example["question"],
                                example["choices"],
                                example["answer"],
                            )
                            formatted_data.append(formatted)
                        break
                    except Exception:
                        continue
            except Exception as e:
                print(f"Warning: Could not load MMLU {subject}: {e}")
                continue

        # Repeat if needed
        if len(formatted_data) > 0:
            while len(formatted_data) < num_examples:
                formatted_data = formatted_data * 2
            formatted_data = formatted_data[:num_examples]

        return formatted_data

    def __len__(self):
        return min(len(self.forget_data), len(self.retain_data))

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        forget_text = self.forget_data[i % len(self.forget_data)]
        retain_text = self.retain_data[i % len(self.retain_data)]

        tokenize_kwargs = dict(
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize forget data
        forget_tokens = self.tokenizer(forget_text, **tokenize_kwargs)

        # Tokenize retain data
        retain_tokens = self.tokenizer(retain_text, **tokenize_kwargs)

        return dict(
            input_ids_forget=forget_tokens["input_ids"].squeeze(0),
            attention_mask_forget=forget_tokens["attention_mask"].squeeze(0),
            input_ids_retain=retain_tokens["input_ids"].squeeze(0),
            attention_mask_retain=retain_tokens["attention_mask"].squeeze(0),
        )
