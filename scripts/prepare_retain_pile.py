#!/usr/bin/env python3
"""Create a combined retain + pile dataset for training without forget data."""

from datasets import concatenate_datasets, load_dataset, load_from_disk

OUTPUT = "data/wmdp_retain_pile"


def main():
    retain = load_from_disk("data/wmdp_retain")
    pile = load_dataset("NeelNanda/pile-10k", split="train")

    # Normalize columns: both need 'text' and 'source'
    pile = pile.remove_columns([c for c in pile.column_names if c != "text"])
    pile = pile.map(lambda x: {"source": "pile"})

    # retain already has 'text' and 'source'='retain'
    retain = retain.select_columns(["text", "source"])

    combined = concatenate_datasets([retain, pile])
    combined.save_to_disk(OUTPUT)
    print(f"Saved {len(combined)} examples to {OUTPUT}")
    print(f"  retain: {len(retain)}, pile: {len(pile)}")


if __name__ == "__main__":
    main()
