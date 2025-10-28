"""
The ARC dataset from Allen AI.
https://huggingface.co/datasets/allenai/ai2_arc
"""

from pathlib import Path
from datasets import load_dataset, Dataset
from tasks.common import Task, render_mc

class ARC(Task):

    def __init__(self, subset, split, data_dir=None, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["ARC-Easy", "ARC-Challenge"], "ARC subset must be ARC-Easy or ARC-Challenge"
        assert split in ["train", "validation", "test"], "ARC split must be train|validation|test"

        if data_dir is not None:
            # Load from local parquet file
            data_path = Path(data_dir) / "arc" / f"{subset}_{split}.parquet"
            if not data_path.exists():
                raise FileNotFoundError(f"Local dataset not found: {data_path}. Please run scripts/download_datasets.py first.")
            self.ds = Dataset.from_parquet(str(data_path))
        else:
            # Load from HuggingFace
            self.ds = load_dataset("allenai/ai2_arc", name=subset, split=split)

        self.ds = self.ds.shuffle(seed=42)

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"] # the question text
        choices = row["choices"]["text"] # the text of each choice
        answer_string = row["answerKey"] # e.g. "A", "B", "C", "D"
        letters = row["choices"]["label"] # e.g. ["A", "B", "C", "D"]
        assert answer_string in letters, f"ARC answer {answer_string} must be one of {letters}" # sanity check
        # create and return the Conversation object
        user_message = render_mc(question, letters, choices)
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer_string}
        ]
        conversation = {
            "messages": messages,
            "letters": letters, # useful during evaluation, so we can narrow and clamp the assistant prediction to one of the letters
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        # the assert here is not strictly speaking needed, but currently the way we eval, we expect this to be true
        # I'm going to leave the assert here to prevent footguns, but possibly in the future can remove it.
        assert assistant_response in conversation['letters'], f"ARC answer {assistant_response} is expected to be one of {conversation['letters']}"
        assistant_message = conversation['messages'][-1]['content'] # e.g. "A"
        return assistant_response == assistant_message
