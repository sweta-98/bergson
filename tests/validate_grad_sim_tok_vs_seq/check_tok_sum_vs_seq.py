
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# device='cuda'

import sys

sys.path.insert(0, "../bergson")
from pathlib import Path

import numpy as np

from bergson.data import load_scores

seq_scr_path = "./teacher_number_scorings_seq/score"
scr_path = "./teacher_number_scorings_tok/score"

seq_scrs = load_scores(Path(seq_scr_path))
seq_scrs = np.array([score[0] for score in seq_scrs])

offsets = np.load(scr_path + "/offsets.npy")
num_token_grads = np.load(scr_path + "/num_token_grads.npy")
total_tokens = int(offsets[-1])
scores = np.memmap(
    Path(scr_path) / "token_scores.bin",
    dtype=np.float32,
    mode="r",
    shape=(total_tokens,),
)

for i in range(1):  # len(dataset)
    ex_scores = scores[offsets[i] : offsets[i + 1]]
    print("---------------")
    print("Sequence score:", seq_scrs[i])
    print("Token scores sum:", ex_scores.sum())
    print("Token scores mean:", ex_scores.mean())
    print("---------------")
    if ex_scores.sum() == seq_scrs[i]:
        print("---> TEST PASSED")
    else:
        print("---> TEST FAILED")
