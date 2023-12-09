from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

paths = [str(x) for x in Path("ss3-txt").glob("*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=2_000, min_frequency=100, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

tokenizer.save_model("EsperBERTo")
