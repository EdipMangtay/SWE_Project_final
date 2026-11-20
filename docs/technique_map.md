# Technique map

| Area | What the code does | Where to inspect it |
|---|---|---|
| A · Neural networks | Train skip-gram word and context vectors using five negative samples. Average known word vectors, train a 64-unit ReLU classifier by backpropagation and apply dropout. | `sentinel/train.py`, `sentinel/model.py` |
| B · Classical NLP | Normalize noisy words, train word unigram/bigram TF-IDF and logistic regression, and score an independent edit-distance rule. | `sentinel/text.py`, `sentinel/train.py`, `scripts/evaluate_all.py` |
| C · Transformers | Apply the original cased multilingual WordPiece vocabulary, update the final two encoder blocks, pooler and classification head. | `sentinel/train.py` |
| D · Efficient models | Dynamically quantize 74 linear layers and quantize the large word embedding by row. Reload from disk on CPU and measure one message at a time. | `sentinel/model.py`, `scripts/check_int8.py`, `scripts/evaluate_all.py` |
