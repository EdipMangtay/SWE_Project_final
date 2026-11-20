# SentinelAI experiment report

**Question.** Can an INT8 version of a fine-tuned multilingual BERT moderate one English message in under 50 ms on average while preserving useful detection of the minority class?

## Data and split

We used the public Jigsaw Toxic Comment Classification Challenge training file, mirrored by Heliosoph. The 159,571 input rows include six human-labeled toxicity flags. A value of one in any flag produces OFFENSIVE. Every flag must be zero for CLEAN. Official unlabeled test data was excluded.

We grouped comments by normalized, case-folded text after the 4,096-character cap. We discarded groups with conflicting binary labels, including 13 conflicting groups, and retained one stable ID per remaining group. There were 159,295 eligible records. Seed 42 selected 5,000 stratified examples for a manageable CPU run. Training used 4,000 comments (406 offensive), validation used 500 (51 offensive), and testing used 500 (50 offensive). Mean original comment lengths were 421, 414 and 365 characters respectively. The stored ID lists and prediction CSVs identify every example.

## Training

| Model | Fitting and selection |
|---|---|
| TF-IDF + logistic regression | Word unigrams/bigrams with minimum document frequency 2, sublinear TF and balanced class weights. C=4 won the two-value validation search. The selected threshold was 0.425. |
| Word2Vec + neural network | Train skip-gram with a context window of two and five negatives on training text. Average known 64-dimensional word vectors. Fit a 64-unit ReLU hidden layer with 20% dropout and weighted binary cross-entropy. Validation selected epoch 17 and a 0.55 threshold. |
| Multilingual BERT | Keep the cased WordPiece model. Freeze embeddings plus the lower ten of twelve blocks. Train 14,767,874 parameters in the last two blocks, pooler and classification head with AdamW, 4e-5, batch 16, up to 64 tokens. Epoch 3 had the best validation macro-F1 (0.878). Threshold 0.575. |
| Quantized multilingual BERT | Starting from the winning FP32 checkpoint, convert 74 linear layers to dynamic INT8, and store the large word embedding as signed INT8 with one FP32 scale per vocabulary row. Retain the FP32 threshold. Remaining floating-point operations include layer normalization and softmax. |

The independent edit-distance rule accepts word-to-lexicon distance at most one. Example: `id1otx` normalizes to `idiotx`, one edit from `idiot`. Used alone, the rule achieved macro-F1 0.639 and recalled 10 of 50 offensive comments. No learned model uses the rule to override its decisions.

## Held-out results

| Model | Macro-F1 | Accuracy | Offensive precision | Offensive recall | FP / FN | Size MB | Mean ms | Median ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + LR | 0.7852 | 92.2% | 0.608 | 0.620 | 20 / 19 | 0.62 | 0.330 | 0.277 | 0.653 |
| Word2Vec + NN | 0.7921 | 91.6% | 0.563 | 0.720 | 28 / 14 | 2.07 | 0.102 | 0.067 | 0.320 |
| BERT FP32 | 0.8002 | 93.0% | 0.660 | 0.620 | 16 / 19 | 714.36 | 25.180 | 26.657 | 28.622 |
| BERT INT8 | 0.8262 | 93.8% | 0.694 | 0.680 | 15 / 16 | 182.93 | 35.038 | 39.354 | 45.076 |

Each model scored the same 500 held-out messages individually. We then used 15 warm-ups and the same 150 single-message timed inputs. Timings include normalization, tokenization/featurization, the forward pass and the decision. They exclude model loading, user-interface rendering and allocation of a new session. Model sizes use decimal MB across the complete runnable model directory, with vocabulary/tokenizer and metadata included. The portable distribution patch is excluded from size comparisons.

Testing ran on an Apple M3 Pro (12 cores, 36 GB), macOS 27.2.1 arm64, Python 3.12.15, with four PyTorch intra-op threads. PyTorch 2.14.1 and Transformers 5.20.2 were used for training and inference. The INT8 engine was QNNPACK.

## Interpretation and limits

INT8 used 74.4% less disk space than FP32. It ran about 9.86 ms slower on average while remaining within the predeclared 50 ms mean budget. Dynamic activation quantization and backend overhead are plausible explanations for the speed difference; neither was profiled separately. INT8 also had the highest macro-F1 on this test. This single 50-positive split cannot establish that quantization consistently improves accuracy. Word2Vec had the highest offensive recall but also masked 28 clean comments.

An actual false negative, ID `779c81facf11970b`, reads “Somebody out there hates you ... See here.” Human label OFFENSIVE; INT8 offensive score 0.0044; output CLEAN. This kind of contextual hostility warrants a separate short-chat evaluation. The single-message system has no conversation history as model input. The confidence scores are uncalibrated. English Wikipedia talk pages differ from live gaming chat. Although the encoder was pretrained in many languages, this experiment evaluated only English. Future work should gather labeled game messages, compare context windows and calibrate scores on validation data.

See `runs/final/results/` for the raw measurements and the four prediction files. The curves and confusion matrices are in `docs/figures/`.
