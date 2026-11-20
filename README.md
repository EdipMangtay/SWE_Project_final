# SentinelAI · Real-time chat moderation

SentinelAI is a Streamlit chat application for SWE015 Use Case 4. It flags English messages, masks flagged text, reports model confidence and times every prediction on CPU. The default model is the reloaded INT8 version of the fine-tuned multilingual BERT. Click **Compare all four models** after sending a message to compare live decisions.

**Ali Edip Mangtay · 210911044 · İstinye University**

| Course material | File |
|---|---|
| Ten-slide presentation | [Presentation PDF](docs/presentation.pdf) |
| Methods, results and error analysis | [Technical report PDF](docs/technical_report.pdf) |
| Video preparation and timed English narration | [Recording guide PDF](docs/video_rehberi.pdf) |
| Measured results for all four models | [Comparison CSV](runs/final/results/comparison.csv) |

## Run the project

**On the project Mac.** The desktop project already has its four trained artifacts and a Python environment. Double-click `Baslat.command` or run it in Terminal from the project folder.

**On a fresh machine.** Use 64-bit Python 3.12 and a PyTorch CPU build with an INT8 linear backend. The measured machine had 36 GB of RAM. Allow several GB of disk space for PyTorch, the base checkpoint and restored models.

**For the exact submitted models**, download [the four trained models](https://github.com/EdipMangtay/SWE_Project_final/releases/download/v1.0.0/SentinelAI_Trained_Models.zip) and extract the contents into the repository root. The directory `runs/final/artifacts/int8/` should now contain `int8.pt`. The ZIP's SHA-256 is `c8daf07122491d168b456432391803f8e295cf4c96c1029ddf4b5b15a7e1629a`. The released files include the tokenizer and both BERT checkpoints. After installing the pinned dependencies, this setup runs without downloading data or training.

On macOS, run `zsh Baslat.command` from the project directory. It installs the dependencies on first use and starts Streamlit. With the model files extracted, it loads the existing checkpoints. Without model files, a source-only checkout trains the four models from the pinned public sources. A compact portable project with the training patch is also available on the [release page](https://github.com/EdipMangtay/SWE_Project_final/releases/tag/v1.0.0).

For manual setup, including on other platforms:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If you extracted the trained models, proceed directly to the application launch command below. For a source-only checkout without fitted models, run `python -m scripts.train_all --mode final`. For the compact portable project ZIP, which includes trained TF-IDF and Word2Vec artifacts, tokenizers and a lossless BERT patch under `weights/`, restore the full FP32 model and export INT8 once:

```sh
python -m scripts.restore_models
```

The restore step downloads the **pinned original 714 MB Google multilingual BERT file** from Hugging Face. All trained differences are supplied locally. Each of the 201 restored tensors is SHA-256 verified against the final experiment, then the INT8 model is produced on the target CPU and checked in a separate process. For an offline restore, provide the original pinned base file with `python -m scripts.restore_models --base-file /path/to/model.safetensors`. On the desktop project the ready-made FP32 and INT8 files already exist. The UI uses local weights and requires no API key or network after setup.

**Application launch command**

```sh
streamlit run app.py
```

Open the local address printed by Streamlit. Try “Thanks for your help!” followed by “You are a stupid id1ot.” The saved INT8 model displayed the first message and masked the second in the verification run. Clear the session with the sidebar button. Long messages are capped at 4,096 characters. BERT processes up to 64 WordPiece tokens.

## Reproduce training, evaluation and the slides

Run commands from the project root with the environment active.

```sh
python -m scripts.verify_environment
python -m scripts.train_all --mode dev
python -m scripts.evaluate_all --mode dev
python -m scripts.train_all --mode final
python -m scripts.evaluate_all --mode final
python -m scripts.figures
python -m scripts.presentation
python -m pytest -q --junitxml=runs/final/results/tests.xml
python -m scripts.verify_submission
```

Development uses 300 genuine Jigsaw records and reduced epochs under `runs/dev/`. The final run uses 5,000 examples under `runs/final/`. Use the supplied final results for the delivered deck. Training again updates that run's files. To keep the delivered snapshot intact, train in a second extracted copy. The audit exits with a nonzero code while submission items are missing.

The data stage downloads the pinned labeled Jigsaw training file. An official Kaggle download also works: `python -m scripts.train_all --stage data --raw /path/to/train.csv`. Then train stages `tfidf`, `neural`, `bert` and `int8` individually with `--stage`. For offline BERT training provide the original local checkpoint directory with `--checkpoint /path/to/mbert`. Run `python -m scripts.train_all --stage data` to reconstruct input splits before re-running evaluation in a portable copy. Training supports CUDA, Apple MPS when available, and CPU. Inference and benchmarking always run on CPU.

## What was measured

| Model | Macro-F1 | Accuracy | Complete size | Mean CPU time |
|---|---:|---:|---:|---:|
| TF-IDF + logistic regression | 0.7852 | 92.2% | 0.62 MB | 0.33 ms |
| Own Word2Vec + feedforward net | 0.7921 | 91.6% | 2.07 MB | 0.10 ms |
| Fine-tuned multilingual BERT | 0.8002 | 93.0% | 714.36 MB | 25.18 ms |
| INT8 fine-tuned multilingual BERT | 0.8262 | 93.8% | 182.93 MB | 35.04 ms |

The test set had 450 clean and 50 offensive comments. For each model, 15 warm-ups preceded 150 timed single-message inferences on an Apple M3 Pro. Each mean includes model preprocessing, inference and the decision. INT8 saved 74.4% in disk space, took longer than FP32, and met the 50 ms mean target. Its p95 was 45.08 ms.

## Model and data design

Six Jigsaw flags are combined with logical OR. Only rows bearing all six known binary labels participate. Conflicting groups and duplicate normalized texts are removed before sampling. A fixed seed of 42 sets the stratified 80/10/10 split. Fitting and threshold selection use training and validation data respectively. Each exported test prediction carries its original Jigsaw ID.

The independent Levenshtein lexicon baseline is evaluated alongside a TF-IDF/logistic baseline. Skip-gram Word2Vec is trained from scratch with negative sampling. A PyTorch feedforward classifier consumes mean word vectors. Multilingual BERT keeps the lower ten blocks frozen and updates the last two, pooler and head. Its saved checkpoint is quantized into dynamically quantized linear layers and per-row INT8 word embeddings. The technical report documents the selection criteria and an actual failure example.

## Files

- `app.py`, `.streamlit/config.toml` — chat, masking, session history and model comparison.
- `sentinel/` — data, normalization, model adapters, training and quantization.
- `scripts/` — data/training entry point, evaluation, restore, plots, presentation and audit.
- `runs/final/artifacts/` — one runnable directory per model after restoration.
- `runs/final/results/` — split IDs, training logs, predictions, CPU measurements, test and audit output.
- `weights/` — the portable lossless patch, distributed separately from public repository source.
- `tests/` — preprocessing, real model inference, export consistency and Streamlit integration tests.
- `docs/` — PDF presentation, editable cover configuration, figures, report, technique map, narration, recording checklist and AI declaration.

## Submission and links

After recording, enter the unlisted YouTube or link-accessible Google Drive URL in the `video_url` field of `docs/submission.json`. Run `python -m scripts.presentation` to regenerate the presentation, and publish the updated configuration and PDF. The code and model download links are already configured. The cover needs both the repository and video link for course submission. Use `python -m scripts.handouts` to regenerate the technical report and recording guide from their Markdown sources.

## Troubleshooting

- **Model unavailable.** Check you launched from the extracted project. Restore the packaged weights or run final training. The app never substitutes another model when a load fails.
- **Dataset unavailable.** Download the labeled competition training CSV and supply `--raw`. If the base model is already on disk, pass the exact pinned weights or checkpoint directory using the commands above.
- **CPU quantization unavailable.** Use the tested PyTorch version on arm64 macOS or a supported x86 CPU wheel. The loader explicitly selects a usable backend. The pinned build emits deprecation warnings for its dynamic-quantization API. Its INT8 modules and inference are exercised by the tests.
- **Port busy or blocked.** Select a free port with `streamlit run app.py --server.port 8502`. A restricted runtime that forbids listening requires running the launch command in a normal terminal. In-process route and chat tests are also supplied.
- **Different timings.** Use the same CPU thread count and single-message batch size. Clock frequency, background load and backend affect latency.
- **Language and context.** This run evaluates English Wikipedia comments. Sarcasm, implied hostility, spelling changes and text beyond the model's token window can yield errors.

## Attribution and AI usage

The comment source is English Wikipedia talk pages. Dataset labels are published under CC0; underlying Wikipedia text retains CC BY-SA 3.0 attribution/share-alike terms as described by the dataset distributors. The failure excerpt is identified by the original row ID in the report. The upstream BERT weights are Apache-2.0. OpenAI ChatGPT / Codex assisted with implementation, debugging, documentation and slides. Experimental results come from the included executed pipelines.

- [Kaggle Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)
- [Mirror and release details](https://huggingface.co/datasets/Heliosoph/Jigsaw-Toxic-Comments)
- [Comment text attribution](https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge)
- [Pretrained multilingual BERT](https://huggingface.co/google-bert/bert-base-multilingual-cased)
- [Mikolov et al., skip-gram and negative sampling](https://arxiv.org/abs/1310.4546)
- [PyTorch dynamic quantization](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.quantize_dynamic.html)
