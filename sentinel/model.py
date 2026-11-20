from dataclasses import dataclass
from time import perf_counter_ns
import joblib
import numpy as np
import torch
from torch import nn
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from .config import TOKENS, read_json
from .text import clean_text, displayed, tokens, validate


class FeedForward(nn.Sequential):
    def __init__(self, dim):
        super().__init__(
            nn.Linear(dim, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1)
        )


def sentence_vectors(texts, vocab, embeddings):
    result = np.zeros((len(texts), embeddings.shape[1]), dtype=np.float32)
    for i, text in enumerate(texts):
        ids = [vocab[token] for token in tokens(text) if token in vocab]
        if ids:
            result[i] = embeddings[ids].mean(0)
    return torch.from_numpy(result)


class RowInt8Embedding(nn.Module):
    """Row-wise symmetric storage; restore only the embeddings the message uses."""

    def __init__(self, weight):
        super().__init__()
        with torch.no_grad():
            scale = weight.abs().amax(1).clamp(min=1e-8) / 127
            self.register_buffer(
                "codes",
                (weight / scale[:, None]).round().clamp(-127, 127).to(torch.int8),
            )
            self.register_buffer("scale", scale.float())

    def forward(self, ids):
        return self.codes[ids].float() * self.scale[ids].unsqueeze(-1)


def quantize(model):
    from torch.ao.quantization import quantize_dynamic

    engine = next(
        (
            x
            for x in ("x86", "fbgemm", "qnnpack")
            if x in torch.backends.quantized.supported_engines
        ),
        None,
    )
    if not engine:
        raise RuntimeError(
            "This PyTorch CPU build has no supported INT8 linear engine."
        )
    torch.backends.quantized.engine = engine
    model.cpu().eval()
    emb = model.bert.embeddings
    emb.word_embeddings = RowInt8Embedding(emb.word_embeddings.weight.detach())
    return quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8, inplace=True)


def quantized_layers(model):
    from torch.ao.nn.quantized.dynamic import Linear

    return sum(
        isinstance(m, Linear) and m.weight().dtype == torch.qint8
        for m in model.modules()
    )


def encoding(tok, texts, device):
    return tok(
        [clean_text(t) for t in texts],
        padding=True,
        truncation=True,
        max_length=TOKENS,
        return_tensors="pt",
    ).to(device)


@dataclass
class Predictor:
    kind: str
    model: object
    threshold: float
    tokenizer: object = None
    vocab: object = None
    embeddings: object = None

    @torch.inference_mode()
    def scores(self, texts):
        if self.kind == "tfidf":
            return self.model.predict_proba(texts)[:, 1]
        if self.kind == "neural":
            x = sentence_vectors(texts, self.vocab, self.embeddings)
            return self.model(x).flatten().sigmoid().cpu().numpy()
        batch = encoding(self.tokenizer, texts, "cpu")
        return self.model(**batch).logits.softmax(-1)[:, 1].cpu().numpy()

    def predict(self, text):
        text = validate(text)
        start = perf_counter_ns()
        p = float(self.scores([text])[0])
        offensive = p >= self.threshold
        ms = (perf_counter_ns() - start) / 1e6
        return {
            "model": self.kind,
            "label": "OFFENSIVE" if offensive else "CLEAN",
            "confidence": p if offensive else 1 - p,
            "offensive_score": p,
            "latency_ms": ms,
            "display": displayed(text, offensive),
        }


def load_model(run, kind):
    folder = run / "artifacts" / kind
    meta = read_json(folder / "metadata.json")
    if kind == "tfidf":
        return Predictor(
            kind, joblib.load(folder / "pipeline.joblib"), meta["threshold"]
        )
    if kind == "neural":
        word = np.load(folder / "vectors.npz", allow_pickle=False)
        vocab = {w: i for i, w in enumerate(word["words"].tolist())}
        vectors = word["vectors"]
        model = FeedForward(vectors.shape[1])
        model.load_state_dict(
            torch.load(folder / "network.pt", map_location="cpu", weights_only=True)
        )
        return Predictor(
            kind, model.eval(), meta["threshold"], vocab=vocab, embeddings=vectors
        )
    tok = BertTokenizer.from_pretrained(folder, local_files_only=True)
    if kind == "bert":
        model = BertForSequenceClassification.from_pretrained(
            folder, local_files_only=True, attn_implementation="eager"
        )
    elif kind == "int8":
        config = BertConfig.from_pretrained(folder, local_files_only=True)
        config._attn_implementation = "eager"
        model = quantize(BertForSequenceClassification(config))
        state = torch.load(folder / "int8.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        if quantized_layers(model) != meta["linear_modules"] or not quantized_layers(
            model
        ):
            raise ValueError("INT8 module verification failed.")
    else:
        raise ValueError(f"Unknown model {kind}")
    return Predictor(kind, model.eval(), meta["threshold"], tokenizer=tok)
