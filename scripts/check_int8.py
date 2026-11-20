import argparse
from sentinel.config import ROOT, seed, save_json
from sentinel.model import load_model, quantized_layers

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["dev", "final"], default="final")
    args = p.parse_args()
    seed()
    run = ROOT / "runs" / args.mode
    model = load_model(run, "int8")
    result = model.predict("Thanks for the game!")
    assert 0 <= result["confidence"] <= 1
    save_json(
        run / "results/int8_reload.json",
        {
            "linear_modules": quantized_layers(model.model),
            "embedding_dtype": str(
                model.model.bert.embeddings.word_embeddings.codes.dtype
            ),
            "prediction": result,
        },
    )
    print(result)
