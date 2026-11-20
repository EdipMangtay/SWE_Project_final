import torch
from scripts.evaluate_all import hardware
from sentinel.config import THREADS

if __name__ == "__main__":
    torch.set_num_threads(THREADS)
    print(hardware())
    print(
        "Training acceleration",
        {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()},
    )
    print("INT8 engines", torch.backends.quantized.supported_engines)
