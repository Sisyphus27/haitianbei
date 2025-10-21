'''
Author: zy
Date: 2025-10-21 15:29:29
LastEditTime: 2025-10-21 15:29:33
LastEditors: zy
Description: 
FilePath: \haitianbei\resoterd\scripts\download_models.py

'''
import argparse
from pathlib import Path
from typing import Literal


def download_with_transformers(model_id: str, out_dir: Path, task: Literal["auto", "seq2seq", "distilbert"]= "auto"):
    out_dir.mkdir(parents=True, exist_ok=True)
    if task == "distilbert":
        from transformers import DistilBertTokenizer, DistilBertModel
        tok = DistilBertTokenizer.from_pretrained(model_id)
        tok.save_pretrained(out_dir)
        mdl = DistilBertModel.from_pretrained(model_id)
        mdl.save_pretrained(out_dir)
    elif task == "seq2seq":
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tok = AutoTokenizer.from_pretrained(model_id)
        tok.save_pretrained(out_dir)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        mdl.save_pretrained(out_dir)
    else:
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(model_id)
        tok.save_pretrained(out_dir)
        mdl = AutoModel.from_pretrained(model_id)
        mdl.save_pretrained(out_dir)


def download_with_hf_hub(model_id: str, out_dir: Path):
    from huggingface_hub import snapshot_download
    out_dir.mkdir(parents=True, exist_ok=True)
    # snapshot_download will create a hashed subfolder; we still put it under out_dir
    snapshot_download(repo_id=model_id, cache_dir=str(out_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["transformers", "hub"], nargs="?", default="transformers")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1] / "model"))
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    # distilbert-base-cased
    distilbert_id = "distilbert-base-cased"
    distilbert_dir = root / "distilbert-base-cased"

    # Babelscape/rebel-large
    rebel_id = "Babelscape/rebel-large"
    rebel_dir = root / "rebel-large"

    if args.method == "transformers":
        download_with_transformers(distilbert_id, distilbert_dir, task="distilbert")
        download_with_transformers(rebel_id, rebel_dir, task="seq2seq")
    else:
        download_with_hf_hub(distilbert_id, distilbert_dir)
        download_with_hf_hub(rebel_id, rebel_dir)

    print("Done. Saved to:")
    print(" -", distilbert_dir)
    print(" -", rebel_dir)


if __name__ == "__main__":
    main()
