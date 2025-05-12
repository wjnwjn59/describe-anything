import argparse
import json
import os
import importlib.util
from tqdm import tqdm
from config import ds_collections  

def load_inference(model_name):
    model_file = os.path.join(os.path.dirname(
        __file__), "inference_models", f"{model_name}.py")
    spec = importlib.util.spec_from_file_location(model_name, model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.inference

def load_config(dataset_name):
    config = ds_collections[dataset_name].copy()
    config["dataset_name"] = dataset_name
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (file in inference_models/)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (key in ds_collections)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file (jsonl)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference")
    args = parser.parse_args()

    config = load_config(args.dataset)
    inference = load_inference(args.model)

    if config["data_path"].endswith(".jsonl"):
        with open(config["data_path"], encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    else:
        with open(config["data_path"], encoding="utf-8") as f:
            data = json.load(f)

    questions = [item["question"] for item in data]
    img_paths = [os.path.join(config["image_folder"], item["image"]) for item in data]
    gts = [item["answer"] for item in data]

    predicts = []
    for i in tqdm(range(0, len(questions), args.batch_size)):
        batch_questions = questions[i:i+args.batch_size]
        batch_img_paths = img_paths[i:i+args.batch_size]
        batch_answers = inference(batch_questions, batch_img_paths, config=config, batch_size=args.batch_size)
        print(batch_answers)
        print(gts[i:i+args.batch_size])
        predicts.extend(batch_answers)

    output_file = args.output or f"{args.model}_{config['dataset_name']}.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for pred, gt in zip(predicts, gts):
            f.write(json.dumps({"predict": pred, "gt": gt}, ensure_ascii=False) + "\n")
    print(f"Saved results to {output_file}")


if __name__ == "__main__":
    main()
