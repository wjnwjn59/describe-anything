import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size: int) -> T.Compose:
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def load_image(image_file: str, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size)
    image = transform(image)
    return image.unsqueeze(0)

_model = None
_tokenizer = None

def get_model(model_name="OpenGVLab/InternVL2_5-8B"):
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True,
            device_map="auto",
        ).eval().to(device)
        _tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False)
    return _model, _tokenizer

def inference(questions, image_paths, config=None, batch_size=8):
    model, tokenizer = get_model()
    prompt = config["prompt"] if config and "prompt" in config else "Answer the question based on the image."
    device = next(model.parameters()).device
    results = []
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i+batch_size]
        batch_image_paths = image_paths[i:i+batch_size]
        batch_prompts = [f"{prompt}\n<image>Question: {q}" for q in batch_questions]
        batch_images = [load_image(img, input_size=448).to(torch.bfloat16).to(device) for img in batch_image_paths]
        batch_images = torch.cat(batch_images, dim=0)
        with torch.no_grad():
            responses = model.chat(
                tokenizer,
                batch_images,
                batch_prompts,
                {"max_new_tokens": config["max_new_tokens"], "do_sample": False, "temperature": 0.01},
            )
        # model.chat trả về list nếu input là list, string nếu input là string
        if isinstance(responses, str):
            responses = [responses]
        answers = [resp.split("Answer:")[-1].strip() for resp in responses]
        results.extend(answers)
    return results