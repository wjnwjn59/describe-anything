import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_processor = None


def get_model(model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
    global _model, _processor
    if _model is None or _processor is None:
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(device)
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        _processor = AutoProcessor.from_pretrained(
            model_name, min_pixels=min_pixels, max_pixels=max_pixels, use_fast=True
        )
        _processor.tokenizer.padding_side = "left"
    return _model, _processor


def inference(questions, image_paths, config=None, batch_size=4):
    model, processor = get_model()
    system_instruction = config["prompt"] if config and "prompt" in config else "Answer the question based on the image."
    messages_batch = []
    for q, img in zip(questions, image_paths):
        messages_batch.append([
            {"role": "system", "content": [
                {"type": "text", "text": system_instruction}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{img}"},
                    {"type": "text", "text": f"Question: {q}"}
                ]
            }
        ])

    texts = [
        processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_batch
    ]

    image_inputs, video_inputs = process_vision_info(messages_batch)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=config["max_new_tokens"], do_sample=True, temperature=0.01)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    decoded = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    answers = [text.splitlines()[0].split("Answer: ")[-1].strip()
               for text in decoded]
    
    return answers
