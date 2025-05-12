import os
import json
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import numpy as np
import argparse
from dotenv import load_dotenv
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator
from haystack.components.builders import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document
from haystack import Pipeline
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_APIKEY")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
orig_stack = np.stack


def patched_stack(arrays, axis=0, *args, **kwargs):
    if "dtype" in kwargs:
        kwargs.pop("dtype")
    return orig_stack(arrays, axis=axis, *args, **kwargs)


np.stack = patched_stack


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rag(question: str, additional_info: str) -> str:
    """
    Perform RAG summarization on additional information.

    Args:
        question (str): The visual question regarding the image.
        additional_info (str): Additional information about the image.

    Returns:
        str: The summarized additional information.
    """
    template = """
Tóm tắt các thông tin liên quan đến câu hỏi từ đoạn văn sau. Không trả lời câu hỏi.

Ngữ cảnh:
{% for document in documents %}
{{ document.content }}
{% endfor %}

Câu hỏi: {{ question }}
"""
    docs = [Document(content=additional_info)]
    docstore = InMemoryDocumentStore()
    docstore.write_documents(docs)
    pipe = Pipeline()

    pipe.add_component(
        "retriever", InMemoryBM25Retriever(document_store=docstore))
    pipe.add_component("prompt_builder", PromptBuilder(
        template=template, required_variables=["question"]))
    pipe.add_component("llm", GoogleAIGeminiGenerator(
        model="gemini-2.0-flash"))
    pipe.connect("retriever", "prompt_builder.documents")
    pipe.connect("prompt_builder", "llm")
    response = pipe.run({"retriever": {"query": question},
                         "prompt_builder": {"question": question}})
    print(response["llm"]["replies"][0])
    return response["llm"]["replies"][0]


def inference(question: str, image_path: str, additional_info: str = None) -> str:
    """
    Perform inference on an image-question pair using the Deepseek-VL2 model.

    Args:
        question (str): The visual question regarding the image.
        image_path (str): The file path to the input image.
        additional_info (str, optional): Additional information about the image. Defaults to None.

    Returns:
        str: The model's answer.
    """

    system_instruction = (
        f"""Bạn là một trợ lý AI chuyên phân tích hình ảnh và trả lời câu hỏi dựa trên hình ảnh (VQA) một cách chính xác. Nhiệm vụ của bạn là:
    - Phân tích hình ảnh{' và mô tả chi tiết kèm theo' if additional_info else ''}.
    - Chỉ trả lời câu hỏi mà không giải thích thêm bất kỳ thông tin nào khác.

    Định dạng bắt buộc:
    Trả lời: <Câu trả lời của bạn>

    Ví dụ:
    Câu hỏi: Bao nhiêu phần trăm dân số bị ảnh hưởng bởi căn bệnh này?
    Trả lời: 12%
    Câu hỏi: Tổng số sản phẩm bán ra trong 3 tháng đầu năm là bao nhiêu?
    Trả lời: 1103
    Câu hỏi: Học sinh đang học môn nào, toán hay vật lí?
    Trả lời: Vật lí
    Câu hỏi: Ai là người sáng lập công ty này?
    Trả lời: Steve Jobs

    Hãy dựa trên hình ảnh{', mô tả chi tiết' if additional_info else ''} và câu hỏi để đưa ra câu trả lời ngắn gọn nhất.""")

    if additional_info:
        user_content = f"Đây là mô tả chi tiết của bức hình:\n{additional_info}\nCâu hỏi: {question}"
    else:
        user_content = f"Câu hỏi: {question}"
    prompt = system_instruction + user_content

    image = Image.open(image_path).convert("RGB")
    inputs = processor.process(images=[image], text=prompt)

    # Move inputs to the correct device and add batch dimension
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    inputs["images"] = inputs["images"].to(torch.bfloat16)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
            output = model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=50,
                                 stop_strings="<|endoftext|>"),
                tokenizer=processor.tokenizer
            )

    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(
        generated_tokens, skip_special_tokens=True)

    answer = generated_text.split("Trả lời:")[-1].strip()
    return answer


set_seed(42)
model_name = "allenai/Molmo-7B-D-0924"
processor = AutoProcessor.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map='auto',
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
).eval().cuda()

parser = argparse.ArgumentParser()
parser.add_argument("--use_pipeline", action="store_true")
parser.add_argument("--info_dir", type=str, default="../output_pipeline",
                    help="Directory of raw additional info files")
parser.add_argument("--image_folder", type=str,
                    default="/mnt/VLAI_data/ViInfographicVQA/sample_images", help="Directory of input images")
parser.add_argument("--data_path", type=str,
                    default="/mnt/VLAI_data/ViInfographicVQA/100_samples.json", help="JSON file of sample questions")
args = parser.parse_args()

with open(args.data_path, encoding='utf-8') as f:
    data = json.load(f)

predicts, gts = [], []
for item in tqdm(data):
    img_path = os.path.join(args.image_folder, item['source_image'])
    additional_info = None

    if args.use_pipeline:
        base_name = os.path.splitext(item['source_image'])[0]
        info_path = os.path.join(args.info_dir, base_name + ".txt")
        with open(info_path, encoding='utf-8') as finfo:
            raw_info = finfo.read().strip()
        additional_info = rag(item['question'], raw_info)

    answer = inference(item['question'], img_path,
                       additional_info=additional_info)
    print(f"Question: {item['question']}")
    print(f"Summary: {additional_info}")
    print(f"Predicted: {answer}, GT: {item['answer']}")
    predicts.append(answer)
    gts.append(item['answer'])
    time.sleep(2)

output_file = f"{model_name.split('/')[-1]}_{'pipeline' if args.use_pipeline else 'normal'}.jsonl"
with open(output_file, 'w', encoding='utf-8') as f:
    for pred, gt in zip(predicts, gts):
        f.write(json.dumps({'predict': pred, 'gt': gt},
                ensure_ascii=False) + "\n")
