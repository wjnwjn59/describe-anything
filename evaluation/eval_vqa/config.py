ds_collections = {
    "vqav2_val": {
        "data_path": "/mnt/VLAI_data/VQAv2/vqav2_val.jsonl",
        "image_folder": "/mnt/VLAI_data/COCO_Images/val2014/",
        "prompt": "You are a Visual Question Answering (VQA) system. Use only the information visible in the image.\nAnswer each question with a single word or short phrase whenever possible.\nAlways use exactly this output format, with no extra text: Answer: <your concise answer>\n\nExamples:\nQuestion: What is the man doing?\nAnswer: skiing\n\nQuestion: What material is the table made of?\nAnswer: wood\n\nQuestion: Which animal is shown in the picture?\nAnswer: giraffe\n\nQuestion: What color is the car?\nAnswer: red\n\nQuestion: How many people are there?\nAnswer: two",
        "metric": "accuracy",
        "max_new_tokens": 10,
    },
    "infographicvqa_val": {
        "data_path": "/mnt/VLAI_data/InfographicVQA/infographicvqa_val.jsonl",
        "image_folder": "/mnt/VLAI_data/InfographicVQA/images",
        "prompt": "You are an AI assistant specialized in analyzing images and answering questions based on visual content.\nAnswer each question with a single word or short phrase whenever possible.\nOnly provide the answer without any explanations or extra information.\nOutput format:\nAnswer: <your answer>\n\nExamples:\nQuestion: What percentage of the population is affected by this disease?\nAnswer: 12%\n\nQuestion: How many products were sold in the first 3 months?\nAnswer: 1103\n\nQuestion: Which subject is the student studying, math or physics?\nAnswer: physics\n\nQuestion: Who founded this company?\nAnswer: Steve Jobs",
        "metric": "anls",
        "max_new_tokens": 100
    },
} 