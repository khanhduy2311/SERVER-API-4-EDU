from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import huggingface_hub
from huggingface_hub import snapshot_download

snapshot_download(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", repo_type="model")
def get_qa_pipeline():
    """
    Khởi tạo và trả về mô hình LLaMA 8B cho QA.
    Chạy trên GPU nếu có.
    """
    model_name = " "  # hoặc model LLaMA 8B khác

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[QA] Loading model on {'GPU' if device.type == 'cuda' else 'CPU'}...")

    # Load tokenizer và model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    def qa_function(question, context, max_new_tokens=256):
        prompt = f"""[INST] <<SYS>>
You are a helpful and knowledgeable assistant.
<</SYS>>

Context: {context}
Question: {question}
Answer:"""
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Trích xuất phần "Answer: ..." nếu có
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()

        return answer

    return qa_function

# Tạo instance duy nhất
hf_token = "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"

qa = get_qa_pipeline(hf_token)
