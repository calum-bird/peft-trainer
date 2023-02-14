import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig


print("Loading PEFT adapter...")
peft_model_id = "../models/codegen-6B-multi-lora-abirate-english-quotes"
config = PeftConfig.from_pretrained(peft_model_id)

print("Loading full model...")
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-6B-multi")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

print("Sending model to cuda...")
model.cuda()
model.eval()

print("Generating text...")
wants_exit = False
while not wants_exit:
    prompt = input("Enter the start of a quote: ")
    batch = tokenizer(prompt, return_tensors='pt').to('cuda')
    with torch.cuda.amp.autocast():
        output_tokens = model.generate(**batch, max_new_tokens=50)

    print(tokenizer.decode(output_tokens[0], skip_special_tokens=True), '\n\n')
