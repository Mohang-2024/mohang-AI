from fastapi import FastAPI, Request
from pydantic import BaseModel
import transformers
import torch

app = FastAPI()

model_id = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'


pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype":torch.bfloat16},
    device_map="auto"
)

class TextGenerationRequset(BaseModel):
    prompt : str

@app.post('/generation/')
async def generate_text(requset: TextGenerationRequset):
    input_text = requset.prompt
    
    
    outputs = pipeline(
        input_text,
        max_new_tokens = 200,
        eos_token_id = pipeline.tokenizer.eos_token_id,
        do_sample=True,
        temperature = 0.6,
        top_p=0.9
    )
    
    generated_text = outputs[0]['generated_text']
    
    return {"generated_text":generated_text}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
        