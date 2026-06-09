import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tiktoken

app = FastAPI(
    title="Prompt Compressor API",
    description="Lightweight API to optimize prompts and save OpenAI context tokens.",
    version="1.0.0"
)

# Initialize OpenAI tokenizer
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    tokenizer = tiktoken.get_encoding("gpt2")

class CompressionRequest(BaseModel):
    text: str

class CompressionResponse(BaseModel):
    original_token_count: int
    compressed_token_count: int
    tokens_saved: int
    savings_percentage: str
    compressed_text: str

def advanced_lightweight_compressor(text: str) -> str:
    if not text.strip():
        return ""

    # 1. Remove polite filler & LLM preambles
    fillers = [
        r"\b(please|kindly|could you|would you mind|go ahead and|take a look at and)\b",
        r"\b(it is important to note that|bear in mind that|keep in mind that|note that)\b",
        r"\b(as you know|as mentioned before|in my opinion|personally speaking)\b",
        r"\b(i want you to|your task is to dynamically|basically|essentially)\b"
    ]
    for pattern in fillers:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # 2. Substitute wordy phrases with dense equivalents
    substitutions = {
        "in order to": "to",
        "for the purpose of": "for",
        "due to the fact that": "because",
        "at the present time": "now",
        "prior to": "before",
        "subsequent to": "after",
        "with regard to": "about",
        "as a matter of fact": "factually",
        "utilize": "use",
        "straight away": "now",
        "concerning the matter of": "on"
    }
    
    for word, replacement in substitutions.items():
        pattern = r'\b' + re.escape(word) + r'\b'
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # 3. Normalize whitespace and newlines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    
    return text.strip()

@app.post("/compress", response_model=CompressionResponse)
async def compress_endpoint(payload: CompressionRequest):
    if not payload.text:
        raise HTTPException(status_code=400, detail="Text prompt cannot be empty.")
    
    original_text = payload.text
    compressed_text = advanced_lightweight_compressor(original_text)
    
    orig_tokens = len(tokenizer.encode(original_text))
    comp_tokens = len(tokenizer.encode(compressed_text))
    tokens_saved = orig_tokens - comp_tokens
    
    savings_pct = (tokens_saved / orig_tokens * 100) if orig_tokens > 0 else 0.0
    
    return CompressionResponse(
        original_token_count=orig_tokens,
        compressed_token_count=comp_tokens,
        tokens_saved=tokens_saved,
        savings_percentage=f"{savings_pct:.1f}%",
        compressed_text=compressed_text
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
