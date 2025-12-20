
import uvicorn
import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

MODEL_PATH = "Qwen3-4b-it-test-VietMedQA.Q4_K_M.gguf"
llm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm
    if os.path.exists(MODEL_PATH):
        print(f"Đang load model GGUF từ: {MODEL_PATH}")
        try:
            llm = Llama(
                model_path=MODEL_PATH,
                verbose=False,
            )
            print("Load Model thành công!")
        except Exception as e:
            print(f"Lỗi load model: {e}")
    else:
        print(f"Không tìm thấy file model, bắt đầu tải từ hugging face")
        hf_hub_download(repo_id="NgVietAnh41/Qwen3-4b-it-VietMedQA-DPO-gguf",
        filename="Qwen3-4b-it-test-VietMedQA.Q4_K_M.gguf",
        local_dir="",
        )
        llm = Llama(
                model_path=MODEL_PATH,
                verbose=False,
            )
        print("Load Model thành công!")
        
    yield 
    print("Đang dọn dẹp tài nguyên...")
    if llm:
        del llm
    print("👋 Server đã tắt!")
    
app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

async def generate_response(user_prompt: str):
    if llm is None:
        raise ValueError("Khởi tạo lỗi")
    try:
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=2048,
            stream=True,
            temperature=0.2,
        )
        for chunk in response:
            choices = chunk.get("choices", [])
            if choices:
                choice = choices[0]
                content = choice.get("delta", {}).get("content")
                if content:
                    yield content
    except Exception as e:
        print(f"Lỗi khi gọi mô hình: {e}")
        yield "Lỗi khi xử lý yêu cầu từ mô hình."

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat_endpoint(prompt: str = Form(...)):
    print(f"Nhận câu hỏi: {prompt}")
    try:
        return StreamingResponse(generate_response(prompt), media_type="text/plain")
    except Exception as e:
        print(f"Lỗi khi xử lý yêu cầu: {e}")
        return JSONResponse({"error": "Đã xảy ra lỗi khi xử lý yêu cầu."}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)