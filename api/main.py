from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types
import os
import json

# Инициализация клиента
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

app = FastAPI()

# --- СПИСКИ МОДЕЛЕЙ (в порядке приоритета) ---
# Для фото: сначала мощная быстрая, потом легкая, потом старая стабильная
MODELS_IMAGE = ['gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemini-1.5-flash']
# Для аудио: 2.5 flash лучше всего работает с нативным аудио
MODELS_AUDIO = ['gemini-2.5-flash', 'gemini-1.5-flash']
# Для чата: Gemma для инструкций или Flash
MODELS_CHAT = ['gemma-2-27b-it', 'gemini-2.5-flash']


# Модели данных
class AnalyzeRequest(BaseModel):
    image_base64: str | None = None
    audio_base64: str | None = None
    mime_type: str | None = None
    clarification: str | None = None


class ChatRequest(BaseModel):
    message: str
    context: dict


# Схема ответа
food_item_schema = {
    "type": "OBJECT",
    "properties": {
        "dish_name": {"type": "STRING"},
        "calories": {"type": "INTEGER"},
        "proteins": {"type": "INTEGER"},
        "fats": {"type": "INTEGER"},
        "carbs": {"type": "INTEGER"},
    },
    "required": ["dish_name", "calories", "proteins", "fats", "carbs"],
}


# --- ФУНКЦИЯ FALLBACK ---
def generate_with_fallback(models_list, contents, config):
    last_error = None
    for model in models_list:
        try:
            print(f"Trying model: {model}...")
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
            return response
        except Exception as e:
            print(f"Model {model} failed: {e}")
            last_error = e
            # Продолжаем цикл к следующей модели

    # Если все модели упали
    raise last_error


@app.post("/analyze-image")
async def analyze_image(req: AnalyzeRequest):
    try:
        prompt = "Распознай еду на фото. Оцени размер порции и верни пищевую ценность в JSON."
        if req.clarification:
            prompt = f'Предыдущий анализ мог быть неточным. Пользователь уточнил: "{req.clarification}". Пересчитай БЖУ и название.'

        response = generate_with_fallback(
            models_list=MODELS_IMAGE,
            contents=[
                types.Part.from_bytes(data=req.image_base64, mime_type=req.mime_type),
                prompt
            ],
            config=types.GenerateContentConfig(
                system_instruction="Ты — диетолог. Возвращай строго валидный JSON.",
                response_mime_type="application/json",
                response_schema=food_item_schema
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"All models failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-audio")
async def analyze_audio(req: AnalyzeRequest):
    try:
        prompt = "Послушай описание еды. Определи блюдо и БЖУ."
        if req.clarification:
            prompt += f' Уточнение: "{req.clarification}".'

        response = generate_with_fallback(
            models_list=MODELS_AUDIO,
            contents=[
                types.Part.from_bytes(data=req.audio_base64, mime_type=req.mime_type),
                prompt
            ],
            config=types.GenerateContentConfig(
                system_instruction="Ты — диетолог. Анализируй аудио. Возвращай JSON.",
                response_mime_type="application/json",
                response_schema=food_item_schema
            )
        )
        return json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat-coach")
async def chat_coach(req: ChatRequest):
    try:
        prompt = f"User query: {req.message}\nContext: {json.dumps(req.context, ensure_ascii=False)}"

        response = generate_with_fallback(
            models_list=MODELS_CHAT,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="Ты краткий AI-тренер. Давай советы на основе данных. Отвечай на русском. Лимит - 2 абзаца.",
                temperature=0.7
            )
        )
        return {"text": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


export_app = app