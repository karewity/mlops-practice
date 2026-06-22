from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import joblib
import os

# 1. Определяем абсолютные пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model.pkl")

# 2. Загружаем модель ОДИН раз при старте сервера
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Iris ML API", description="API для предсказания сорта Ириса")


# 3. Редирект на документацию
@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")


# 4. Контракт данных (Pydantic)
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# 5. Эндпоинт проверки здоровья
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Server is running"}


# 6. Эндпоинт для предсказания
@app.post("/predict")
def predict(features: IrisFeatures):
    data = [
        [
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width,
        ]
    ]
    prediction = model.predict(data)
    return {"predicted_class": int(prediction[0])}
