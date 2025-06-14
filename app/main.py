from fastapi import FastAPI
from .routes.predict import router as predict_router


app = FastAPI(title="Industrial Anomaly Detection API")

# Mount each router with a URL prefix and a documentation tag
app.include_router(predict_router, prefix="/predict", tags=["Prediction"])

