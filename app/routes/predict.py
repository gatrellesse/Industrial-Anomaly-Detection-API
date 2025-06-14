# api/predict.py

from fastapi import APIRouter, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from PIL import Image
import io
from app.services.inference import Detector

router = APIRouter()

@router.post("/load_model")
def load_model(model_path: str, request: Request):
    try:
        request.app.state.model = Detector(model_path)
        return {"message": "Model loaded successfully"}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@router.post("/predict")
async def predict_anomaly(request: Request, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid image file"})

    model = getattr(request.app.state, "model", None)
    if model is None:
        return JSONResponse(status_code=400, content={"error": "Model not loaded yet"})

    try:
        results = model.inference(image)
         # Get all detections
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    'class': result.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'box': box.xyxy[0].tolist()  # Convert tensor to list
                })
        
        # Find detection with highest confidence
        if not detections:
            return {"label": None, "confidence": 0.0}
        
        best_detection = max(detections, key=lambda x: x['confidence'])
        return jsonable_encoder({
            "label": best_detection['class'],
            "confidence": best_detection['confidence']
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
