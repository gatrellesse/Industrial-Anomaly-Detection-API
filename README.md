# Industrial-Anomaly-Detection-API

ml_api/
├── app/
│   ├── main.py             # FastAPI instance
│   ├── api/                # Routers
│   │   ├── predict.py
│   │   ├── auth.py
│   │   └── health.py
│   ├── core/
│   │   ├── config.py       # Settings
│   ├── models/             # ML model files
│   ├── services/
│   │   └── inference.py    # Model prediction logic
│   ├── utils/
│   │   └── image_utils.py
│   ├── database/           # Heroku or Render
├── tests/                  # API tests
├── requirements.txt
├── Dockerfile              # For deployment 
└── .env                    # Environment variables

# run from app

fastapi dev main.py

# Idea

Set of imgs --> Detects anomaly --> Saves frame --> Save img id in a database

https://www.mvtec.com/company/research/datasets

endpoint --> inference of a video, get imgs, get img ids, get metrics of the batch(anomalies_frame/total_frame)