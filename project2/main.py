# from fastapi import FastAPI, Form, Request
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles
# import joblib
# import os  
# import uvicorn

# # This will get the absolute path of the directory where main.py is located
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # Load the model and vectorizer
# model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
# vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))

# app = FastAPI()
# templates = Jinja2Templates(directory="templates")

# # Optional: serve static files like CSS
# app.mount("/static", StaticFiles(directory="static"), name="static")

# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/predict", response_class=HTMLResponse)
# async def predict(request: Request, text: str = Form(...)):
#     try:
#         transformed_text = vectorizer.transform([text])  # Transform input text
#         prediction = model.predict(transformed_text)[0]  # Predict label
#         result = "Fake News" if prediction == 1 else "Real News"
#     except Exception as e:
#         result = f"Error: {str(e)}"
#     return templates.TemplateResponse("index.html", {"request": request, "prediction": result, "request": request})


from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import os
import uvicorn

# Get the absolute path of the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the full pipeline (vectorizer + model)
pipeline = joblib.load(os.path.join(BASE_DIR, "pipeline.pkl"))

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# @app.get("/")
# def home():
#     return {"message": "Hello! The app is running."}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):
    try:
        prediction = pipeline.predict([text])[0]
        result = "Fake News" if prediction == 1 else "Real News"
    except Exception as e:
        result = f"Error: {str(e)}"
    return templates.TemplateResponse("index.html", {"request": request, "prediction": result})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=False)

