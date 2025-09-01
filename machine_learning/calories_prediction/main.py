import joblib

model=joblib.load('calories_model.pkl')

from fastapi import FastAPI,Request,Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import numpy as np
app=FastAPI()

template_folder=Jinja2Templates(directory="template")
@app.get("/calories_calculator",response_class=HTMLResponse)
async def calories_prediction(request:Request) :
    return template_folder.TemplateResponse("index.html",{'request':request})

@app.post("/get_calories_calculator", response_class=HTMLResponse)
async def output_prediction(request: Request,heartRate:str=Form(...),bodyTemp:str=Form(...),duration:str=Form(...)):
    
	features = np.array([int(heartRate), float(bodyTemp), int(duration)]).reshape(1, -1)
	prediction = model.predict(features)
    
	return template_folder.TemplateResponse(
        "index.html",
        {
            "request": request,
            "heartRate": heartRate,
            "bodyTemp": bodyTemp,
            "duration": duration,
            "prediction": round(float(prediction[0]), 2),
        })

import uvicorn

if __name__ == "__main__":
	uvicorn.run(app, host="127.0.0.1", port=8000)

