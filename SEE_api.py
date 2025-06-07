# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("SEE_api")

# Create input/output pydantic models
input_model = create_model("SEE_api_input", **{'prec': 2.4800000190734863, 'flex': 2.0299999713897705, 'resl': 2.8299999237060547, 'team': 1.100000023841858, 'pmat': 3.119999885559082, 'rely': 1.100000023841858, 'data': 0.8999999761581421, 'cplx': 1.1699999570846558, 'ruse': 1.0, 'docu': 1.0, 'time': 1.0, 'stor': 1.0, 'pvol': 0.7799999713897705, 'acap': 1.0, 'pcap': 1.0, 'pcon': 1.0, 'apex': 1.0, 'plex': 1.0, 'ltex': 0.9100000262260437, 'tool': 1.0, 'site': 1.0, 'sced': 1.1399999856948853, 'kloc': 66.5999984741211, 'Team Size ': 1.149999976158142, 'Lack of Team Trust': 1.149999976158142, 'Competence Level': 0.8500000238418579, 'Client Involvement ': 0.8500000238418579, 'Geographic distribution ': 1.149999976158142, 'Knowledge Management ': 0.8500000238418579, 'Project Effort': 1.149999976158142, 'Design and technology newness': 1.149999976158142, 'Communication Infrastructure ': 0.8500000238418579, 'Work Dispersion': 0.8500000238418579, 'Response Delay': 1.149999976158142, 'Task Allocation': 0.8500000238418579, 'Travel Cost': 1.149999976158142, 'Time zone difference': 1.149999976158142, 'Language/Cultural difference': 1.149999976158142, 'Project Management Effort': 1.149999976158142, 'Contract Design ': 0.8500000238418579, 'Rework': 1.149999976158142, 'Requirements Legibility': 0.8500000238418579, 'Sharing of Resources': 0.8500000238418579, 'Development Productivity': 0.8500000238418579, 'Reuse': 0.8500000238418579, 'Process Compliance ': 0.8500000238418579, 'Process Maturity': 0.8500000238418579, 'Process Model': 0.8500000238418579, 'Work Pressure': 1.149999976158142})
output_model = create_model("SEE_api_output", prediction=300.0)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
