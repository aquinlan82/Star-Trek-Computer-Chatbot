print("server.py starting...")


from dotenv import load_dotenv
load_dotenv() #for local development

#fast api specific imports
from fastapi import FastAPI
from fastapi.concurrency import iterate_in_threadpool
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse, RedirectResponse
import boto3, botocore

#handle logging
import logging, sys
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

#disable logging for health checks
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.args and len(record.args) >= 3:
            to_ignore = ["/health","/assets","/favicon.ico","/alba.png","/joe.png"]
            path = record.args[2]
            return not any([path.startswith(x) for x in to_ignore])
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

#other imports
import datetime
from typing import Dict, List
import threading
import time
from contextlib import asynccontextmanager

from assets.util import *

start_date = datetime.datetime.now()



#-----------------FASTAPI CONFIGURATION-----------------#

app = FastAPI()

origins = [
    "http://localhost:5173",
    "localhost:5173"
]


app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],  # Allows all origins
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

#-----------------API ENDPOINTS-----------------#

@app.get("/health")
async def health():
    return Response(content="OK", status_code=200)
    
@app.get("/startDate")
async def get_start_date():
    return Response(content=str(start_date), status_code=200)

@app.post("/chat")
async def chat(dataIn: Dict):
    print("data in", str(dataIn)[:300]+"...")
    
    history = dataIn["history"]
    filters = dataIn["filters"]
    response_generator = iterate_in_threadpool(make_resp(history, filters))
    return StreamingResponse(response_generator, media_type="application/json")

@app.post("/debug_data")
async def debug_data(dataIn: Dict):
    filters = dataIn["filters"]
    sources = get_debug_data(filters)
    return sources

@app.post("/issue")
async def issue(data: Dict): 
    # print("received issue", data)
    e = send_issue(data)

    if e == "OK":
        return Response(content="Issue sent successfully", status_code=200)
    else:
        return Response(content=e, status_code=500)
    
@app.post("/create_summary")
async def create_summary(data: Dict): 
    # print("received export", data)
    summary = extract_summary(data)
    return summary


# open reference document in *prod* (need aws creds to be set to prod to work)
# http://localhost:8000/documents/document_pool/480096_Vorstellung.pdf        test link
# http://localhost:8000/documents/document_pool/Arguments_Example_800.pdf      test link
@app.get("/documents/document_pool/{filename:path}")  
async def documents(filename: str):
    session = boto3.Session()

    s3 = session.client('s3',config=botocore.client.Config(
                    s3={'use_accelerate_endpoint': True}))
    
    print("filename", filename)
    if filename == "Arguments_Example_800.pdf":
        response_type = "text/plain"
        filename = "Arguments_Example_800.csv"
    else:
        response_type = "application/pdf"

    params = {
        'Bucket': "sales-and-marketing-818945945684" ,
        'Key': f'documents/document_pool/{filename}',
        "ResponseContentType": response_type}
    url = s3.generate_presigned_url(
        ClientMethod='get_object',
        Params=params,
        ExpiresIn=600
    )
    return RedirectResponse(url=url)

app.mount("/", StaticFiles(directory="dist", html=True), name="dist")
