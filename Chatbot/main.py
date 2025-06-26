from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Business Chat Assistant API is running"}
