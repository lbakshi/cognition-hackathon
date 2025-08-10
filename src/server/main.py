from fastapi import FastAPI
import os

app = FastAPI()


@app.get("/api/hello")
async def read_hello():
    return {"message": "Hello from server!"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=port)
