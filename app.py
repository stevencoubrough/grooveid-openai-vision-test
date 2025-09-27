"""
FastAPI server for testing OpenAI's vision API with record covers.

This minimal example exposes a single endpoint, `/identify`, that accepts an
uploaded image and uses OpenAI's vision model to generate a description.

The OpenAI API client is imported only if available. You must set the
environment variable `OPENAI_API_KEY` with your API key for this code to work.

Note: This example is a skeleton intended for demonstration purposes. The
assistant cannot execute calls to OpenAI's API on your behalf, so the
`identify` endpoint returns a placeholder response if the `openai` module is
not installed or no API key is configured.
"""

import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from fastapi.responses import FileResponse
from pathlib import Path

app = FastAPI(title="OpenAI Vision Test API")

# Allow CORS so the frontend (served from another domain) can call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    import openai  # type: ignore
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None  # type: ignore
    OPENAI_AVAILABLE = False


@app.post("/identify")
async def identify(request: Request):
    """
    Identify a record cover using OpenAI's vision model.

    The client should POST a JSON payload with a single field `image_base64` which
    contains the image encoded as a base64 string. This avoids the need for
    `python-multipart` while still allowing binary image data to be sent.

    The endpoint returns a JSON object with a `description` field containing
    the model's description of the image, or a stub message if the OpenAI
    library or API key are not available.
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")
    b64_image = payload.get("image_base64")
    if not b64_image:
        raise HTTPException(status_code=400, detail="Missing 'image_base64' in request body.")
    import base64
    try:
        contents = base64.b64decode(b64_image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data.")

    # If the openai library or API key is unavailable, return a stub response
    if not OPENAI_AVAILABLE or not os.getenv("OPENAI_API_KEY"):
        return JSONResponse(
            {
                "description": (
                    "OpenAI API key not configured or openai library not installed. "
                    "Please install the openai Python package and set the OPENAI_API_KEY "
                    "environment variable to enable vision predictions."
                )
            }
        )

    # Prepare the request to OpenAI's vision model
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant that describes record covers succinctly."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this record cover. Include any readable text."
                        },
                        {
                            "type": "image",
                            "image": contents
                        }
                    ],
                },
            ],
            max_tokens=200,
            temperature=0.2,
        )
        description = response.choices[0].message["content"].strip()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse({"description": description})


@app.get("/")
async def serve_frontend():
    """Serve the test HTML page.

    When visiting the root of the server in a browser, this endpoint returns the
    `index.html` file from the `templates` directory. The file is served
    directly using a `FileResponse` to keep the example self-contained.
    """
    file_path = Path(__file__).parent / "templates" / "index.html"
    return FileResponse(file_path)
