# GrooveID – Vision → Discogs Resolver

Photo in → Discogs link out (stateless).  
Uses OpenAI Vision to extract text/visual cues, then Google Programmable Search (CSE) with Discogs filtering, and an optional visual similarity re-check.

## Endpoints
- **GET /health** – sanity check environment keys.
- **POST /identify** – identify a record
  - Form field: `file` (JPG/PNG/WEBP)
  - Query parameter: `max_candidates` (default 8)
  - Query parameter: `do_visual_check` (default true)

## Running locally
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export $(cat .env | xargs)  # set your API keys
uvicorn app.main:app --reload --port 8000
```
Open Swagger UI at `http://127.0.0.1:8000/docs` to try it out.

## Deploying on Render
1. Create a **Web Service** pointing to this repo.
2. Set the following environment variables:
   - `OPENAI_API_KEY` – your OpenAI API key with vision access.
   - `GOOGLE_API_KEY` – Google Custom Search API key.
   - `GOOGLE_CSE_ID` – Search engine ID from Programmable Search.
3. Leave the build command blank (Render auto-detects Python).
4. Start command:  
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 10000
   ```
5. Once deployed, visit `/health` to verify the env vars are loaded.

## Notes
- If OpenAI returns non-JSON or a model isn’t available to your key, the service logs an error and degrades gracefully (check Render logs).
- If Google CSE fails (wrong API key, not enabled, or quota exceeded) you’ll get a 502 with a log entry `[google] FAILED`.
- To tighten security, restrict CORS origins in `app/main.py` and consider adding rate limiting.
