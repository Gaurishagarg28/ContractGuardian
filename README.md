# ContractGuardian

ContractGuardian extracts clauses from PDF contracts, classifies them with a
local ML model, and identifies rule-based risk indicators.

## Run the API

```powershell
.\venv\Scripts\python.exe app.py
```

The API is available at `http://127.0.0.1:5000`.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Service health check |
| `POST /api/analyze-text` | Analyze JSON text: `{ "text": "...", "heading": "optional" }` |
| `POST /api/analyze-pdf` | Analyze a PDF uploaded in the multipart `file` field |

Set `CONTRACTGUARDIAN_DEVICE=cpu` to force CPU inference. The default `auto`
uses CUDA when available and falls back to CPU if GPU memory is unavailable.

## Verify

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -p test_api.py
.\venv\Scripts\python.exe -m unittest discover -s tests -p test_text_risk.py
```
