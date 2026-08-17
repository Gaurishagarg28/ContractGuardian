from pathlib import Path
from tempfile import TemporaryDirectory

from flask import Flask, jsonify, render_template_string, request
from src.ingestion.document_models import Clause


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ContractGuardian</title>
<style>
body{margin:0;background:#f5f7fb;color:#172033;font:16px system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:880px;margin:56px auto;padding:0 22px}.brand{color:#3268d6;font-weight:800;letter-spacing:.04em}
h1{font-size:42px;margin:10px 0}p{color:#5d6980;line-height:1.55}.card{background:#fff;border:1px solid #e2e7f1;border-radius:16px;padding:24px;margin-top:28px;box-shadow:0 8px 25px #18223b0b}
label{font-weight:700;display:block;margin-bottom:8px}textarea,input{box-sizing:border-box;width:100%;border:1px solid #cbd4e3;border-radius:9px;padding:12px;font:inherit}textarea{min-height:180px;resize:vertical}
.row{display:flex;gap:12px;margin-top:16px;align-items:center}button{background:#3268d6;color:#fff;border:0;border-radius:9px;padding:12px 18px;font-weight:700;cursor:pointer}button:disabled{opacity:.6}.hint{font-size:14px;color:#71809a}pre{display:none;white-space:pre-wrap;background:#101827;color:#ddedff;border-radius:9px;padding:16px;overflow:auto;margin-top:18px}
</style></head><body><main><div class="brand">CONTRACTGUARDIAN</div><h1>Review a contract clause</h1><p>Classify a clause and identify language that may need closer review. Results are decision support only, not legal advice.</p>
<section class="card"><label for="heading">Clause heading <span class="hint">(optional)</span></label><input id="heading" placeholder="e.g. Renewal Term"><br><br><label for="text">Clause text</label><textarea id="text" placeholder="Paste a contract clause here..."></textarea><div class="row"><button id="analyze">Analyze clause</button><span class="hint" id="status"></span></div><pre id="result"></pre></section>
<script>
const button=document.querySelector('#analyze'),status=document.querySelector('#status'),output=document.querySelector('#result');
button.onclick=async()=>{const text=document.querySelector('#text').value,heading=document.querySelector('#heading').value;if(!text.trim()){status.textContent='Enter a clause first.';return}button.disabled=true;status.textContent='Analyzing…';output.style.display='none';try{const response=await fetch('/api/analyze-text',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,heading})});const data=await response.json();if(!response.ok)throw new Error(data.error||'Analysis failed');output.textContent=JSON.stringify(data,null,2);output.style.display='block';status.textContent='Analysis complete.'}catch(error){status.textContent=error.message}finally{button.disabled=false}};
</script></main></body></html>"""


def create_contract_analyzer():
    """Delay heavyweight ML imports until an analysis request arrives."""
    from src.analysis.contract_analyzer import ContractAnalyzer

    return ContractAnalyzer()


def create_app(analyzer_factory=None):
    app = Flask(__name__)
    get_analyzer = analyzer_factory or create_contract_analyzer

    @app.get("/")
    def home():
        return render_template_string(PAGE)

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", service="ContractGuardian AI")

    @app.post("/api/analyze-text")
    def analyze_text():
        payload = request.get_json(silent=True) or {}
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            return jsonify(error="'text' must be a non-empty string."), 400
        try:
            analyzer = get_analyzer()
            clause_analyzer = getattr(analyzer, "clause_analyzer", analyzer)
            result = clause_analyzer.analyze(
                Clause("TEXT-0001", text, 1, 1, heading=payload.get("heading"))
            )
            return jsonify(result)
        except Exception as error:
            app.logger.exception("Text analysis failed")
            return jsonify(error=str(error)), 500

    @app.post("/api/analyze-pdf")
    def analyze_pdf():
        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename:
            return jsonify(error="Provide a PDF in the 'file' field."), 400
        if Path(uploaded.filename).suffix.lower() != ".pdf":
            return jsonify(error="Only PDF files are supported."), 400
        try:
            with TemporaryDirectory() as directory:
                pdf_path = Path(directory) / "contract.pdf"
                uploaded.save(pdf_path)
                return jsonify(get_analyzer().analyze_pdf(pdf_path))
        except Exception as error:
            app.logger.exception("PDF analysis failed")
            return jsonify(error=str(error)), 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)