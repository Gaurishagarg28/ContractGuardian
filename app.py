import os
import threading
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from flask import Flask, jsonify, render_template_string, request, send_file
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


LAW_RULES = (
    ("Indian Contract Act, 1872, Section 27", ("non-compete", "not compete", "restraint of trade"), "A restraint on carrying on a lawful trade or business may be void to that extent, subject to statutory exceptions.", "Narrow the restriction to legitimate confidential-information and customer protections, with a defined scope, duration, and geography.", "https://www.indiacode.nic.in/show-data?actid=AC_CEN_3_20_00035_187209_1523268996428&orderno=28&orgactid=AC_CEN_3_20_00035_187209_1523268996428&statehandle=123456789%2F1362"),
    ("Indian Contract Act, 1872, Section 74", ("liquidated damages", "penalty", "penal sum"), "A stipulated amount for breach is subject to the statutory standard of reasonable compensation, not exceeding the amount stated.", "Describe the amount as a reasonable estimate of likely loss, set a proportionate cap, and remove punitive or automatic-payment wording.", "https://www.indiacode.nic.in/show-data?actid=AC_CEN_3_20_00035_187209_1523268996428&orderno=75&sectionId=38678&sectionno=74"),
    ("Indian Contract Act, 1872, Section 124", ("indemnify", "indemnification", "hold harmless"), "Section 124 defines a contract of indemnity as a promise to save another from specified loss.", "Limit indemnity to specified third-party claims caused by the indemnifying party, include defence control, and address notice and settlement consent.", "https://www.indiacode.nic.in/show-data?actid=AC_CEN_3_20_00035_187209_1523268996428&orderno=125&sectionId=42837&sectionno=124"),
    ("Indian Contract Act, 1872, Section 28", ("waive the right to sue", "shall not bring any action", "no legal proceedings"), "Certain absolute restrictions on enforcing contractual rights through ordinary legal proceedings are void to that extent; exceptions may apply.", "Use a clear dispute-resolution clause that preserves lawful remedies instead of an absolute restriction on legal proceedings.", "https://www.indiacode.nic.in/show-data?actid=AC_CEN_3_20_00035_187209_1523268996428&orderno=29&sectionId=38632&sectionno=28"),
    ("Indian Contract Act, 1872, Section 29", ("sole discretion", "as determined solely", "from time to time"), "Agreements whose meaning is not certain, or cannot be made certain, are void.", "Define objective criteria, notice, timing, and commercial terms instead of relying on open-ended discretion.", "https://www.indiacode.nic.in/show-data?actid=AC_CEN_3_20_00035_187209_1523268996428&orderno=30&sectionId=38633&sectionno=29"),
)
BASE_RISK = {"Uncapped Liability": 90, "Irrevocable Or Perpetual License": 85, "Non-Compete": 75, "Exclusivity": 70, "Liquidated Damages": 60, "Indemnity": 55, "Renewal Term": 40}


def review_clause(clause):
    text = clause.get("text", "").lower()
    laws = [{"section": section, "summary": summary, "suggestion": suggestion, "url": url} for section, terms, summary, suggestion, url in LAW_RULES if any(term in text for term in terms)]
    resolved = clause.get("resolved_classification", {})
    label = resolved.get("final_prediction") or clause.get("classification", {}).get("predicted_clause", "Unknown")
    confidence = float(resolved.get("confidence", clause.get("classification", {}).get("confidence", 0)))
    text_points = int(clause.get("risk", {}).get("risk_points", 0))
    score = round(min(100, .60 * BASE_RISK.get(label, 30) + .40 * (text_points / 60 * 100) + 8 * len(laws)))
    severity = "High" if score >= 65 else "Medium" if score >= 35 else "Low"
    clause["review"] = {"risk_score": score, "severity": severity, "confidence": confidence, "laws": laws, "suggestion": " ".join(item["suggestion"] for item in laws) or "No targeted revision was generated. Review the commercial scope, timing, and allocation of risk with qualified counsel."}
    return clause


def make_report(result):
    stream = BytesIO(); doc = SimpleDocTemplate(stream, pagesize=A4); styles = getSampleStyleSheet()
    content = [Paragraph("ContractGuardian Contract Review", styles["Title"]), Paragraph("Decision-support output only. This report is not legal advice or a legal opinion.", styles["BodyText"]), Spacer(1, 12)]
    flagged = 0
    for clause in result["clauses"]:
        item = clause.get("review")
        if clause.get("status") != "success" or not item or item["severity"] == "Low": continue
        flagged += 1; content += [Paragraph(f"{clause['clause_id']} — {item['severity']} risk ({item['risk_score']}/100)", styles["Heading2"]), Paragraph(f"Model confidence: {item['confidence']:.0%}", styles["BodyText"]), Paragraph(item["suggestion"], styles["BodyText"])]
        for law in item["laws"]: content.append(Paragraph(f"{law['section']}: {law['summary']}  {law['url']}", styles["BodyText"]))
        content.append(Spacer(1, 10))
    if not flagged: content.append(Paragraph("No medium- or high-risk clauses were automatically flagged. Human review remains necessary.", styles["BodyText"]))
    doc.build(content); stream.seek(0); return stream.getvalue()


PAGE = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>ContractGuardian</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f7fb;color:#172033;font:16px system-ui,-apple-system,Segoe UI,sans-serif}main{max-width:980px;margin:0 auto;padding:52px 20px}.brand{font-size:13px;letter-spacing:.14em;font-weight:800;color:#315fc4}.hero{display:grid;grid-template-columns:1.5fr 1fr;gap:28px;align-items:end}.hero h1{font-size:42px;line-height:1.08;margin:10px 0}.sub{line-height:1.6;color:#5c6980}.card{background:#fff;border:1px solid #dfe6f1;border-radius:16px;padding:24px;margin-top:26px;box-shadow:0 10px 32px #1720330d}.upload{border:1px dashed #9fb1cf;border-radius:12px;padding:22px;background:#f9fbff}.upload input{display:block;width:100%;margin-top:10px}.btn{margin-top:16px;border:0;border-radius:9px;padding:12px 18px;background:#315fc4;color:#fff;font:inherit;font-weight:700;cursor:pointer}.btn:disabled{opacity:.55}.progress{height:9px;background:#e7edf7;border-radius:20px;overflow:hidden;margin:18px 0 9px}.progress span{display:block;height:100%;width:0;background:#315fc4;transition:width .4s}.status{color:#52627d;font-size:14px}.summary{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.metric{padding:15px;border-radius:10px;background:#f7f9fd}.metric b{font-size:25px;display:block}.issue{padding:18px;margin-top:12px;border-left:5px solid #d08a2d;background:#fff8ed;border-radius:7px}.issue.high{border-color:#c64242;background:#fff2f2}.muted{font-size:14px;color:#5d6980}a{color:#2455b8;font-weight:600}@media(max-width:640px){.hero{grid-template-columns:1fr}.hero h1{font-size:34px}.summary{grid-template-columns:1fr}}</style></head><body><main><div class=hero><div><div class=brand>CONTRACTGUARDIAN</div><h1>Understand contract risk before you sign.</h1><p class=sub>Upload a text-based PDF for a clause-by-clause risk review, suggested improvements, and relevant Indian-law references.</p></div><p class=sub><b>Important:</b> this is decision support, not legal advice. Have qualified Indian counsel review the contract.</p></div><section class=card><div class=upload><b>Upload contract PDF</b><div class=muted>Text-based PDFs work best. Your file is used only for this review.</div><input id=file type=file accept='.pdf,application/pdf'></div><button class=btn id=review>Start contract review</button><div id=work hidden><div class=progress><span id=bar></span></div><div class=status id=status>Preparing review…</div></div></section><section id=results></section></main><script>
const file=document.querySelector('#file'),button=document.querySelector('#review'),work=document.querySelector('#work'),bar=document.querySelector('#bar'),status=document.querySelector('#status'),results=document.querySelector('#results');let timer;function show(v,t){bar.style.width=v+'%';status.textContent=t}function render(data){const flagged=data.clauses.filter(c=>c.status==='success'&&c.review&&c.review.severity!=='Low'),high=flagged.filter(c=>c.review.severity==='High').length;results.innerHTML=`<section class=card><h2>Review complete</h2><div class=summary><div class=metric><b>${data.document.clauses_analyzed}</b><span class=muted>clauses analyzed</span></div><div class=metric><b>${flagged.length}</b><span class=muted>clauses need review</span></div><div class=metric><b>${high}</b><span class=muted>high-risk clauses</span></div></div><p><a href='${data.report_download_url}'>Download the review PDF</a></p></section>`+flagged.map(c=>`<article class='issue ${c.review.severity==='High'?'high':''}'><b>${c.clause_id} · ${c.review.severity} risk · ${c.review.risk_score}/100</b><div class=muted>Model confidence: ${Math.round(c.review.confidence*100)}%</div><p>${c.review.suggestion}</p>${c.review.laws.map(l=>`<p><b>${l.section}</b><br><span class=muted>${l.summary}</span><br><a target=_blank href='${l.url}'>Read the India Code section</a></p>`).join('')}</article>`).join('')}
async function poll(id){const r=await fetch('/api/reviews/'+id),d=await r.json();if(d.status==='failed'){show(100,d.error);button.disabled=false;return}if(d.status==='completed'){show(100,'Review complete.');render(d.result);button.disabled=false;return}show(d.progress,d.message);timer=setTimeout(()=>poll(id),900)}button.onclick=async()=>{if(!file.files[0]){alert('Choose a PDF first.');return}button.disabled=true;work.hidden=false;results.innerHTML='';show(4,'Uploading contract…');const form=new FormData();form.append('file',file.files[0]);try{const r=await fetch('/api/reviews',{method:'POST',body:form}),d=await r.json();if(!r.ok)throw Error(d.error||'Upload failed');poll(d.review_id)}catch(e){show(100,e.message);button.disabled=false}};</script></body></html>"""


def create_app():
    app = Flask(__name__); jobs = {}; reports = {}; analyzer = None; analyzer_lock = threading.Lock()
    def get_analyzer():
        nonlocal analyzer
        with analyzer_lock:
            if analyzer is None:
                from src.analysis.contract_analyzer import ContractAnalyzer
                analyzer = ContractAnalyzer()
            return analyzer
    def update(job, progress, message, status="running", **more): jobs[job].update(status=status, progress=progress, message=message, **more)
    def run_review(job, filename):
        try:
            update(job, 12, "Loading the contract model…"); result = get_analyzer().analyze_pdf(filename)
            update(job, 86, "Calculating risk scores and Indian-law references…"); result["clauses"] = [review_clause(c) if c.get("status") == "success" else c for c in result["clauses"]]
            update(job, 94, "Creating your downloadable report…"); report_id = uuid4().hex; reports[report_id] = make_report(result); result["report_download_url"] = f"/api/reports/{report_id}.pdf"; update(job, 100, "Review complete.", status="completed", result=result)
        except Exception as error: update(job, 100, "Review failed.", status="failed", error=str(error))
        finally:
            try: os.unlink(filename)
            except OSError: pass
    @app.get("/")
    def home(): return render_template_string(PAGE)
    @app.post("/api/reviews")
    def start_review():
        upload = request.files.get("file")
        if upload is None or Path(upload.filename or "").suffix.lower() != ".pdf": return jsonify(error="Choose a PDF file."), 400
        temporary = NamedTemporaryFile(delete=False, suffix=".pdf"); upload.save(temporary); temporary.close(); review_id = uuid4().hex; jobs[review_id] = {"status":"queued", "progress":4, "message":"Upload received."}; threading.Thread(target=run_review, args=(review_id, temporary.name), daemon=True).start(); return jsonify(review_id=review_id), 202
    @app.get("/api/reviews/<review_id>")
    def review_status(review_id):
        job = jobs.get(review_id)
        return (jsonify(error="Review not found."), 404) if job is None else jsonify(job)
    @app.get("/api/reports/<report_id>.pdf")
    def report(report_id):
        data = reports.get(report_id)
        return (jsonify(error="Report expired. Run the review again."), 404) if data is None else send_file(BytesIO(data), mimetype="application/pdf", as_attachment=True, download_name="contractguardian-review.pdf")
    return app


app = create_app()
if __name__ == "__main__": app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True)
