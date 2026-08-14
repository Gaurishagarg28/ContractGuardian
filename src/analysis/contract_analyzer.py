from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.clause_segmenter import ClauseSegmenter

from src.analysis.clause_analyzer import ClauseAnalyzer

from src.ml.inference.embedding_service import EmbeddingService
from src.ml.inference.predictor import ContractPredictor


class ContractAnalyzer:

    def __init__(self):

        print("Initializing ContractGuardian...")

        # -----------------------------------------------------
        # DOCUMENT PIPELINE
        # -----------------------------------------------------

        self.pdf_loader = PDFLoader()

        self.clause_segmenter = ClauseSegmenter()

        # -----------------------------------------------------
        # ML PIPELINE
        # -----------------------------------------------------

        self.embedding_service = EmbeddingService()

        self.predictor = ContractPredictor()

        self.clause_analyzer = ClauseAnalyzer(
            embedding_service=self.embedding_service,
            predictor=self.predictor
        )

        print("ContractGuardian initialized.")

    def analyze_pdf(self, pdf_path):

        # =====================================================
        # 1. LOAD PDF
        # =====================================================

        pages = self.pdf_loader.load(
            pdf_path
        )

        if not pages:

            raise ValueError(
                "No pages were extracted from the PDF."
            )

        # =====================================================
        # 2. VALIDATE EXTRACTED TEXT
        # =====================================================

        total_characters = sum(
            len(page.text)
            for page in pages
        )

        if total_characters == 0:

            raise ValueError(
                "No text could be extracted from the PDF."
            )

        # =====================================================
        # 3. SEGMENT CLAUSES
        # =====================================================

        clauses = self.clause_segmenter.segment(
            pages
        )

        if not clauses:

            raise ValueError(
                "No clauses were detected."
            )

        # =====================================================
        # 4. ANALYZE EVERY CLAUSE
        # =====================================================

        clause_results = []

        for index, clause in enumerate(
            clauses,
            start=1
        ):

            print(
                f"\nAnalyzing clause "
                f"{index}/{len(clauses)}: "
                f"{clause.clause_id}"
            )

            try:

                result = (
                    self.clause_analyzer.analyze(
                        clause
                    )
                )

                result["status"] = "success"

                clause_results.append(
                    result
                )

            except Exception as error:

                print(
                    f"WARNING: Failed to analyze "
                    f"{clause.clause_id}: {error}"
                )

                clause_results.append({

                    "clause_id":
                        getattr(
                            clause,
                            "clause_id",
                            None
                        ),

                    "section":
                        getattr(
                            clause,
                            "section",
                            None
                        ),

                    "heading":
                        getattr(
                            clause,
                            "heading",
                            None
                        ),

                    "page_start":
                        getattr(
                            clause,
                            "page_start",
                            None
                        ),

                    "page_end":
                        getattr(
                            clause,
                            "page_end",
                            None
                        ),

                    "text":
                        getattr(
                            clause,
                            "text",
                            ""
                        ),

                    "status":
                        "failed",

                    "error":
                        str(error)
                })

        # =====================================================
        # 5. CONTRACT SUMMARY
        # =====================================================

        successful = [
            result
            for result in clause_results
            if result["status"] == "success"
        ]

        failed = [
            result
            for result in clause_results
            if result["status"] == "failed"
        ]

        return {

            "document": {

                "pdf_path": str(
                    pdf_path
                ),

                "pages": len(pages),

                "characters":
                    total_characters,

                "clauses_detected":
                    len(clauses),

                "clauses_analyzed":
                    len(successful),

                "clauses_failed":
                    len(failed)
            },

            "clauses":
                clause_results
        }