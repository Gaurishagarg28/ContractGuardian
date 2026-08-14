import re
from pathlib import Path

import pymupdf

from .document_models import PageText


class PDFLoader:

    def load(self, pdf_path):

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                "Only PDF files are supported."
            )

        pages = []

        with pymupdf.open(pdf_path) as document:

            for page_number, page in enumerate(
                document,
                start=1
            ):

                text = page.get_text("text")

                text = self.clean_text(text)

                pages.append(
                    PageText(
                        page_number=page_number,
                        text=text
                    )
                )

        return pages

    @staticmethod
    def clean_text(text):

        if not text:
            return ""

        # -----------------------------------------------------
        # Normalize line endings
        # -----------------------------------------------------

        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # -----------------------------------------------------
        # Fix PDF word-boundary extraction problems
        #
        # Examples:
        #
        # forLicensee -> for Licensee
        # withoutLicensor -> without Licensor
        # subjectto -> subject to
        # shallsurvive -> shall survive
        # constitutethe -> constitute the
        # -----------------------------------------------------

        text = re.sub(
            r"([a-z])([A-Z])",
            r"\1 \2",
            text
        )

        # Common lower-case word collisions caused by PDF layout.
        replacements = {
            "atthe": "at the",
            "ofthe": "of the",
            "inthe": "in the",
            "onthe": "on the",
            "tothe": "to the",
            "forthe": "for the",
            "fromthe": "from the",
            "withthe": "with the",
            "bythe": "by the",
            "subjectto": "subject to",
            "shallsurvive": "shall survive",
            "constitutethe": "constitute the",
            "appropriateto": "appropriate to",
            "forconvenience": "for convenience",
            "effectiveDate": "effective Date",
        }

        for bad, good in replacements.items():
            text = text.replace(bad, good)

        # -----------------------------------------------------
        # Normalize spaces
        # -----------------------------------------------------

        text = re.sub(
            r"[ \t]+",
            " ",
            text
        )

        # -----------------------------------------------------
        # Clean line edges
        # -----------------------------------------------------

        text = "\n".join(
            line.strip()
            for line in text.splitlines()
        )

        # -----------------------------------------------------
        # Remove excessive blank lines
        # -----------------------------------------------------

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text
        )

        return text.strip()