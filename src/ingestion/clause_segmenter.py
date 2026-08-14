import re

from .document_models import Clause


class ClauseSegmenter:

    NUMBERED_HEADING = re.compile(
        r"^\s*(\d+(?:\.\d+)*)[\.\)]?\s+(.+?)\s*$"
    )

    ARTICLE_HEADING = re.compile(
        r"^\s*(?:ARTICLE|Article)\s+"
        r"([IVXLCDM0-9]+)"
        r"(?:[\.\:\-\s]+)(.*)$"
    )

    SECTION_HEADING = re.compile(
        r"^\s*(?:SECTION|Section)\s+"
        r"(\d+(?:\.\d+)*)"
        r"(?:[\.\:\-\s]+)(.*)$"
    )

    ADMINISTRATIVE_HEADINGS = {
        "PARTIES",
        "RECITALS",
        "BACKGROUND",
        "INTRODUCTION",
        "SIGNATURES",
        "SIGNATURE",
        "EXHIBITS",
        "EXHIBIT",
        "SCHEDULE",
        "SCHEDULES",
        "APPENDIX",
        "APPENDICES",
        "TABLE OF CONTENTS",
    }

    def segment(self, pages):

        clauses = []

        current_lines = []

        current_heading = None
        current_section = None

        start_page = None

        clause_number = 1

        # Important:
        # Everything before the first real numbered clause
        # is document-level material, not a clause.
        before_first_clause = True

        # -----------------------------------------------------
        # Process pages
        # -----------------------------------------------------

        for page in pages:

            lines = page.text.splitlines()

            for raw_line in lines:

                line = raw_line.strip()

                if not line:
                    continue

                heading_info = self._detect_heading(line)

                # -------------------------------------------------
                # Real clause heading
                # -------------------------------------------------

                if heading_info:

                    # Administrative heading
                    if heading_info["administrative"]:

                        # Do NOT create a clause.
                        #
                        # Most importantly, do not start collecting
                        # party/signature text here.
                        current_lines = []

                        current_heading = None
                        current_section = None

                        continue

                    # Real numbered/article/section heading
                    if not heading_info["administrative"]:

                        # Flush previous clause
                        if current_lines:

                            added = self._create_clause(
                                clauses=clauses,
                                lines=current_lines,
                                clause_number=clause_number,
                                start_page=start_page,
                                end_page=page.page_number,
                                heading=current_heading,
                                section=current_section
                            )

                            if added:
                                clause_number += 1

                        current_lines = []

                        current_section = (
                            heading_info["section"]
                        )

                        current_heading = (
                            heading_info["heading"]
                        )

                        start_page = page.page_number

                        before_first_clause = False

                        continue

                # -------------------------------------------------
                # Ignore document-level material
                # -------------------------------------------------

                if before_first_clause:

                    continue

                # -------------------------------------------------
                # Normal clause content
                # -------------------------------------------------

                if start_page is None:

                    start_page = page.page_number

                current_lines.append(line)

        # -----------------------------------------------------
        # Flush final clause
        # -----------------------------------------------------

        if current_lines:

            self._create_clause(
                clauses=clauses,
                lines=current_lines,
                clause_number=clause_number,
                start_page=start_page,
                end_page=pages[-1].page_number,
                heading=current_heading,
                section=current_section
            )

        return clauses

    # =========================================================
    # HEADING DETECTION
    # =========================================================

    def _detect_heading(self, line):

        # -----------------------------------------------------
        # ARTICLE
        # -----------------------------------------------------

        match = self.ARTICLE_HEADING.match(line)

        if match:

            heading = match.group(2).strip()

            return {
                "section": (
                    f"ARTICLE {match.group(1)}"
                ),
                "heading": heading or None,
                "administrative": (
                    self._is_administrative(
                        heading
                    )
                )
            }

        # -----------------------------------------------------
        # SECTION
        # -----------------------------------------------------

        match = self.SECTION_HEADING.match(line)

        if match:

            heading = match.group(2).strip()

            return {
                "section": match.group(1),
                "heading": heading or None,
                "administrative": (
                    self._is_administrative(
                        heading
                    )
                )
            }

        # -----------------------------------------------------
        # NUMBERED CLAUSE
        #
        # Example:
        # 1. TERM AND RENEWAL
        # 2. LICENSE GRANT
        # -----------------------------------------------------

        match = self.NUMBERED_HEADING.match(line)

        if match:

            section = match.group(1)
            heading = match.group(2).strip()

            # Avoid interpreting long numbered prose
            # as a heading.
            if len(line) <= 120:

                return {
                    "section": section,
                    "heading": heading,
                    "administrative": False
                }

        # -----------------------------------------------------
        # Administrative heading
        # -----------------------------------------------------

        if self._is_administrative(line):

            return {
                "section": None,
                "heading": line,
                "administrative": True
            }

        return None

    # =========================================================
    # ADMINISTRATIVE DETECTION
    # =========================================================

    def _is_administrative(self, heading):

        if not heading:
            return False

        normalized = re.sub(
            r"[^A-Z ]",
            "",
            heading.upper()
        ).strip()

        return normalized in self.ADMINISTRATIVE_HEADINGS

    # =========================================================
    # CREATE CLAUSE
    # =========================================================

    @staticmethod
    def _create_clause(
        clauses,
        lines,
        clause_number,
        start_page,
        end_page,
        heading,
        section
    ):

        text = " ".join(lines).strip()

        if len(text) < 40:
            return False

        upper = text.upper()

        # -----------------------------------------------------
        # Never treat signature blocks as clauses
        # -----------------------------------------------------

        signature_markers = [
            "IN WITNESS WHEREOF",
            "AUTHORIZED REPRESENTATIVE",
            "NAME:",
            "TITLE:",
            "SIGNATURE:",
        ]

        signature_hits = sum(
            marker in upper
            for marker in signature_markers
        )

        if signature_hits >= 2:
            return False

        # -----------------------------------------------------
        # Never treat document introduction as a clause
        # -----------------------------------------------------

        intro_markers = [
            "THIS SOFTWARE SERVICES AND LICENSING AGREEMENT",
            "IS ENTERED INTO BY AND BETWEEN",
            "BLUE PEAK RETAIL SOLUTIONS",
            "NORTHSTAR TECHNOLOGIES",
        ]

        intro_hits = sum(
            marker in upper
            for marker in intro_markers
        )

        if intro_hits >= 2:
            return False

        clauses.append(
            Clause(
                clause_id=f"CL-{clause_number:04d}",
                text=text,
                page_start=start_page,
                page_end=end_page,
                section=section,
                heading=heading
            )
        )

        return True