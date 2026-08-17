import json
from pathlib import Path


class LegalChunker:

    def __init__(self, knowledge_base_path):

        self.knowledge_base_path = Path(
            knowledge_base_path
        )

    def load(self):

        if not self.knowledge_base_path.exists():

            raise FileNotFoundError(
                f"Knowledge base not found: "
                f"{self.knowledge_base_path}"
            )

        with open(
            self.knowledge_base_path,
            "r",
            encoding="utf-8"
        ) as file:

            documents = json.load(file)

        if not isinstance(
            documents,
            list
        ):

            raise ValueError(
                "Legal knowledge base must "
                "contain a JSON list."
            )

        return documents

    def create_chunks(self):

        documents = self.load()

        chunks = []

        for document in documents:

            text = document.get(
                "text",
                ""
            ).strip()

            if not text:
                continue

            chunks.append({

                "source_id":
                    document.get(
                        "source_id"
                    ),

                "act":
                    document.get(
                        "act"
                    ),

                "section":
                    document.get(
                        "section"
                    ),

                "title":
                    document.get(
                        "title"
                    ),

                "text":
                    text,

                "topics":
                    document.get(
                        "topics",
                        []
                    )
            })

        return chunks