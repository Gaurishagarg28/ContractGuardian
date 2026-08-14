from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PageText:
    page_number: int
    text: str


@dataclass
class Clause:
    clause_id: str
    text: str
    page_start: int
    page_end: int
    section: Optional[str] = None
    heading: Optional[str] = None
    metadata: dict = field(default_factory=dict)