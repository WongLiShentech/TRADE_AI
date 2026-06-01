from pydantic import BaseModel


class JournalGenerateResponse(BaseModel):
    path: str
    week_label: str
    total: int
