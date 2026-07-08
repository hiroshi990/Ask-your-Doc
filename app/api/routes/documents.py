import shutil
import uuid
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from app.config import get_settings
from app.ingestion.pipeline import IngestionPipeline
from app.models.schemas import DocumentInfo, DocumentUploadResponse, PastedTextRequest

router = APIRouter()
settings = get_settings()
pipeline = IngestionPipeline()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    document_name: str = Form(),
) -> DocumentUploadResponse:
    suffix = Path(file.filename or "document.txt").suffix
    save_path = settings.upload_dir / f"{uuid.uuid4()}{suffix}"

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        doc_info, _ = await pipeline.ingest_file(
            file_path=save_path,
            document_name=document_name or None,
        )
    except:
        raise HTTPException(status_code=400,detail="Document is too big to process")
    finally:
        save_path.unlink(missing_ok=True)

    return DocumentUploadResponse(
        document=doc_info,
        chunks_created=doc_info.chunk_count,
        message=f"Successfully ingested '{doc_info.document_name}' with {doc_info.chunk_count} chunks",
    )


@router.post("/paste", response_model=DocumentUploadResponse)
async def paste_text(payload: PastedTextRequest) -> DocumentUploadResponse:
    try:

        doc_info, _ = await pipeline.ingest_pasted_text(
            content=payload.content,
            title=payload.title,
        )
    except:
        raise HTTPException(status_code=400,detail="Text is too large to process")

    return DocumentUploadResponse(
        document=doc_info,
        chunks_created=doc_info.chunk_count,
        message=f"Successfully ingested pasted text '{doc_info.document_name}' with {doc_info.chunk_count} chunks",
    )

@router.get("/",response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    return pipeline.document_store.list_all()

@router.get("/{document_id}", response_model=DocumentInfo)
def get_documents(document_id:str) -> Optional[DocumentInfo]:
    return pipeline.document_store.get(document_id)


@router.delete("/flush_all",status_code=204)
async def delete() -> None:
    if not await pipeline.database_flush():
        raise HTTPException(status_code=404, detail = 'Already empty')


@router.delete("/delete_by_id", status_code=204)
async def delete_document(document_id: str) -> None:
    if not await pipeline.delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
