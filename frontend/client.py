from typing import Any,Optional
from fastapi import applications
import requests

BASE_URL = "http://127.0.0.1:8000/api/v1"
TIMEOUT_SECONDS = 300

class BackendAPIError(RuntimeError):
    pass

def upload_file(uploaded_file:Any) -> dict[str,Any]:
    files = {
        "file": (    
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }

    data = {
        "document_name": uploaded_file.name.rsplit(".",1)[0]
    }

    response = requests.post(
        f"{BASE_URL}/documents/upload",
        files = files,
        timeout = TIMEOUT_SECONDS,
        data = data
    )

    if not response.ok:
        raise BackendAPIError(
            f"Upload failed ({response.status_code}):{response.text}"

        )
    return response.json()

def paste_text(text:str) -> dict[str,Any]:
    payload = {
        "text_content":text
    }
    
    response = requests.post(
        f"{BASE_URL}/documents/paste",
        json=payload
    )
    if not response.ok:
        raise BackendAPIError(
            f"Upload failed ({response.status_code})"
        )

    return response.json()

def ask_question(
    query:str,
    use_cache:Optional[bool] = True
) -> dict[str,Any]:
    payload = {
        "query":query,
        "use_cache":use_cache
    }

    response = requests.post(
        f"{BASE_URL}/chat/chat",
        json=payload,
        timeout=TIMEOUT_SECONDS
    )

    if not response.ok:
        raise BackendAPIError(
            f"Chat request failed ({response.status_code}):{response.text}"
        )
    return response.json()

def delete(doc_info:dict[str,Any]):
    document_id=doc_info.get("document_id")
    response = requests.delete(
        f"{BASE_URL}/documents/delete_by_id",
        params = {"document_id": document_id},
        timeout = 30
    )

    if not response.ok:
        raise BackendAPIError(
            f"Delete failed ({response.status_code}) : {response.text}"
        )
    return "Success"