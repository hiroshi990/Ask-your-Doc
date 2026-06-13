"""Document parsing via Docling."""

import logging
import tempfile
from pathlib import Path
from typing import Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter,PdfFormatOption
from docling_core.types.doc.document import DoclingDocument

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    ".html", ".htm", ".md", ".txt", ".csv", ".json", ".xml",
    ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp",
}


class DocumentParser:
    """Parses files and pasted text into DoclingDocument objects."""

    def __init__(self) -> None:
        self.options = PdfPipelineOptions()
        self.options.do_ocr = False
        self.do_table_structure = False
        self._converter = DocumentConverter(format_options={
            InputFormat.PDF:PdfFormatOption(
                pipeline_options=self.options
            )
        })

    def parse_file(self, file_path: Path) -> DoclingDocument:
        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        logger.info("Parsing file: %s", file_path.name)
        result = self._converter.convert(str(file_path))
        return result.document

    def parse_text(
        self,
        content: str,
        title: str = "Pasted Text",
    ) -> DoclingDocument:
        """Treat pasted text exactly like a document."""
        logger.info("Parsing pasted text: %s", title)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".md",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            return self.parse_file(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    @staticmethod
    def detect_source_type(file_path: Optional[Path], is_pasted: bool = False) -> str:
        if is_pasted:
            return "pasted_text"
        if file_path is None:
            return "unknown"
        suffix = file_path.suffix.lower().lstrip(".")
        return suffix or "unknown"
