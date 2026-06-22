import logging
import tempfile
from pathlib import Path
from typing import Optional
import gc
import pypdf
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
        self.options.do_table_structure = False
        self.converter = DocumentConverter(format_options={
                    InputFormat.PDF:PdfFormatOption(
                        pipeline_options=self.options
                    )})
        self.page_interval = 4

    def direct_parsing(self,file_path:Path) -> DoclingDocument:
        logger.info("Parsing file: %s", file_path.name)
        result = self.converter.convert(str(file_path))
        return result.document

    def parse_file(self, file_path: Path) -> DoclingDocument:
        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        if suffix == '.pdf':
            final_text = []
            reader = pypdf.PdfReader(str(file_path))
            total_pages = reader.get_num_pages()
            if total_pages > self.page_interval:
                logger.info("Parsing file: %s", file_path.name)
                for start_idx in range(1,total_pages + 1,self.page_interval):
                    end_idx = min(start_idx+self.page_interval - 1,total_pages)

                    try:
                        result = self.converter.convert(str(file_path),page_range=(start_idx,end_idx))
                        md_text = result.document.export_to_markdown()
                        final_text.append(md_text)
                        del result
                    except Exception as e:
                        print("Error parsing the file",e)
                    finally:
                        gc.collect()
                if not final_text:
                    raise ValueError("No text can be extracted from pdf")
                content = "\n\n".join(final_text) 
                response = self.converter.convert_string(content,format=InputFormat.MD)
                return response.document

            else: return self.direct_parsing(file_path)

        else:
            return self.direct_parsing(file_path)


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
