import pypdf
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat


pdf_path = r"uploads\733ae99e-b726-43c6-bb09-f75edffd4ddb.pdf"
chunk_size = 5

reader = pypdf.PdfReader(pdf_path)
total_pages = reader.get_num_pages()

text = []

# convertor = DocumentConverter()

# result = convertor.convert(pdf_path)
# print(result.document.num_pages())
for start_idx in range(0,total_pages,chunk_size):
    end_idx = min(start_idx + chunk_size,total_pages)

    try:
        convertor = DocumentConverter()

        result = convertor.convert(pdf_path,page_range=(start_idx,end_idx))
        # print(result.document.num_pages)

        md_text = result.document.export_to_markdown()

        text.append(md_text)

    except Exception as e:
        print(e)

final_text = "\n\n".join(text)
response = convertor.convert_string(final_text,format=InputFormat.MD)
output = response.document
print(output)
# with open("test_output.md","w",encoding="utf-8") as f:
#     f.write(output)
print("Conversion complete")


