"""
parser.py
---------
Handles reading raw resume / job description files and turning them into
plain text that the rest of the pipeline can work with.

Supports:
- PDF files (via pdfplumber)
- Plain text (.txt) files

Streamlit gives us an "uploaded file" object (a file-like object), so every
function here is written to accept either a real file path (str) or that
kind of file-like object, since both come up depending on whether we're
testing from the command line or running inside app.py.
"""

import pdfplumber


def extract_text_from_pdf(file):
    """
    Extracts and concatenates text from every page of a PDF.

    Parameters
    ----------
    file : str or file-like object
        Either a path to a .pdf file, or an in-memory uploaded file
        (e.g. what st.file_uploader() returns in Streamlit).

    Returns
    -------
    str : the extracted text, with pages joined by a newline.
    """
    pages_text = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)

    if not pages_text:
        raise ValueError(
            "Could not extract any text from this PDF. "
            "It might be a scanned/image-based PDF, which this project "
            "does not handle (no OCR step)."
        )

    return "\n".join(pages_text)


def extract_text_from_txt(file):
    """
    Reads a plain text file. Accepts either a path or a Streamlit
    UploadedFile object (which behaves like a bytes buffer).
    """
    if hasattr(file, "read"):
        raw = file.read()
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="ignore")
        return raw
    else:
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def parse_document(file, filename=None):
    """
    Single entry point used by app.py: figures out whether the uploaded
    file is a PDF or plain text based on its name, and routes it to the
    right extractor.

    Parameters
    ----------
    file : str or file-like object
    filename : str, optional
        Needed when `file` is a file-like object without a reliable name
        attribute (Streamlit's UploadedFile does have `.name`, but we
        allow overriding it for flexibility / testing).
    """
    name = filename or getattr(file, "name", "") or (file if isinstance(file, str) else "")
    name = name.lower()

    if name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif name.endswith(".txt"):
        return extract_text_from_txt(file)
    else:
        raise ValueError(
            f"Unsupported file type for '{name}'. Please upload a .pdf or .txt file."
        )
