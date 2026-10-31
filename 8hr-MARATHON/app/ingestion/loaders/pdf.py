import logfire
from pypdf import PdfReader


def parse_pdf(file_path: str) -> str:
    """
    Extract text from a PDF locally using pypdf.
    Falls back to pdfplumber for pages that yield no text (e.g. image-heavy pages).
    """
    with logfire.span("PDF Parsing (local)", filename=file_path):
        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            logfire.info(f"PDF has {total_pages} pages.")

            # Indexed by page so pdfplumber fallbacks land in reading order, not at the end.
            text_parts: list[str] = [""] * total_pages
            blank_pages: list[int] = []

            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    text_parts[i] = text
                else:
                    blank_pages.append(i + 1)

            # Fallback: use pdfplumber for any pages pypdf returned blank
            if blank_pages:
                logfire.info(f"pypdf returned blank on pages {blank_pages} — retrying with pdfplumber.")
                try:
                    import pdfplumber
                    with pdfplumber.open(file_path) as pdf:
                        for page_num in blank_pages:
                            page = pdf.pages[page_num - 1]
                            fallback_text = page.extract_text() or ""
                            if fallback_text.strip():
                                text_parts[page_num - 1] = fallback_text
                except Exception as plumber_err:
                    logfire.warning(f"pdfplumber fallback failed: {plumber_err}")

            # Blank line between pages gives the chunker a paragraph boundary.
            full_text = "\n\n".join(p for p in text_parts if p.strip())

            if not full_text.strip():
                logfire.warning(f"No text extracted from {file_path}. File may be fully image-based.")
            else:
                logfire.info(f"Extracted {len(full_text)} characters from {file_path}.")

            return full_text

        except Exception as e:
            logfire.error(f"PDF Parse Failed for {file_path}: {e}")
            raise
