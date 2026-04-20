from pathlib import Path
import textwrap


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def wrap_markdown_lines(text: str, width: int = 92) -> list[str]:
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()

        if line.startswith("# "):
            out.append("__TITLE__ " + line[2:].strip())
            out.append("")
            continue

        if line.startswith("## "):
            out.append("__H2__ " + line[3:].strip())
            continue

        prefix = ""
        content = line
        if line.startswith("- "):
            prefix = "• "
            content = line[2:].strip()
        elif len(line) > 2 and line[0].isdigit() and line[1] == ".":
            prefix = f"{line[0]}. "
            content = line[2:].strip()

        if not content:
            out.append("")
            continue

        wrapped = textwrap.wrap(content, width=max(20, width - len(prefix)))
        for i, part in enumerate(wrapped):
            out.append((prefix if i == 0 else " " * len(prefix)) + part)

    return out


def write_simple_pdf(lines: list[str], output_path: Path) -> None:
    page_w = 612  # 8.5 in * 72
    page_h = 792  # 11 in * 72
    left = 54
    top = 752
    bottom = 46

    font_body = 11
    font_h2 = 13
    font_title = 16

    leading_body = 14
    leading_h2 = 18
    leading_title = 22

    pages: list[list[str]] = []
    page_cmds: list[str] = []
    y = top

    def new_page() -> None:
        nonlocal page_cmds, y
        pages.append(page_cmds)
        page_cmds = []
        y = top

    for line in lines:
        if line.startswith("__TITLE__ "):
            text = line.replace("__TITLE__ ", "", 1)
            needed = leading_title
            size = font_title
        elif line.startswith("__H2__ "):
            text = line.replace("__H2__ ", "", 1)
            needed = leading_h2
            size = font_h2
        else:
            text = line
            needed = leading_body if text else 8
            size = font_body

        if y - needed < bottom:
            new_page()

        if text:
            escaped = pdf_escape(text)
            page_cmds.append(f"BT /F1 {size} Tf {left} {y} Td ({escaped}) Tj ET")
        y -= needed

    if page_cmds or not pages:
        pages.append(page_cmds)

    objects: list[bytes] = []

    def add_obj(content: str | bytes) -> int:
        if isinstance(content, str):
            b = content.encode("latin-1", errors="replace")
        else:
            b = content
        objects.append(b)
        return len(objects)

    # 1: Font object
    font_id = add_obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_ids = []
    content_ids = []

    for cmds in pages:
        stream = "\n".join(cmds) + "\n"
        stream_bytes = stream.encode("latin-1", errors="replace")
        content_id = add_obj(b"<< /Length " + str(len(stream_bytes)).encode() + b" >>\nstream\n" + stream_bytes + b"endstream")
        content_ids.append(content_id)

        page_dict = f"<< /Type /Page /Parent 0 0 R /MediaBox [0 0 {page_w} {page_h}] /Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
        page_id = add_obj(page_dict)
        page_ids.append(page_id)

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    pages_id = add_obj(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>")

    # patch parent references
    for pid in page_ids:
        raw = objects[pid - 1].decode("latin-1")
        objects[pid - 1] = raw.replace("/Parent 0 0 R", f"/Parent {pages_id} 0 R").encode("latin-1")

    catalog_id = add_obj(f"<< /Type /Catalog /Pages {pages_id} 0 R >>")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]

    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{idx} 0 obj\n".encode("latin-1"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_pos = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode("latin-1"))

    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode(
            "latin-1"
        )
    )

    output_path.write_bytes(pdf)


def main() -> None:
    here = Path(__file__).resolve().parent
    md_files = sorted(here.glob("0*_*.md"))
    if not md_files:
        raise SystemExit("No instruction markdown files found.")

    for md_file in md_files:
        lines = wrap_markdown_lines(md_file.read_text(encoding="utf-8"))
        pdf_file = md_file.with_suffix(".pdf")
        write_simple_pdf(lines, pdf_file)
        print(f"Generated: {pdf_file.relative_to(here.parent)}")


if __name__ == "__main__":
    main()
