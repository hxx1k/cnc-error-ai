import fitz
import os

PDFS = [
    {
        "pdf": "L2100 車床程式說明手冊.pdf",
        "out": "page_images/programming"
    },
    {
        "pdf": "L2100車床中文維護手冊(全).pdf",
        "out": "page_images/maintenance"
    },
    {
        "pdf": "L2100車床參數警報手冊.pdf",
        "out": "page_images/parameter"
    }
]

for item in PDFS:

    pdf_path = item["pdf"]
    out_dir = item["out"]

    os.makedirs(out_dir, exist_ok=True)

    print(f"開始轉換：{pdf_path}")

    doc = fitz.open(pdf_path)

    for i, page in enumerate(doc):

        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

        out_path = os.path.join(
            out_dir,
            f"page_{i+1:04d}.png"
        )

        pix.save(out_path)

    print(f"完成：{pdf_path}")

print("全部完成")