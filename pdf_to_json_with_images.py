import fitz
import os
import io
from PIL import Image

PDFS = [
    "L2100 車床程式說明手冊.pdf",
    "L2100車床中文維護手冊(全).pdf",
    "L2100車床參數警報手冊.pdf"
]

OUTPUT_DIR = "static/images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n=== 開始抽取 PDF 圖片 ===\n")

for pdf_path in PDFS:

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    print(f"\n==============================")
    print(f"處理 PDF：{pdf_name}")
    print(f"==============================")

    try:
        doc = fitz.open(pdf_path)

    except Exception as e:
        print(f"開啟失敗：{e}")
        continue

    saved_count = 0

    for page_index in range(len(doc)):

        page = doc[page_index]

        image_list = page.get_images(full=True)

        if not image_list:
            continue

        print(f"\n第 {page_index + 1} 頁找到 {len(image_list)} 張圖片")

        for img_index, img in enumerate(image_list):

            try:

                xref = img[0]

                base_image = doc.extract_image(xref)

                image_bytes = base_image["image"]

                image_ext = base_image["ext"]

                pil_img = Image.open(io.BytesIO(image_bytes))

                width, height = pil_img.size

                # ====================================
                # 過濾 1：太小圖片
                # ====================================
                if width < 120 or height < 120:
                    print(f"跳過小圖片：{width}x{height}")
                    continue

                # ====================================
                # 過濾 2：比例異常
                # ====================================
                ratio = width / height

                if ratio < 0.3 or ratio > 5:
                    print(f"跳過比例異常：{ratio:.2f}")
                    continue

                # ====================================
                # 過濾 3：檔案太小
                # ====================================
                if len(image_bytes) < 5000:
                    print("跳過超小檔案")
                    continue

                # ====================================
                # 灰階分析
                # ====================================
                gray = pil_img.convert("L")

                pixels = list(gray.getdata())

                avg = sum(pixels) / len(pixels)

                # 幾乎全黑
                if avg < 15:
                    print("跳過純黑圖片")
                    continue

                # ====================================
                # 白底比例分析（工程圖判斷）
                # ====================================
                white_pixels = sum(
                    1 for p in pixels if p > 240
                )

                white_ratio = white_pixels / len(pixels)

                if white_ratio < 0.35:
                    print(
                        f"跳過非工程圖：白底比例 {white_ratio:.2f}"
                    )
                    continue

                # ====================================
                # 彩色比例分析（過濾 logo/icon）
                # ====================================
                rgb = pil_img.convert("RGB")

                rgb_pixels = list(rgb.getdata())

                colorful = 0

                for r, g, b in rgb_pixels:

                    if (
                        abs(r - g) > 20
                        or abs(r - b) > 20
                        or abs(g - b) > 20
                    ):
                        colorful += 1

                color_ratio = colorful / len(rgb_pixels)

                if color_ratio > 0.25:
                    print(
                        f"跳過高彩色圖片：{color_ratio:.2f}"
                    )
                    continue

                # ====================================
                # 儲存圖片
                # ====================================
                image_name = (
                    f"{pdf_name}_p{page_index+1}_{img_index+1}.{image_ext}"
                )

                image_path = os.path.join(
                    OUTPUT_DIR,
                    image_name
                )

                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                saved_count += 1

                print(
                    f"保留圖片：{image_name} "
                    f"({width}x{height})"
                )

            except Exception as e:
                print(f"圖片處理失敗：{e}")

    print(
        f"\n{pdf_name} 完成，共保留 {saved_count} 張圖片"
    )

print("\n=== 全部完成 ===")