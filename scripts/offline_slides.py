from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor
from html import escape
from io import BytesIO
import subprocess
from pypdf import PdfReader, PdfWriter
from PIL import Image
from scripts.presentation import STEM
from sentinel.config import ROOT


def build():
    """Render the PDF and embed downloads in a self-contained slideshow (macOS)."""
    root = ROOT
    work = root / ".work/offline_slides"
    work.mkdir(parents=True, exist_ok=True)
    delivery = root / "delivery"
    stem = STEM
    pdf = root / "docs/presentation.pdf"
    ppt = delivery / (stem + ".pptx")
    doc = PdfReader(pdf)

    def uri(kind, data):
        return "data:" + kind + ";base64," + b64encode(data).decode("ascii")

    def render(i):
        pg = PdfReader(pdf).pages[i]
        pg.scale_by(2)
        source = work / f"{i + 1:02}.pdf"
        target = source.with_suffix(".png")
        w = PdfWriter()
        w.add_page(pg)
        w.write(source)
        subprocess.run(
            ["sips", "-s", "format", "png", str(source), "--out", str(target)],
            stdout=subprocess.DEVNULL,
            check=True,
        )
        with Image.open(target) as im:
            assert im.size == (1920, 1080), im.size
            opaque = Image.new("RGB", im.size, "white")
            if im.mode == "RGBA":
                opaque.paste(im, mask=im.getchannel("A"))
            else:
                opaque.paste(im)
            stream = BytesIO()
            opaque.save(stream, format="PNG", optimize=True)
            return uri("image/png", stream.getvalue())

    with ThreadPoolExecutor(max_workers=4) as pool:
        imgs = list(pool.map(render, range(len(doc.pages))))
    pages = []
    for i, (page, image) in enumerate(zip(doc.pages, imgs), 1):
        lines = page.extract_text().splitlines()
        title = lines[1] if i == 1 else lines[2]
        hotspots = []
        for ref in page.get("/Annots", []):
            annot = ref.get_object()
            url = annot.get("/A", {}).get("/URI", "")
            if not url.startswith("https://"):
                continue
            x1, y1, x2, y2 = map(float, annot["/Rect"])
            w, h = float(page.mediabox.width), float(page.mediabox.height)
            style = f"left:{100 * x1 / w:.4f}%;top:{100 * (h - y2) / h:.4f}%;width:{100 * (x2 - x1) / w:.4f}%;height:{100 * (y2 - y1) / h:.4f}%"
            hotspots.append(
                f'<a class="hotspot" href="{escape(url, quote=True)}" aria-label="{escape(url, quote=True)}" style="{style}" target="_blank" rel="noopener noreferrer"></a>'
            )
        pages.append(
            f'<section id="slayt-{i}" aria-label="Slayt {i}: {escape(title)}"><div class="slide"><img alt="{escape(title, quote=True)}" width="1920" height="1080" src="{image}">{"".join(hotspots)}</div><details><summary>Slayt {i} / 10 · Metni göster</summary><p>{escape(chr(10).join(lines))}</p></details></section>'
        )
    numberlinks = "".join(
        f'<a href="#slayt-{i}" aria-label="Slayt {i}">{i}</a>'
        for i in range(1, len(pages) + 1)
    )
    html = """<!doctype html><html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SentinelAI — Sunum</title>
    <style>
    *{box-sizing:border-box}body{margin:0;background:#eff3ef;color:#143d35;font:16px/1.4 system-ui,sans-serif}
    header{position:sticky;top:0;z-index:2;background:#103e36;color:#fff;box-shadow:0 1px 4px #143d3522}
    nav{max-width:1180px;padding:12px 16px;margin:auto;display:flex;flex-wrap:wrap;align-items:center;gap:14px}
    nav strong{font-size:1.2rem}a{color:inherit}nav .slides{display:flex;flex-wrap:wrap;gap:2px}nav .slides a{padding:3px 6px;text-decoration:none;font-variant-numeric:tabular-nums}nav .slides a:hover,nav .slides a:focus{background:#37685b}
    .files{margin-left:auto;display:flex;gap:16px}.files a{text-underline-offset:3px}
    main{max-width:1180px;margin:auto;padding:16px}section{scroll-margin-top:80px;margin:0 0 20px}.slide{position:relative;box-shadow:0 2px 10px #143d3520}.slide img{display:block;width:100%;height:auto}
    .hotspot{position:absolute}.hotspot:focus-visible{outline:3px solid #ff9600}details{padding:7px 0;color:#3a5650;font-size:.88rem}details p{white-space:pre-line}summary{cursor:pointer;display:list-item;width:fit-content}
    @media(max-width:760px){header{position:static}.files{margin:0}main{padding:8px}section{scroll-margin-top:8px}}
    @media print{header,details{display:none}body,main{padding:0;margin:0;background:white;max-width:none}section{break-after:page;margin:0}.slide{box-shadow:none}section:last-child{break-after:auto}@page{size:landscape;margin:0}}
    </style><header><nav aria-label="Sunum"><strong>SentinelAI</strong><span class="slides">NUMBERLINKS</span><span class="files"><a download="SentinelAI.pdf" href="PDF_URI">PDF’yi indir</a><a download="SentinelAI.pptx" href="PPT_URI">PowerPoint’i indir</a></span></nav></header><main>PAGES</main></html>"""
    html = (
        html.replace("NUMBERLINKS", numberlinks)
        .replace("PDF_URI", uri("application/pdf", pdf.read_bytes()))
        .replace(
            "PPT_URI",
            uri(
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                ppt.read_bytes(),
            ),
        )
        .replace("PAGES", "".join(pages))
    )
    out = delivery / "Sunumu_Ac.html"
    out.write_text(html)
    print(out, out.stat().st_size)


if __name__ == "__main__":
    build()
