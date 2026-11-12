"""Create the 30-benchmark comparison PDF from the current plot images."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = ROOT / "results" / "plots"
OUTPUT = ROOT / "output" / "pdf" / "benchmark_comparison.pdf"
BENCHMARKS = [
    "ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6",
    "Binh1", "Binh3", "Deb", "Deb2", "Fonseca", "Fonseca2", "Kursawe",
    "Laumanns", "Murata", "Quagliarella", "Rendon", "Rendon2", "Schaffer",
    "Schaffer2", "Srinivas", "Tanaka", "Viennet", "Viennet2",
    "DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7",
]


def main():
    missing = [name for name in BENCHMARKS if not (PLOT_DIR / f"{name.lower()}_pareto_front.png").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing benchmark plots: {', '.join(missing)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    page_width, page_height = A4
    pdf = canvas.Canvas(str(OUTPUT), pagesize=A4, pageCompression=1)
    pdf.setTitle("Benchmark Pareto Front Comparison")
    pdf.setAuthor("Multi-objective optimization benchmark suite")

    for page_index, start in enumerate(range(0, len(BENCHMARKS), 3), start=1):
        pdf.setFillColorRGB(.10, .19, .34)
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(28, page_height - 32, "Benchmark Pareto Front Comparison")
        pdf.setFillColorRGB(.30, .38, .50)
        pdf.setFont("Helvetica", 8)
        pdf.drawString(28, page_height - 46,
                       "NSGA-II | 5 independent runs | DTLZ-specific evaluation budgets")
        pdf.drawRightString(page_width - 28, page_height - 46,
                            f"Page {page_index} of 10")

        band_y = page_height - 73
        pdf.setFillColorRGB(.91, .94, .97)
        pdf.roundRect(28, band_y - 12, page_width - 56, 22, 5, fill=1, stroke=0)
        pdf.setFillColorRGB(.20, .27, .37)
        pdf.setFont("Helvetica-Bold", 7)
        pdf.drawCentredString(280, band_y - 4, "OBTAINED FRONT                                      TRUE FRONT")

        names = BENCHMARKS[start:start + 3]
        row_top = band_y - 25
        row_height = 224
        image_x = 91
        image_width = page_width - image_x - 28
        image_height = image_width / 2.4
        for row, name in enumerate(names):
            top = row_top - row * row_height
            image_y = top - image_height
            pdf.setFillColorRGB(.10, .19, .34)
            pdf.setFont("Helvetica-Bold", 8)
            pdf.drawString(29, image_y + image_height / 2, name.upper())
            image_path = PLOT_DIR / f"{name.lower()}_pareto_front.png"
            pdf.drawImage(ImageReader(str(image_path)), image_x, image_y,
                          width=image_width, height=image_height,
                          preserveAspectRatio=True, anchor="c", mask="auto")
            if row < len(names) - 1:
                pdf.setStrokeColorRGB(.82, .86, .90)
                pdf.line(28, image_y - 12, page_width - 28, image_y - 12)

        pdf.setStrokeColorRGB(.82, .86, .90)
        pdf.line(28, 29, page_width - 28, 29)
        pdf.setFillColorRGB(.30, .38, .50)
        pdf.setFont("Helvetica", 7)
        pdf.drawString(28, 17,
                       "Blue: obtained nondominated union across runs | Red: analytical/reference front")
        pdf.drawRightString(page_width - 28, 17, "NSGA-II")
        pdf.showPage()

    pdf.save()
    print(OUTPUT)


if __name__ == "__main__":
    main()
