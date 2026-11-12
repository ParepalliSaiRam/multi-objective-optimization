from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

root = Path(__file__).resolve().parents[1]
plot_dir = root / "results" / "plots" / "mmo_24"
output = root / "output" / "pdf" / "CEC2020_MMO_PS_PF_Comparison.pdf"
problems = (
    "MMF1", "MMF2", "MMF4", "MMF5", "MMF7", "MMF8", "MMF10", "MMF11",
    "MMF12", "MMF13", "MMF14", "MMF15", "MMF1_e", "MMF14_a", "MMF15_a",
    "MMF10_l", "MMF11_l", "MMF12_l", "MMF13_l", "MMF15_l", "MMF15_a_l",
    "MMF16_l1", "MMF16_l2", "MMF16_l3",
)
page_width, page_height = A4
margin = 30
image_width = page_width - 2 * margin
image_height = image_width / 2.4
output.parent.mkdir(parents=True, exist_ok=True)
pdf = canvas.Canvas(str(output), pagesize=A4, pageCompression=1)
pdf.setTitle("CEC2020 MMO Pareto Set and Front Comparisons")
pdf.setAuthor("Multi-objective optimization benchmark suite")

for page_number, name in enumerate(problems, start=1):
    pdf.setFillColorRGB(.10, .19, .34)
    pdf.setFont("Helvetica-Bold", 17)
    pdf.drawString(margin, page_height - 32, "CEC2020 MMO: PS and PF comparison")
    pdf.setFillColorRGB(.30, .38, .50)
    pdf.setFont("Helvetica", 8)
    pdf.drawString(margin, page_height - 45, "NSGA-II obtained solutions compared with the supplied reference data")
    pdf.drawRightString(page_width - margin, page_height - 45, f"{page_number} / {len(problems)}")

    for index, (label, suffix) in enumerate((("Pareto Set (decision space)", "pareto_set"),
                                              ("Pareto Front (objective space)", "pareto_front"))):
        label_y = page_height - 75 - index * 350
        image_path = plot_dir / f"{name.lower()}_{suffix}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing plot: {image_path}")
        pdf.setFillColorRGB(.10, .19, .34)
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(margin, label_y, f"{name} - {label}")
        image_y = label_y - 12 - image_height
        pdf.drawImage(ImageReader(str(image_path)), margin, image_y,
                      width=image_width, height=image_height,
                      preserveAspectRatio=True, anchor="c", mask="auto")

    pdf.setStrokeColorRGB(.82, .86, .90)
    pdf.line(margin, 27, page_width - margin, 27)
    pdf.setFillColorRGB(.30, .38, .50)
    pdf.setFont("Helvetica", 7)
    pdf.drawString(margin, 16, "Blue: obtained solutions | Red: reference Pareto Set / Pareto Front")
    pdf.drawRightString(page_width - margin, 16, "NSGA-II")
    pdf.showPage()

pdf.save()
print(output)
