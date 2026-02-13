import os
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Preformatted,
    PageBreak,
    HRFlowable,
)
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

OUTPUT_FILE = "all_python_code.pdf"

# Folders to ignore
IGNORE_DIRS = {"venv", ".venv", "__pycache__", ".git", "build", "dist"}


def get_python_files():
    current_script = os.path.basename(__file__)
    py_files = []

    for root, dirs, files in os.walk("."):
        # Remove ignored directories from traversal
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            if file.endswith(".py") and file != current_script:
                full_path = os.path.join(root, file)
                py_files.append(full_path)

    return sorted(py_files)


def create_pdf(py_files):
    doc = SimpleDocTemplate(OUTPUT_FILE)
    elements = []

    styles = getSampleStyleSheet()

    header_style = styles["Heading1"]

    code_style = ParagraphStyle(
        name="CodeStyle",
        fontName="Courier",
        fontSize=8,
        leading=10,
    )

    for idx, file_path in enumerate(py_files):
        relative_path = os.path.relpath(file_path)

        # Section title = file path
        elements.append(Paragraph(relative_path, header_style))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
        elements.append(Spacer(1, 0.2 * inch))

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        elements.append(Preformatted(code, code_style))

        if idx < len(py_files) - 1:
            elements.append(PageBreak())

    doc.build(elements)


if __name__ == "__main__":
    py_files = get_python_files()

    if not py_files:
        print("No Python files found.")
    else:
        create_pdf(py_files)
        print(f"PDF created successfully: {OUTPUT_FILE}")
