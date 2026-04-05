"""Generate sample medical report PDFs for testing."""
from fpdf import FPDF

def create_pdf(input_txt, output_pdf):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Courier", size=9)
    
    with open(input_txt, "r", encoding="utf-8") as f:
        for line in f:
            # Replace any non-latin1 chars
            safe_line = line.rstrip().encode("latin-1", errors="replace").decode("latin-1")
            pdf.cell(0, 4, txt=safe_line, ln=True)
    
    pdf.output(output_pdf)
    print("Created:", output_pdf)

create_pdf("sample_reports/blood_test_report.txt", "sample_reports/blood_test_report.pdf")
create_pdf("sample_reports/urine_report.txt", "sample_reports/urine_report.pdf")
print("Done! Upload these PDFs in the Streamlit sidebar.")
