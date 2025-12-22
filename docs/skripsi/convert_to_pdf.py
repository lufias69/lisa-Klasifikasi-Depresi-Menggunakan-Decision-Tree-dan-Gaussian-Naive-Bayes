"""
Script untuk mengkonversi DIAGRAM_ALUR_PENELITIAN.html ke PDF
"""
import os
import pdfkit

try:
    # Path file
    html_file = "DIAGRAM_ALUR_PENELITIAN.html"
    pdf_file = "DIAGRAM_ALUR_PENELITIAN.pdf"
    
    # Konfigurasi pdfkit (cari wkhtmltopdf)
    config = pdfkit.configuration()
    
    # Konversi HTML ke PDF
    pdfkit.from_file(html_file, pdf_file, configuration=config)
    print(f"✅ Berhasil membuat {pdf_file}")
    
except OSError as e:
    print("❌ wkhtmltopdf tidak ditemukan")
    print("Download dari: https://wkhtmltopdf.org/downloads.html")
    print("\nAlternatif: Buka DIAGRAM_ALUR_PENELITIAN.html di browser dan print to PDF")
except Exception as e:
    print(f"❌ Error: {e}")
