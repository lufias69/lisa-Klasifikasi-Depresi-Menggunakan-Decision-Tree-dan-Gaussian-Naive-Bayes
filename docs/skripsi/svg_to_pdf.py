"""
Script untuk convert SVG diagram ke PDF dengan kualitas tinggi
"""
import cairosvg

# File paths
svg_file = "DIAGRAM_ALUR_PENELITIAN.svg"
pdf_file = "DIAGRAM_ALUR_PENELITIAN.pdf"

try:
    print(f"📄 Converting SVG ke PDF...")
    print(f"📥 Input: {svg_file}")
    
    # Convert SVG to PDF dengan kualitas tinggi
    cairosvg.svg2pdf(url=svg_file, write_to=pdf_file)
    
    print(f"✅ Berhasil! PDF tersimpan sebagai: {pdf_file}")
    print(f"✨ PDF dengan background putih dan teks yang tajam (tidak blur)")
    
except FileNotFoundError:
    print(f"❌ File {svg_file} tidak ditemukan!")
    print("Jalankan dulu: python render_diagram.py")
except Exception as e:
    print(f"❌ Error: {e}")
