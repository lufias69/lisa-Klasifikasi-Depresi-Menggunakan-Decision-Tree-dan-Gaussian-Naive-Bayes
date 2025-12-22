"""
Script untuk convert PNG diagram ke PDF
"""
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

# File paths
png_file = "DIAGRAM_ALUR_PENELITIAN.png"
pdf_file = "DIAGRAM_ALUR_PENELITIAN.pdf"

try:
    # Buka gambar
    img = Image.open(png_file)
    img_width, img_height = img.size
    
    print(f"📐 Ukuran diagram: {img_width} x {img_height} pixels")
    
    # Gunakan portrait A4
    page_width, page_height = A4
    
    # Minimal margin
    margin = 20
    max_width = page_width - (2 * margin)
    
    # Scale berdasarkan lebar saja untuk memaksimalkan penggunaan halaman
    ratio = max_width / img_width
    
    # Ukuran akhir
    final_width = img_width * ratio
    final_height = img_height * ratio
    
    # Hitung jumlah halaman yang dibutuhkan
    pages_needed = (final_height + margin) / (page_height - 2 * margin)
    
    print(f"📄 Membuat PDF portrait A4...")
    print(f"📊 Diagram akan menggunakan ~{pages_needed:.1f} halaman")
    
    # Posisi x (center horizontal)
    x = (page_width - final_width) / 2
    
    # Jika tinggi melebihi 1 halaman, buat custom page height
    if final_height > (page_height - 2 * margin):
        # Buat halaman dengan tinggi custom
        custom_height = final_height + (2 * margin) + 40  # Extra untuk judul
        c = canvas.Canvas(pdf_file, pagesize=(page_width, custom_height))
        
        # Judul di atas
        c.setFont("Helvetica-Bold", 16)
        title = "Diagram Alur Penelitian"
        c.drawCentredString(page_width/2, custom_height - 30, title)
        
        # Gambar mulai dari bawah judul
        y = margin
        c.drawImage(png_file, x, y, width=final_width, height=final_height, preserveAspectRatio=True)
        
        print(f"📊 Ukuran PDF: {page_width:.0f} x {custom_height:.0f} points")
    else:
        # Fit dalam 1 halaman A4 portrait
        c = canvas.Canvas(pdf_file, pagesize=A4)
        
        # Judul
        c.setFont("Helvetica-Bold", 16)
        title = "Diagram Alur Penelitian"
        c.drawCentredString(page_width/2, page_height - 30, title)
        
        # Gambar
        y = (page_height - final_height - 40) / 2
        c.drawImage(png_file, x, y, width=final_width, height=final_height, preserveAspectRatio=True)
        
        print(f"📊 Ukuran PDF: {page_width:.0f} x {page_height:.0f} points (A4 Portrait)")
    
    # Save
    c.save()
    
    print(f"✅ Berhasil! PDF tersimpan sebagai: {pdf_file}")
    
except FileNotFoundError:
    print(f"❌ File {png_file} tidak ditemukan!")
except Exception as e:
    print(f"❌ Error: {e}")
