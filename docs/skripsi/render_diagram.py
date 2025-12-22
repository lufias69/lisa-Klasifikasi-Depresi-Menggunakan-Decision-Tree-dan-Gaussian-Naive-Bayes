"""
Script untuk render diagram mermaid menjadi gambar SVG (lebih tajam)
"""
import base64
import urllib.request
import urllib.parse
import json

# Baca file markdown
with open('DIAGRAM_ALUR_PENELITIAN.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Ekstrak kode mermaid
start = content.find('```mermaid')
end = content.find('```', start + 10)
mermaid_code = content[start+10:end].strip()

# Encode mermaid code untuk URL
encoded = base64.urlsafe_b64encode(mermaid_code.encode('utf-8')).decode('ascii')

# URL untuk mermaid.ink dengan background putih dan format SVG
svg_url = f"https://mermaid.ink/svg/{encoded}"
png_url = f"https://mermaid.ink/img/{encoded}"

print(f"📥 Downloading diagram dalam format SVG (tajam & scalable)...")

try:
    # Download SVG
    urllib.request.urlretrieve(svg_url, 'DIAGRAM_ALUR_PENELITIAN.svg')
    print(f"✅ SVG tersimpan sebagai DIAGRAM_ALUR_PENELITIAN.svg")
    
    # Download PNG standar
    print(f"📥 Downloading PNG...")
    urllib.request.urlretrieve(png_url, 'DIAGRAM_ALUR_PENELITIAN.png')
    print(f"✅ PNG tersimpan sebagai DIAGRAM_ALUR_PENELITIAN.png")
    
    print(f"\n✨ Diagram dengan background putih dan kualitas tinggi berhasil dibuat!")
    print(f"💡 File SVG memberikan kualitas terbaik untuk PDF")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print(f"\nAlternatif: Buka URL ini di browser untuk download manual:")
    print(f"SVG: {svg_url}")
    print(f"PNG: {png_url}")
