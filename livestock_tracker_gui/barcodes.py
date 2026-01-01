import os
from barcode import Code128
from barcode.writer import ImageWriter

IMAGES_DIR = "images/barcodes"

def ensure_dirs() -> None:
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

def generate_barcode(tag_id: str) -> str:
    """
    Generate a Code128 barcode PNG for tag_id.
    Returns the file path of the saved image.
    """
    ensure_dirs()
    filename = os.path.join(IMAGES_DIR, tag_id)
    obj = Code128(tag_id, writer=ImageWriter())
    path = obj.save(filename, {"text": tag_id, "module_width": 0.3, "module_height": 10.0})
    return path  # images/barcodes/<TAG>.png
