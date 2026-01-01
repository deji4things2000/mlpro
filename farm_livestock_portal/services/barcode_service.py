import barcode
from barcode.writer import ImageWriter
import os

def generate_barcode(animal_tag):
    if not os.path.exists("assets/barcodes"):
        os.makedirs("assets/barcodes")

    code128 = barcode.get("code128", animal_tag, writer=ImageWriter())
    file_path = f"assets/barcodes/{animal_tag}"
    code128.save(file_path)
    return file_path
