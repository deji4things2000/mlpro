# gui/livestock_form.py

import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
from services.db_service import insert_livestock_extended
import random
import string
import os
import csv
import barcode
from barcode.writer import ImageWriter
import qrcode
import urllib.parse

# Species/Breed source CSV (FAO list)
SPECIES_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fao_dad_list - Species.csv")

def load_species_breeds():
    """
    Load species→breeds mapping from CSV with headers: Species,Breed.
    Falls back to a small default map if CSV is missing.
    """
    mapping = {}
    try:
        with open(SPECIES_CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                species = (row.get("Species") or "").strip()
                breed = (row.get("Breed") or "").strip()
                if not species or not breed:
                    continue
                mapping.setdefault(species, []).append(breed)
        if mapping:
            return mapping
    except Exception:
        pass
    # Fallback minimal list if CSV cannot be read
    return {
        "Cow": ["Holstein", "Jersey", "Angus", "Hereford"],
        "Goat": ["Boer", "Kiko", "Saanen", "Nubian"],
        "Sheep": ["Merino", "Suffolk", "Dorper", "Hampshire"],
        "Pig": ["Berkshire", "Yorkshire", "Landrace", "Duroc"],
        "Chicken": ["Leghorn", "Rhode Island Red", "Plymouth Rock", "Orpington"],
    }

LIVESTOCK_BREEDS = load_species_breeds()
DEFAULT_SPECIES = next(iter(LIVESTOCK_BREEDS.keys()), "Cow")
DEFAULT_BREED = (LIVESTOCK_BREEDS.get(DEFAULT_SPECIES) or [""])[0]

BARCODE_DIR = "assets/barcodes/"
os.makedirs(BARCODE_DIR, exist_ok=True)

def generate_tag(animal_type):
    """Generate random unique tag"""
    random_part = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"{animal_type[:3].upper()}-{random_part}"

def generate_barcode(tag):
    """Generate barcode PNG for a given tag"""
    CODE128 = barcode.get_barcode_class('code128')
    code = CODE128(tag, writer=ImageWriter())
    filename = os.path.join(BARCODE_DIR, tag)
    code.save(filename)
    return filename + ".png"

def breed_search_url(species, breed):
    query = f"{species} {breed} livestock breed"
    # Google Images search
    return "https://www.google.com/search?tbm=isch&q=" + urllib.parse.quote_plus(query)

def generate_qr(tag, species=None, breed=None):
    """Generate QR code PNG; encode breed search link when provided."""
    data = tag
    if species and breed:
        data = breed_search_url(species, breed)
    img = qrcode.make(data)
    filename = os.path.join(BARCODE_DIR, f"{tag}_qr.png")
    img.save(filename)
    return filename

def open_livestock_form(parent, refresh_callback):
    window = tk.Toplevel(parent)
    window.title("Add New Livestock")
    # Make the form larger and resizable so calendar fits
    window.geometry("700x800")
    window.resizable(True, True)

    font_style = ("Helvetica", 11)

    # Generate random tag initially based on available species
    animal_type_default = DEFAULT_SPECIES
    tag_var = tk.StringVar(value=generate_tag(animal_type_default))

    tk.Label(window, text="Animal Tag", font=font_style).pack(pady=5)
    tag_entry = tk.Entry(window, textvariable=tag_var, font=font_style, state="readonly")
    tag_entry.pack(fill="x", padx=20)

    tk.Label(window, text="Species", font=font_style).pack(pady=5)
    type_var = tk.StringVar(value=animal_type_default)
    type_dropdown = ttk.Combobox(window, textvariable=type_var, values=list(LIVESTOCK_BREEDS.keys()), state="readonly", font=font_style)
    type_dropdown.pack(fill="x", padx=20)

    tk.Label(window, text="Breed", font=font_style).pack(pady=5)
    breed_var = tk.StringVar(value=DEFAULT_BREED)
    breed_dropdown = ttk.Combobox(window, textvariable=breed_var, values=LIVESTOCK_BREEDS.get(animal_type_default, []), state="readonly", font=font_style)
    breed_dropdown.pack(fill="x", padx=20)

    # Livestock Type (category)
    tk.Label(window, text="Livestock Type", font=font_style).pack(pady=5)
    type_options = ["Dairy", "Beef", "Breeding", "Draft", "Layer", "Broiler", "Wool", "Pack", "Companion"]
    livestock_type_var = tk.StringVar()
    livestock_type_dropdown = ttk.Combobox(window, textvariable=livestock_type_var, values=type_options, font=font_style)
    livestock_type_dropdown.pack(fill="x", padx=20)

    # Color
    tk.Label(window, text="Color", font=font_style).pack(pady=5)
    color_options = ["Black", "White", "Brown", "Red", "Grey", "Tan", "Gold", "Speckled"]
    color_var = tk.StringVar()
    color_dropdown = ttk.Combobox(window, textvariable=color_var, values=color_options, font=font_style)
    color_dropdown.pack(fill="x", padx=20)

    tk.Label(window, text="Age (years)", font=font_style).pack(pady=5)
    age_var = tk.StringVar()
    age_entry = tk.Entry(window, textvariable=age_var, font=font_style)
    age_entry.pack(fill="x", padx=20)

    tk.Label(window, text="Health Status", font=font_style).pack(pady=5)
    health_var = tk.StringVar()
    health_entry = tk.Entry(window, textvariable=health_var, font=font_style)
    health_entry.pack(fill="x", padx=20)

    tk.Label(window, text="Purchase Date", font=font_style).pack(pady=5)
    date_entry = DateEntry(window, date_pattern='yyyy-mm-dd', font=font_style)
    date_entry.pack(fill="x", padx=20)

    def on_type_change(event):
        selected_type = type_var.get()
        breeds = LIVESTOCK_BREEDS.get(selected_type, [])
        breed_dropdown['values'] = breeds
        if breeds:
            breed_var.set(breeds[0])
        else:
            breed_var.set("")
        tag_var.set(generate_tag(selected_type))

    type_dropdown.bind("<<ComboboxSelected>>", on_type_change)

    def save():
        try:
            age = int(age_var.get())
            # Validate breed belongs to selected species
            selected_species = type_var.get()
            selected_breed = breed_var.get()
            valid_breeds = LIVESTOCK_BREEDS.get(selected_species, [])
            if selected_breed and selected_breed not in valid_breeds:
                messagebox.showerror("Error", "Selected breed is not valid for the chosen species.")
                return
            data_ext = (
                tag_var.get(),
                selected_species,
                selected_breed,
                age,
                health_var.get(),
                date_entry.get_date(),
                livestock_type_var.get(),
                color_var.get()
            )
            insert_livestock_extended(data_ext)
            generate_barcode(tag_var.get())
            generate_qr(tag_var.get(), selected_species, selected_breed)
            messagebox.showinfo("Success", "Livestock added successfully!")
            refresh_callback()
            window.destroy()
        except ValueError:
            messagebox.showerror("Error", "Age must be a number")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")

    save_button = tk.Button(window, text="Save Livestock", command=save, font=font_style, bg="#4CAF50", fg="white")
    save_button.pack(pady=15, ipadx=10, ipady=5)
