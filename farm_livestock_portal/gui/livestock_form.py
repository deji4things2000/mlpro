# gui/livestock_form.py

import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
from services.db_service import insert_livestock_extended, get_livestock_by_id, update_livestock_extended
import random
import string
import os
import csv
import barcode
from barcode.writer import ImageWriter
import qrcode
import urllib.parse
from PIL import Image, ImageTk
from gui.styles import apply_base_styles

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

def open_livestock_form(parent, refresh_callback, current_user: str | None = None):
    window = tk.Toplevel(parent)
    window.title("Add New Livestock")
    # Make the form larger and resizable so calendar fits
    window.geometry("700x800")
    window.resizable(True, True)

    try:
        apply_base_styles(window)
    except Exception:
        pass

    def close_window(event=None):
        try:
            window.destroy()
        except Exception:
            pass

    window.bind("<Escape>", close_window)
    window.protocol("WM_DELETE_WINDOW", close_window)

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

    tk.Label(window, text="Date of Birth", font=font_style).pack(pady=5)
    dob_entry = DateEntry(window, date_pattern='yyyy-mm-dd', font=font_style)
    dob_entry.pack(fill="x", padx=20)

    tk.Label(window, text="Age (years)", font=font_style).pack(pady=5)
    age_var = tk.StringVar()
    age_entry = tk.Entry(window, textvariable=age_var, font=font_style)
    age_entry.pack(fill="x", padx=20)

    tk.Label(window, text="Health Status", font=font_style).pack(pady=5)
    health_options = ["Healthy", "Sick", "Recovering", "Injured", "Under Treatment", "Unknown"]
    health_var = tk.StringVar(value="Healthy")
    health_dropdown = ttk.Combobox(window, textvariable=health_var, values=health_options, state="readonly", font=font_style)
    health_dropdown.pack(fill="x", padx=20)

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

    def _update_age_from_dob(event=None):
        try:
            from datetime import date
            dob = dob_entry.get_date()
            today = date.today()
            years = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            if years < 0:
                years = 0
            age_var.set(str(years))
        except Exception:
            pass

    # Update age when a DOB is selected from the calendar
    try:
        dob_entry.bind("<<DateEntrySelected>>", _update_age_from_dob)
    except Exception:
        pass

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
                dob_entry.get_date(),
                health_var.get(),
                date_entry.get_date(),
                livestock_type_var.get(),
                color_var.get(),
                (current_user or "")
            )
            insert_livestock_extended(data_ext)
            barcode_path = generate_barcode(tag_var.get())
            qr_path = generate_qr(tag_var.get(), selected_species, selected_breed)
            messagebox.showinfo("Success", "Livestock added successfully!")
            # Preview the generated barcode
            try:
                preview = tk.Toplevel(window)
                preview.title(f"Barcode: {tag_var.get()}")
                img = Image.open(barcode_path)
                # Barcodes are wide; limit height while allowing width
                try:
                    img.thumbnail((800, 300), Image.LANCZOS)
                except Exception:
                    img.thumbnail((800, 300))
                photo = ImageTk.PhotoImage(img)
                # Keep a reference to avoid garbage collection
                preview.img_ref = photo
                lbl = ttk.Label(preview, image=photo)
                lbl.pack(padx=10, pady=10)
                preview.geometry(f"{img.width + 40}x{img.height + 80}")
                preview.resizable(True, True)
            except Exception:
                pass
            # Preview the generated QR code
            try:
                qr_prev = tk.Toplevel(window)
                qr_prev.title(f"QR: {tag_var.get()}")
                qimg = Image.open(qr_path)
                # QR codes are square; cap at 600x600
                try:
                    qimg.thumbnail((600, 600), Image.LANCZOS)
                except Exception:
                    qimg.thumbnail((600, 600))
                qphoto = ImageTk.PhotoImage(qimg)
                qr_prev.img_ref = qphoto
                qlbl = ttk.Label(qr_prev, image=qphoto)
                qlbl.pack(padx=10, pady=10)
                qr_prev.geometry(f"{qimg.width + 40}x{qimg.height + 80}")
                qr_prev.resizable(True, True)
            except Exception:
                pass
            refresh_callback()
            window.destroy()
        except ValueError:
            messagebox.showerror("Error", "Age must be a number")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")

    btns = ttk.Frame(window)
    btns.pack(pady=15)
    save_button = ttk.Button(btns, text="Save Livestock", command=save, style="Primary.TButton")
    save_button.pack(side="left", padx=10)
    cancel_button = ttk.Button(btns, text="Cancel", command=close_window, style="Secondary.TButton")
    cancel_button.pack(side="left", padx=10)


def open_edit_livestock_form(parent, refresh_callback, livestock_id):
    window = tk.Toplevel(parent)
    window.title("Edit Livestock")
    window.geometry("700x800")
    window.resizable(True, True)

    try:
        apply_base_styles(window)
    except Exception:
        pass

    def close_window(event=None):
        try:
            window.destroy()
        except Exception:
            pass

    window.bind("<Escape>", close_window)
    window.protocol("WM_DELETE_WINDOW", close_window)

    font_style = ("Helvetica", 11)

    record = get_livestock_by_id(livestock_id)
    if not record:
        messagebox.showerror("Error", "Could not load selected record")
        window.destroy()
        return
    # record: (id, tag, species, breed, age, health, purchase_date, livestock_type, color)
    # record may include created_by at the end; ignore it for editing
    _, tag_val, species_val, breed_val, age_val, dob_val, health_val, purchase_date_val, livestock_type_val, color_val, *_rest = record

    tk.Label(window, text="Animal Tag", font=font_style).pack(pady=5)
    tag_var = tk.StringVar(value=tag_val)
    tag_entry = tk.Entry(window, textvariable=tag_var, font=font_style, state="readonly")
    tag_entry.pack(fill="x", padx=20)

    tk.Label(window, text="Species", font=font_style).pack(pady=5)
    type_var = tk.StringVar(value=species_val)
    type_dropdown = ttk.Combobox(window, textvariable=type_var, values=list(LIVESTOCK_BREEDS.keys()), state="readonly", font=font_style)
    type_dropdown.pack(fill="x", padx=20)

    tk.Label(window, text="Breed", font=font_style).pack(pady=5)
    breeds_list = LIVESTOCK_BREEDS.get(species_val, [])
    breed_var = tk.StringVar(value=(breed_val or (breeds_list[0] if breeds_list else "")))
    breed_dropdown = ttk.Combobox(window, textvariable=breed_var, values=breeds_list, state="readonly", font=font_style)
    breed_dropdown.pack(fill="x", padx=20)

    tk.Label(window, text="Livestock Type", font=font_style).pack(pady=5)
    type_options = ["Dairy", "Beef", "Breeding", "Draft", "Layer", "Broiler", "Wool", "Pack", "Companion"]
    livestock_type_var = tk.StringVar(value=(livestock_type_val or ""))
    livestock_type_dropdown = ttk.Combobox(window, textvariable=livestock_type_var, values=type_options, font=font_style)
    livestock_type_dropdown.pack(fill="x", padx=20)

    tk.Label(window, text="Color", font=font_style).pack(pady=5)
    color_options = ["Black", "White", "Brown", "Red", "Grey", "Tan", "Gold", "Speckled"]
    color_var = tk.StringVar(value=(color_val or ""))
    color_dropdown = ttk.Combobox(window, textvariable=color_var, values=color_options, font=font_style)
    color_dropdown.pack(fill="x", padx=20)

    tk.Label(window, text="Date of Birth", font=font_style).pack(pady=5)
    dob_entry = DateEntry(window, date_pattern='yyyy-mm-dd', font=font_style)
    try:
        if dob_val:
            dob_entry.set_date(dob_val)
    except Exception:
        pass
    dob_entry.pack(fill="x", padx=20)

    tk.Label(window, text="Age (years)", font=font_style).pack(pady=5)
    age_var = tk.StringVar(value=str(age_val or ""))
    age_entry = tk.Entry(window, textvariable=age_var, font=font_style)
    age_entry.pack(fill="x", padx=20)

    tk.Label(window, text="Health Status", font=font_style).pack(pady=5)
    health_options = ["Healthy", "Sick", "Recovering", "Injured", "Under Treatment", "Unknown"]
    health_var = tk.StringVar(value=(health_val or "Healthy"))
    health_dropdown = ttk.Combobox(window, textvariable=health_var, values=health_options, state="readonly", font=font_style)
    health_dropdown.pack(fill="x", padx=20)

    tk.Label(window, text="Purchase Date", font=font_style).pack(pady=5)
    date_entry = DateEntry(window, date_pattern='yyyy-mm-dd', font=font_style)
    try:
        if purchase_date_val:
            date_entry.set_date(purchase_date_val)
    except Exception:
        pass
    date_entry.pack(fill="x", padx=20)

    def on_type_change(event):
        selected_type = type_var.get()
        breeds = LIVESTOCK_BREEDS.get(selected_type, [])
        breed_dropdown['values'] = breeds
        if breeds:
            # Keep existing breed if valid, else set first
            if breed_var.get() not in breeds:
                breed_var.set(breeds[0])
        else:
            breed_var.set("")

    type_dropdown.bind("<<ComboboxSelected>>", on_type_change)

    def _update_age_from_dob(event=None):
        try:
            from datetime import date
            dob = dob_entry.get_date()
            today = date.today()
            years = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            if years < 0:
                years = 0
            age_var.set(str(years))
        except Exception:
            pass

    try:
        dob_entry.bind("<<DateEntrySelected>>", _update_age_from_dob)
    except Exception:
        pass

    def save():
        try:
            age = int(age_var.get()) if (age_var.get() or "").strip() != "" else None
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
                dob_entry.get_date(),
                health_var.get(),
                date_entry.get_date(),
                livestock_type_var.get(),
                color_var.get(),
                livestock_id,
            )
            update_livestock_extended(data_ext)
            messagebox.showinfo("Success", "Livestock updated successfully!")
            refresh_callback()
            window.destroy()
        except ValueError:
            messagebox.showerror("Error", "Age must be a number")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update: {e}")

    btns = ttk.Frame(window)
    btns.pack(pady=15)
    save_button = ttk.Button(btns, text="Save Changes", command=save, style="Primary.TButton")
    save_button.pack(side="left", padx=10)
    cancel_button = ttk.Button(btns, text="Cancel", command=close_window, style="Secondary.TButton")
    cancel_button.pack(side="left", padx=10)
