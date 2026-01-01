# gui/livestock_form.py

import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
from services.db_service import insert_livestock
import random
import string
import os
import barcode
from barcode.writer import ImageWriter

# Predefined livestock types and breeds
LIVESTOCK_BREEDS = {
    "Cow": ["Holstein", "Jersey", "Angus", "Hereford"],
    "Goat": ["Boer", "Kiko", "Saanen", "Nubian"],
    "Sheep": ["Merino", "Suffolk", "Dorper", "Hampshire"],
    "Pig": ["Berkshire", "Yorkshire", "Landrace", "Duroc"],
    "Chicken": ["Leghorn", "Rhode Island Red", "Plymouth Rock", "Orpington"],
    "Duck": ["Pekin", "Muscovy", "Khaki Campbell"],
    "Horse": ["Thoroughbred", "Arabian", "Quarter Horse"],
    "Camel": ["Dromedary", "Bactrian"]
}

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

def open_livestock_form(parent, refresh_callback):
    window = tk.Toplevel(parent)
    window.title("Add New Livestock")
    window.geometry("400x400")
    window.resizable(False, False)

    font_style = ("Helvetica", 11)

    # Generate random tag initially
    animal_type_default = "Cow"
    tag_var = tk.StringVar(value=generate_tag(animal_type_default))

    tk.Label(window, text="Animal Tag", font=font_style).pack(pady=5)
    tag_entry = tk.Entry(window, textvariable=tag_var, font=font_style, state="readonly")
    tag_entry.pack(fill="x", padx=20)

    tk.Label(window, text="Animal Type", font=font_style).pack(pady=5)
    type_var = tk.StringVar(value=animal_type_default)
    type_dropdown = ttk.Combobox(window, textvariable=type_var, values=list(LIVESTOCK_BREEDS.keys()), state="readonly", font=font_style)
    type_dropdown.pack(fill="x", padx=20)

    tk.Label(window, text="Breed", font=font_style).pack(pady=5)
    breed_var = tk.StringVar(value=LIVESTOCK_BREEDS[animal_type_default][0])
    breed_dropdown = ttk.Combobox(window, textvariable=breed_var, values=LIVESTOCK_BREEDS[animal_type_default], state="readonly", font=font_style)
    breed_dropdown.pack(fill="x", padx=20)

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
        breed_dropdown['values'] = LIVESTOCK_BREEDS[selected_type]
        breed_var.set(LIVESTOCK_BREEDS[selected_type][0])
        tag_var.set(generate_tag(selected_type))

    type_dropdown.bind("<<ComboboxSelected>>", on_type_change)

    def save():
        try:
            age = int(age_var.get())
            data = (
                tag_var.get(),
                type_var.get(),
                breed_var.get(),
                age,
                health_var.get(),
                date_entry.get_date()
            )
            insert_livestock(data)
            generate_barcode(tag_var.get())
            messagebox.showinfo("Success", "Livestock added successfully!")
            refresh_callback()
            window.destroy()
        except ValueError:
            messagebox.showerror("Error", "Age must be a number")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")

    save_button = tk.Button(window, text="Save Livestock", command=save, font=font_style, bg="#4CAF50", fg="white")
    save_button.pack(pady=15, ipadx=10, ipady=5)
