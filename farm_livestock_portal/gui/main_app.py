# gui/main_app.py

import tkinter as tk
from tkinter import ttk, messagebox
from services.db_service import fetch_all_livestock
from gui.livestock_form import open_livestock_form, generate_barcode, generate_qr
import os

BARCODE_DIR = "assets/barcodes/"

class FarmApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Farm Livestock Portal")
        # Resize to fit screen and allow resizing
        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_w}x{screen_h}")
        self.root.resizable(True, True)

        self.tree = ttk.Treeview(root, columns=("ID", "Tag", "Type", "Breed", "Age", "Health", "Date"), show="headings", selectmode="extended")
        for col in self.tree['columns']:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)
        self.tree.pack(fill="both", expand=True, padx=10, pady=10)

        button_frame = tk.Frame(root)
        button_frame.pack(pady=5)

        add_button = tk.Button(button_frame, text="Add Livestock", command=self.open_form, bg="#2196F3", fg="white", padx=10, pady=5)
        add_button.pack(side="left", padx=5)

        barcode_button = tk.Button(button_frame, text="Generate Barcodes for Selected", command=self.generate_barcode_selected, bg="#FF9800", fg="white", padx=10, pady=5)
        barcode_button.pack(side="left", padx=5)

        qr_button = tk.Button(button_frame, text="Generate QR Codes for Selected", command=self.generate_qr_selected, bg="#795548", fg="white", padx=10, pady=5)
        qr_button.pack(side="left", padx=5)

        self.refresh_table()

    def refresh_table(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        rows = fetch_all_livestock()
        for row in rows:
            self.tree.insert("", "end", values=row)

    def open_form(self):
        open_livestock_form(self.root, self.refresh_table)

    def generate_barcode_selected(self):
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Select at least one animal")
            return
        files = []
        for item in selection:
            values = self.tree.item(item, "values")
            tag = values[1]
            files.append(generate_barcode(tag))
        messagebox.showinfo("Success", f"Generated {len(files)} barcodes. Last: {files[-1]}")

    def generate_qr_selected(self):
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Select at least one animal")
            return
        files = []
        for item in selection:
            values = self.tree.item(item, "values")
            tag = values[1]
            species = values[2]
            breed = values[3]
            files.append(generate_qr(tag, species, breed))
        messagebox.showinfo("Success", f"Generated {len(files)} QR codes. Last: {files[-1]}")

if __name__ == "__main__":
    os.makedirs(BARCODE_DIR, exist_ok=True)
    root = tk.Tk()
    app = FarmApp(root)
    root.mainloop()
