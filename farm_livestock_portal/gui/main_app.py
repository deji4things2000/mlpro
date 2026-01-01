# gui/main_app.py

import tkinter as tk
from tkinter import ttk, messagebox
from services.db_service import fetch_all_livestock
from gui.livestock_form import open_livestock_form, generate_barcode
import os

BARCODE_DIR = "assets/barcodes/"

class FarmApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Farm Livestock Portal")
        self.root.geometry("800x500")
        self.root.resizable(False, False)

        self.tree = ttk.Treeview(root, columns=("ID", "Tag", "Type", "Breed", "Age", "Health", "Date"), show="headings")
        for col in self.tree['columns']:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)
        self.tree.pack(fill="both", expand=True, padx=10, pady=10)

        button_frame = tk.Frame(root)
        button_frame.pack(pady=5)

        add_button = tk.Button(button_frame, text="Add Livestock", command=self.open_form, bg="#2196F3", fg="white", padx=10, pady=5)
        add_button.pack(side="left", padx=5)

        barcode_button = tk.Button(button_frame, text="Generate Barcode for Selected", command=self.generate_barcode_selected, bg="#FF9800", fg="white", padx=10, pady=5)
        barcode_button.pack(side="left", padx=5)

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
        selected = self.tree.focus()
        if not selected:
            messagebox.showwarning("Warning", "Select an animal first")
            return
        values = self.tree.item(selected, "values")
        tag = values[1]
        filename = generate_barcode(tag)
        messagebox.showinfo("Success", f"Barcode generated at {filename}")

if __name__ == "__main__":
    os.makedirs(BARCODE_DIR, exist_ok=True)
    root = tk.Tk()
    app = FarmApp(root)
    root.mainloop()
