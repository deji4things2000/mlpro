# gui/main_app.py

import tkinter as tk
from tkinter import ttk, messagebox
from services.db_service import fetch_all_livestock, delete_livestock_by_id
from gui.livestock_form import open_livestock_form, generate_barcode, generate_qr
import os
from PIL import Image, ImageTk

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

        # Improve base styles
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except Exception:
            pass
        style.configure("Treeview", rowheight=26, font=("Helvetica", 11))
        style.configure("Treeview.Heading", font=("Helvetica", 12, "bold"))

        header = ttk.Label(root, text="Farm Livestock Portal", font=("Helvetica", 18, "bold"))
        header.pack(pady=(10, 5))

        # Search / filter bar
        search_frame = ttk.Frame(root)
        search_frame.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Label(search_frame, text="Search:").pack(side="left")
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=30)
        search_entry.pack(side="left", padx=6)
        self.search_field_var = tk.StringVar(value="Tag")
        field_box = ttk.Combobox(search_frame, textvariable=self.search_field_var, values=["Tag", "Type", "Breed", "Health"], width=12, state="readonly")
        field_box.pack(side="left")
        ttk.Button(search_frame, text="Apply", command=self.apply_filter).pack(side="left", padx=6)
        ttk.Button(search_frame, text="Clear", command=self.clear_filter).pack(side="left")

        self.tree = ttk.Treeview(root, columns=("ID", "Tag", "Type", "Breed", "Age", "Health", "Date"), show="headings", selectmode="extended")
        self.sort_reverse = {}
        for col in self.tree['columns']:
            self.tree.heading(col, text=col, command=lambda c=col: self.sort_by(c))
            default_width = 100
            if col in ("Tag", "Breed"):
                default_width = 160
            if col == "Health":
                default_width = 140
            self.tree.column(col, width=default_width, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=10, pady=10)

        # Striped rows
        self.tree.tag_configure('odd', background='#f7f7f7')
        self.tree.tag_configure('even', background='#ffffff')

        button_frame = ttk.Frame(root)
        button_frame.pack(pady=5)

        add_button = ttk.Button(button_frame, text="Add Livestock", command=self.open_form)
        add_button.pack(side="left", padx=5)

        barcode_button = ttk.Button(button_frame, text="Generate Barcodes for Selected", command=self.generate_barcode_selected)
        barcode_button.pack(side="left", padx=5)

        qr_button = ttk.Button(button_frame, text="Generate QR Codes for Selected", command=self.generate_qr_selected)
        qr_button.pack(side="left", padx=5)

        preview_button = ttk.Button(button_frame, text="Preview Barcode", command=self.preview_selected_barcode)
        preview_button.pack(side="left", padx=5)

        preview_qr_button = ttk.Button(button_frame, text="Preview QR", command=self.preview_selected_qr)
        preview_qr_button.pack(side="left", padx=5)

        delete_button = ttk.Button(button_frame, text="Delete Selected", command=self.delete_selected)
        delete_button.pack(side="left", padx=5)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status = ttk.Label(root, textvariable=self.status_var, anchor="w")
        status.pack(fill="x", padx=10, pady=(0, 8))

        self.refresh_table()

    def refresh_table(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        rows = fetch_all_livestock()
        self.all_rows = rows
        for idx, row in enumerate(rows):
            tag = 'odd' if idx % 2 else 'even'
            self.tree.insert("", "end", values=row, tags=(tag,))
        self.status_var.set(f"{len(rows)} record(s)")

    def apply_filter(self):
        query = (self.search_var.get() or "").strip().lower()
        field = self.search_field_var.get()
        index_map = {"Tag": 1, "Type": 2, "Breed": 3, "Health": 5}
        idx = index_map.get(field, 1)
        if not query:
            self.refresh_table()
            return
        filtered = [r for r in getattr(self, 'all_rows', []) if query in str(r[idx]).lower()]
        for i in self.tree.get_children():
            self.tree.delete(i)
        for i, row in enumerate(filtered):
            tag = 'odd' if i % 2 else 'even'
            self.tree.insert("", "end", values=row, tags=(tag,))
        self.status_var.set(f"Filtered: {len(filtered)} record(s)")

    def clear_filter(self):
        self.search_var.set("")
        self.apply_filter()

    def sort_by(self, col):
        # Determine column index
        cols = ["ID", "Tag", "Type", "Breed", "Age", "Health", "Date"]
        idx = cols.index(col)
        items = [(self.tree.item(c, 'values')[idx], c) for c in self.tree.get_children('')]
        reverse = not self.sort_reverse.get(col, False)
        try:
            items.sort(key=lambda x: float(x[0]), reverse=reverse)
        except Exception:
            items.sort(key=lambda x: str(x[0]).lower(), reverse=reverse)
        for i, (_, c) in enumerate(items):
            self.tree.move(c, '', i)
        self.sort_reverse[col] = reverse

    def open_form(self):
        open_livestock_form(self.root, self.refresh_table)

    def generate_barcode_selected(self):
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Select at least one animal")
            return
        files = []
        last_tag = None
        for item in selection:
            values = self.tree.item(item, "values")
            tag = values[1]
            last_tag = tag
            files.append(generate_barcode(tag))
        messagebox.showinfo("Success", f"Generated {len(files)} barcodes.")
        # Preview last generated barcode
        try:
            self.preview_image(files[-1], title=f"Barcode: {last_tag}")
        except Exception:
            pass

    def generate_qr_selected(self):
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Select at least one animal")
            return
        files = []
        last_tag = None
        for item in selection:
            values = self.tree.item(item, "values")
            tag = values[1]
            species = values[2]
            breed = values[3]
            last_tag = tag
            files.append(generate_qr(tag, species, breed))
        messagebox.showinfo("Success", f"Generated {len(files)} QR codes.")
        try:
            self.preview_image(files[-1], title=f"QR: {last_tag}")
        except Exception:
            pass

    def preview_image(self, image_path, title="Preview"):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("500x300")
        img = Image.open(image_path)
        photo = ImageTk.PhotoImage(img)
        win.img_ref = photo
        lbl = ttk.Label(win, image=photo)
        lbl.pack(padx=10, pady=10)

    def preview_selected_barcode(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Select an animal to preview")
            return
        values = self.tree.item(selected[0], "values")
        tag = values[1]
        path = os.path.join(BARCODE_DIR, f"{tag}.png")
        if not os.path.exists(path):
            path = generate_barcode(tag)
        self.preview_image(path, title=f"Barcode: {tag}")

    def preview_selected_qr(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Select an animal to preview")
            return
        values = self.tree.item(selected[0], "values")
        tag = values[1]
        species = values[2]
        breed = values[3]
        path = os.path.join(BARCODE_DIR, f"{tag}_qr.png")
        if not os.path.exists(path):
            path = generate_qr(tag, species, breed)
        self.preview_image(path, title=f"QR: {tag}")

    def delete_selected(self):
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Select at least one animal to delete")
            return
        if not messagebox.askyesno("Confirm Delete", f"Delete {len(selection)} selected record(s)? This cannot be undone."):
            return
        errors = 0
        for item in selection:
            values = self.tree.item(item, "values")
            try:
                livestock_id = int(values[0])
                delete_livestock_by_id(livestock_id)
            except Exception as e:
                errors += 1
        self.refresh_table()
        if errors:
            messagebox.showerror("Delete", f"Completed with {errors} error(s).")
        else:
            messagebox.showinfo("Delete", "Selected record(s) deleted.")

if __name__ == "__main__":
    os.makedirs(BARCODE_DIR, exist_ok=True)
    root = tk.Tk()
    app = FarmApp(root)
    root.mainloop()
