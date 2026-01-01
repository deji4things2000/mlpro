# gui/health_records.py

import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
import os
import csv
from services.health_service import (
    ensure_health_table,
    fetch_health_records,
    insert_health_record,
    delete_health_record,
    update_health_record,
)
from services.db_service import fetch_all_livestock
from gui.styles import apply_base_styles

SPECIES_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fao_dad_list - Species.csv")


def load_species_list():
    species = set()
    try:
        with open(SPECIES_CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                s = (row.get("Species") or "").strip()
                if s:
                    species.add(s)
    except Exception:
        # Fallback minimal list
        species = {"Cow", "Goat", "Sheep", "Pig", "Chicken"}
    return sorted(species)


class HealthRecordsWindow:
    def __init__(self, parent, refresh_callback=None):
        self.parent = parent
        self.refresh_callback = refresh_callback
        self.window = tk.Toplevel(parent)
        self.window.title("Health Records")
        self.window.geometry("900x700")
        self.window.resizable(True, True)

        # Apply shared styles
        try:
            apply_base_styles(self.window)
        except Exception:
            pass

        ensure_health_table()

        font_style = ("Helvetica", 11)

        # Filters
        filter_frame = ttk.Frame(self.window)
        filter_frame.pack(fill="x", padx=10, pady=10)

        # Animal dropdown (pre-populated from portal)
        ttk.Label(filter_frame, text="Animal:", font=font_style).pack(side="left")
        self.animal_var = tk.StringVar()
        self.animal_dropdown = ttk.Combobox(filter_frame, textvariable=self.animal_var, state="readonly")
        self.animal_dropdown.pack(side="left", padx=6)
        self.animal_dropdown.bind("<<ComboboxSelected>>", self.on_animal_select)

        ttk.Label(filter_frame, text="Species:", font=font_style).pack(side="left")
        self.species_var = tk.StringVar()
        self.species_dropdown = ttk.Combobox(filter_frame, textvariable=self.species_var, values=load_species_list(), state="readonly")
        self.species_dropdown.pack(side="left", padx=6)

        ttk.Label(filter_frame, text="Tag:", font=font_style).pack(side="left", padx=(10, 0))
        self.tag_var = tk.StringVar()
        self.tag_entry = ttk.Entry(filter_frame, textvariable=self.tag_var, width=18)
        self.tag_entry.pack(side="left", padx=6)

        ttk.Button(filter_frame, text="Apply", command=self.apply_filter).pack(side="left", padx=6)
        ttk.Button(filter_frame, text="Clear", command=self.clear_filter).pack(side="left")

        # Records table
        table_frame = ttk.Frame(self.window)
        table_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        cols = ("ID", "Tag", "Species", "Date", "Diagnosis", "Treatment", "Medication", "Dosage", "Vet", "Lab Result", "Severity", "Notes", "Next Check", "Withdrawal End")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", selectmode="extended")
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=120, anchor="center")
        self.tree.column("Diagnosis", width=160)
        self.tree.column("Treatment", width=160)
        self.tree.column("Notes", width=200)

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        # Actions
        action_frame = ttk.Frame(self.window)
        action_frame.pack(pady=5)
        ttk.Button(action_frame, text="Add Record", command=self.open_add_form, style="Primary.TButton").pack(side="left", padx=6)
        ttk.Button(action_frame, text="Edit Selected", command=self.open_edit_form, style="Secondary.TButton").pack(side="left", padx=6)
        ttk.Button(action_frame, text="Delete Selected", command=self.delete_selected, style="Danger.TButton").pack(side="left", padx=6)

        self.refresh()
        self.load_animals()

    def refresh(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        species = self.species_var.get() or None
        tag = (self.tag_var.get() or "").strip() or None
        rows = fetch_health_records(species, tag)
        for r in rows:
            self.tree.insert("", "end", values=r)

    def load_animals(self):
        try:
            rows = fetch_all_livestock()
            # Expect rows like (id, tag, type/species, ...)
            animals = [f"{r[1]} ({r[2]})" for r in rows]
            self.animal_dropdown['values'] = animals
        except Exception:
            self.animal_dropdown['values'] = []

    def on_animal_select(self, event=None):
        val = self.animal_var.get()
        # Parse "TAG (Species)"
        if val and "(" in val and ")" in val:
            try:
                tag = val.split("(")[0].strip()
                species = val[val.find("(")+1:val.find(")")].strip()
                self.tag_var.set(tag)
                self.species_var.set(species)
                self.apply_filter()
            except Exception:
                pass
    def apply_filter(self):
        self.refresh()

    def clear_filter(self):
        self.species_var.set("")
        self.tag_var.set("")
        self.refresh()

    def delete_selected(self):
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Select at least one record")
            return
        if not messagebox.askyesno("Confirm Delete", f"Delete {len(selection)} selected record(s)?"):
            return
        for item in selection:
            rid = int(self.tree.item(item, "values")[0])
            delete_health_record(rid)
        self.refresh()

    def open_add_form(self):
        win = tk.Toplevel(self.window)
        win.title("Add Health Record")
        win.geometry("600x700")
        win.resizable(True, True)
        try:
            apply_base_styles(win)
        except Exception:
            pass
        font_style = ("Helvetica", 11)

        # Animal selector (prefills tag/species)
        ttk.Label(win, text="Animal", font=font_style).pack(pady=5)
        animal_var = tk.StringVar(value=(self.animal_var.get() or ""))
        animal_dropdown = ttk.Combobox(win, textvariable=animal_var, state="readonly")
        animal_dropdown.pack(fill="x", padx=20)
        # populate animals
        try:
            rows = fetch_all_livestock()
            animal_dropdown['values'] = [f"{r[1]} ({r[2]})" for r in rows]
        except Exception:
            animal_dropdown['values'] = []

        # Tag / Species
        ttk.Label(win, text="Animal Tag", font=font_style).pack(pady=5)
        tag_var = tk.StringVar(value=(self.tag_var.get() or ""))
        ttk.Entry(win, textvariable=tag_var, font=font_style).pack(fill="x", padx=20)

        ttk.Label(win, text="Species", font=font_style).pack(pady=5)
        species_var = tk.StringVar(value=(self.species_var.get() or ""))
        species_dropdown = ttk.Combobox(win, textvariable=species_var, values=load_species_list(), state="readonly", font=font_style)
        species_dropdown.pack(fill="x", padx=20)

        def on_add_animal_select(event=None):
            val = animal_var.get()
            if val and "(" in val and ")" in val:
                try:
                    tag = val.split("(")[0].strip()
                    species = val[val.find("(")+1:val.find(")")].strip()
                    tag_var.set(tag)
                    species_var.set(species)
                except Exception:
                    pass
        animal_dropdown.bind("<<ComboboxSelected>>", on_add_animal_select)

        ttk.Label(win, text="Record Date", font=font_style).pack(pady=5)
        rec_date = DateEntry(win, date_pattern='yyyy-mm-dd', font=font_style)
        rec_date.pack(fill="x", padx=20)

        # Medical fields
        fields = [
            ("Diagnosis", ""),
            ("Treatment", ""),
            ("Medication", ""),
            ("Dosage", ""),
            ("Vet", ""),
            ("Lab Result", ""),
            ("Notes", ""),
        ]
        vars_map = {}
        for label, default in fields:
            ttk.Label(win, text=label, font=font_style).pack(pady=5)
            v = tk.StringVar(value=default)
            ttk.Entry(win, textvariable=v, font=font_style).pack(fill="x", padx=20)
            vars_map[label] = v

        ttk.Label(win, text="Severity", font=font_style).pack(pady=5)
        severity_var = tk.StringVar()
        ttk.Combobox(win, textvariable=severity_var, values=["Mild", "Moderate", "Severe"], state="readonly", font=font_style).pack(fill="x", padx=20)

        ttk.Label(win, text="Next Check Date", font=font_style).pack(pady=5)
        next_check = DateEntry(win, date_pattern='yyyy-mm-dd', font=font_style)
        next_check.pack(fill="x", padx=20)

        ttk.Label(win, text="Withdrawal End Date", font=font_style).pack(pady=5)
        withdraw_date = DateEntry(win, date_pattern='yyyy-mm-dd', font=font_style)
        withdraw_date.pack(fill="x", padx=20)

        def save_record():
            try:
                data = (
                    tag_var.get(),
                    species_var.get(),
                    rec_date.get_date(),
                    vars_map["Diagnosis"].get(),
                    vars_map["Treatment"].get(),
                    vars_map["Medication"].get(),
                    vars_map["Dosage"].get(),
                    vars_map["Vet"].get(),
                    vars_map["Lab Result"].get(),
                    severity_var.get(),
                    vars_map["Notes"].get(),
                    next_check.get_date(),
                    withdraw_date.get_date(),
                )
                if not data[0] or not data[1]:
                    messagebox.showerror("Error", "Tag and Species are required")
                    return
                insert_health_record(data)
                messagebox.showinfo("Success", "Health record added")
                win.destroy()
                self.refresh()
                try:
                    if self.refresh_callback:
                        self.refresh_callback()
                except Exception:
                    pass
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")

        ttk.Button(win, text="Save", command=save_record, style="Primary.TButton").pack(pady=15)

    def open_edit_form(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Select a record to edit")
            return
        if len(selected) > 1:
            messagebox.showwarning("Warning", "Please select only one record to edit")
            return
        values = self.tree.item(selected[0], "values")
        # values order matches columns defined in refresh
        (rid, tag0, species0, date0, diagnosis0, treatment0, medication0, dosage0, vet0, lab0, severity0, notes0, next_check0, withdraw0) = values

        win = tk.Toplevel(self.window)
        win.title("Edit Health Record")
        win.geometry("600x700")
        win.resizable(True, True)
        try:
            apply_base_styles(win)
        except Exception:
            pass
        font_style = ("Helvetica", 11)

        ttk.Label(win, text="Animal Tag", font=font_style).pack(pady=5)
        tag_var = tk.StringVar(value=tag0)
        ttk.Entry(win, textvariable=tag_var, font=font_style).pack(fill="x", padx=20)

        ttk.Label(win, text="Species", font=font_style).pack(pady=5)
        species_var = tk.StringVar(value=species0)
        species_dropdown = ttk.Combobox(win, textvariable=species_var, values=load_species_list(), state="readonly", font=font_style)
        species_dropdown.pack(fill="x", padx=20)

        ttk.Label(win, text="Record Date", font=font_style).pack(pady=5)
        rec_date = DateEntry(win, date_pattern='yyyy-mm-dd', font=font_style)
        try:
            if date0:
                rec_date.set_date(date0)
        except Exception:
            pass
        rec_date.pack(fill="x", padx=20)

        fields = [
            ("Diagnosis", diagnosis0),
            ("Treatment", treatment0),
            ("Medication", medication0),
            ("Dosage", dosage0),
            ("Vet", vet0),
            ("Lab Result", lab0),
            ("Notes", notes0),
        ]
        vars_map = {}
        for label, default in fields:
            ttk.Label(win, text=label, font=font_style).pack(pady=5)
            v = tk.StringVar(value=(default or ""))
            ttk.Entry(win, textvariable=v, font=font_style).pack(fill="x", padx=20)
            vars_map[label] = v

        ttk.Label(win, text="Severity", font=font_style).pack(pady=5)
        severity_var = tk.StringVar(value=(severity0 or ""))
        ttk.Combobox(win, textvariable=severity_var, values=["Mild", "Moderate", "Severe"], state="readonly", font=font_style).pack(fill="x", padx=20)

        ttk.Label(win, text="Next Check Date", font=font_style).pack(pady=5)
        next_check = DateEntry(win, date_pattern='yyyy-mm-dd', font=font_style)
        try:
            if next_check0:
                next_check.set_date(next_check0)
        except Exception:
            pass
        next_check.pack(fill="x", padx=20)

        ttk.Label(win, text="Withdrawal End Date", font=font_style).pack(pady=5)
        withdraw_date = DateEntry(win, date_pattern='yyyy-mm-dd', font=font_style)
        try:
            if withdraw0:
                withdraw_date.set_date(withdraw0)
        except Exception:
            pass
        withdraw_date.pack(fill="x", padx=20)

        def save_changes():
            try:
                data = (
                    tag_var.get(),
                    species_var.get(),
                    rec_date.get_date(),
                    vars_map["Diagnosis"].get(),
                    vars_map["Treatment"].get(),
                    vars_map["Medication"].get(),
                    vars_map["Dosage"].get(),
                    vars_map["Vet"].get(),
                    vars_map["Lab Result"].get(),
                    severity_var.get(),
                    vars_map["Notes"].get(),
                    next_check.get_date(),
                    withdraw_date.get_date(),
                )
                if not data[0] or not data[1]:
                    messagebox.showerror("Error", "Tag and Species are required")
                    return
                update_health_record(int(rid), data)
                messagebox.showinfo("Success", "Health record updated")
                win.destroy()
                self.refresh()
                try:
                    if self.refresh_callback:
                        self.refresh_callback()
                except Exception:
                    pass
            except Exception as e:
                messagebox.showerror("Error", f"Failed to update: {e}")

        ttk.Button(win, text="Save Changes", command=save_changes, style="Primary.TButton").pack(pady=15)


def open_health_records(parent, refresh_callback=None):
    HealthRecordsWindow(parent, refresh_callback)
