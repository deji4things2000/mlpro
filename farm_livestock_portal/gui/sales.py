# gui/sales.py

import tkinter as tk
from tkinter import ttk, messagebox
from services.product_service import (
    ensure_products_table,
    insert_product,
    fetch_products,
    update_product,
    delete_product,
)


def open_sales(root):
    ensure_products_table()
    win = tk.Toplevel(root)
    win.title("Sales")
    win.geometry("900x600")
    win.resizable(True, True)

    header = ttk.Label(win, text="Product Catalog", font=("Helvetica", 18, "bold"))
    header.pack(pady=(16, 8))

    container = ttk.Notebook(win)
    catalog_tab = ttk.Frame(container)
    add_tab = ttk.Frame(container)
    container.add(catalog_tab, text="Catalog")
    container.add(add_tab, text="Add Product")
    container.pack(fill="both", expand=True)

    # Catalog list
    cols = ("ID", "Name", "Category", "Price", "Stock", "Description")
    tree = ttk.Treeview(catalog_tab, columns=cols, show="headings")
    for c in cols:
        tree.heading(c, text=c)
        tree.column(c, width=140 if c in ("Name", "Description") else 100, anchor="center")
    tree.pack(fill="both", expand=True, padx=12, pady=8)

    def refresh():
        for i in tree.get_children():
            tree.delete(i)
        for r in fetch_products():
            tree.insert("", "end", values=r)

    button_frame = ttk.Frame(catalog_tab)
    button_frame.pack(pady=6)

    def edit_selected():
        sel = tree.selection()
        if not sel:
            messagebox.showwarning("Warning", "Select a product to edit")
            return
        values = tree.item(sel[0], "values")
        (pid, name0, cat0, price0, stock0, desc0) = values
        ew = tk.Toplevel(win)
        ew.title("Edit Product")
        ew.geometry("500x450")

        ttk.Label(ew, text="Name").pack(pady=6)
        name = tk.StringVar(value=name0)
        ttk.Entry(ew, textvariable=name).pack(fill="x", padx=20)

        ttk.Label(ew, text="Category").pack(pady=6)
        cat = tk.StringVar(value=cat0)
        ttk.Entry(ew, textvariable=cat).pack(fill="x", padx=20)

        ttk.Label(ew, text="Price").pack(pady=6)
        price = tk.StringVar(value=str(price0))
        ttk.Entry(ew, textvariable=price).pack(fill="x", padx=20)

        ttk.Label(ew, text="Stock").pack(pady=6)
        stock = tk.StringVar(value=str(stock0))
        ttk.Entry(ew, textvariable=stock).pack(fill="x", padx=20)

        ttk.Label(ew, text="Description").pack(pady=6)
        desc = tk.StringVar(value=(desc0 or ""))
        ttk.Entry(ew, textvariable=desc).pack(fill="x", padx=20)

        def save_edit():
            try:
                p = float(price.get())
                s = int(stock.get())
                update_product(int(pid), name.get(), cat.get(), p, s, desc.get())
                messagebox.showinfo("Updated", "Product updated")
                ew.destroy()
                refresh()
            except Exception as e:
                messagebox.showerror("Error", f"Failed: {e}")

        ttk.Button(ew, text="Save Changes", command=save_edit).pack(pady=12)

    def delete_selected():
        sel = tree.selection()
        if not sel:
            messagebox.showwarning("Warning", "Select a product to delete")
            return
        pid = int(tree.item(sel[0], "values")[0])
        if not messagebox.askyesno("Delete", "Delete selected product?"):
            return
        delete_product(pid)
        refresh()

    ttk.Button(button_frame, text="Edit Selected", command=edit_selected).pack(side="left", padx=6)
    ttk.Button(button_frame, text="Delete Selected", command=delete_selected).pack(side="left", padx=6)

    # Add product form
    ttk.Label(add_tab, text="Name").pack(pady=6)
    name = tk.StringVar()
    ttk.Entry(add_tab, textvariable=name).pack(fill="x", padx=20)

    ttk.Label(add_tab, text="Category").pack(pady=6)
    cat = tk.StringVar()
    ttk.Entry(add_tab, textvariable=cat).pack(fill="x", padx=20)

    ttk.Label(add_tab, text="Price").pack(pady=6)
    price = tk.StringVar()
    ttk.Entry(add_tab, textvariable=price).pack(fill="x", padx=20)

    ttk.Label(add_tab, text="Stock").pack(pady=6)
    stock = tk.StringVar()
    ttk.Entry(add_tab, textvariable=stock).pack(fill="x", padx=20)

    ttk.Label(add_tab, text="Description").pack(pady=6)
    desc = tk.StringVar()
    ttk.Entry(add_tab, textvariable=desc).pack(fill="x", padx=20)

    def save_new():
        try:
            p = float(price.get())
            s = int(stock.get())
            insert_product(name.get(), cat.get(), p, s, desc.get())
            messagebox.showinfo("Saved", "Product added")
            name.set(""); cat.set(""); price.set(""); stock.set(""); desc.set("")
            refresh()
        except Exception as e:
            messagebox.showerror("Error", f"Failed: {e}")

    ttk.Button(add_tab, text="Add Product", command=save_new).pack(pady=12)

    refresh()
