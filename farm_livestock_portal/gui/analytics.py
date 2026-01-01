# gui/analytics.py

import tkinter as tk
from tkinter import ttk
from collections import Counter
from services.db_service import fetch_all_livestock


def open_analytics(root):
    win = tk.Toplevel(root)
    win.title("Analytics")
    win.geometry("700x500")
    win.resizable(True, True)

    header = ttk.Label(win, text="Livestock Analytics", font=("Helvetica", 18, "bold"))
    header.pack(pady=(16, 8))

    rows = fetch_all_livestock()
    species_counts = Counter([r[2] for r in rows])

    frame = ttk.Frame(win)
    frame.pack(fill="both", expand=True, padx=12, pady=8)

    tree = ttk.Treeview(frame, columns=("Species", "Count"), show="headings")
    tree.heading("Species", text="Species")
    tree.heading("Count", text="Count")
    tree.column("Species", width=200)
    tree.column("Count", width=120, anchor="center")
    tree.pack(fill="both", expand=True)

    for s, c in species_counts.most_common():
        tree.insert("", "end", values=(s, c))

    total = ttk.Label(win, text=f"Total Animals: {len(rows)}", font=("Helvetica", 12))
    total.pack(pady=(8, 12))
