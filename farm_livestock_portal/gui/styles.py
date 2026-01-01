# gui/styles.py

from tkinter import ttk

PRIMARY_BG = "#2E7D32"   # Green
PRIMARY_BG_ACTIVE = "#1B5E20"
SECONDARY_BG = "#5F6368"  # Neutral dark grey
SECONDARY_BG_ACTIVE = "#4A4D51"
DANGER_BG = "#C62828"    # Red
DANGER_BG_ACTIVE = "#8E0000"
TEXT_LIGHT = "#FFFFFF"
TEXT_DARK = "#111111"


def apply_base_styles(root):
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except Exception:
        pass

    # Table styles
    style.configure("Treeview", rowheight=26, font=("Helvetica", 11))
    style.configure("Treeview.Heading", font=("Helvetica", 12, "bold"))

    # Button styles (ttk)
    style.configure("Primary.TButton", font=("Helvetica", 11, "bold"), foreground=TEXT_LIGHT, padding=6)
    style.configure("Secondary.TButton", font=("Helvetica", 11), foreground=TEXT_LIGHT, padding=6)
    style.configure("Danger.TButton", font=("Helvetica", 11, "bold"), foreground=TEXT_LIGHT, padding=6)

    # Background mapping for active/normal states
    style.map("Primary.TButton",
              background=[('active', PRIMARY_BG_ACTIVE), ('!active', PRIMARY_BG)],
              foreground=[('disabled', '#cccccc'), ('!disabled', TEXT_LIGHT)])
    style.map("Secondary.TButton",
              background=[('active', SECONDARY_BG_ACTIVE), ('!active', SECONDARY_BG)],
              foreground=[('disabled', '#cccccc'), ('!disabled', TEXT_LIGHT)])
    style.map("Danger.TButton",
              background=[('active', DANGER_BG_ACTIVE), ('!active', DANGER_BG)],
              foreground=[('disabled', '#cccccc'), ('!disabled', TEXT_LIGHT)])

    # Labels
    style.configure("Header.TLabel", font=("Helvetica", 18, "bold"))
    style.configure("Subheader.TLabel", font=("Helvetica", 12))

    return style
