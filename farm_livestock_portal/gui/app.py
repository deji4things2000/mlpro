import os
import sys
import tkinter as tk

# Ensure project root is on sys.path
try:
	ROOT = os.path.dirname(os.path.dirname(__file__))
	if ROOT not in sys.path:
		sys.path.insert(0, ROOT)
except Exception:
	pass

from gui.main_app import FarmApp
from gui.styles import apply_base_styles


def main():
	root = tk.Tk()
	# Size to screen dimensions for better UX
	root.update_idletasks()
	sw = root.winfo_screenwidth()
	sh = root.winfo_screenheight()
	root.geometry(f"{sw}x{sh}")
	root.resizable(True, True)

	try:
		apply_base_styles(root)
	except Exception:
		pass

	app = FarmApp(root)
	root.mainloop()


if __name__ == "__main__":
	main()
