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
from gui.auth import open_auth


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

	# Show sign-in page on startup; main app is available after
	# Hide main window until login
	root.withdraw()

	def on_login(user):
		try:
			# Show main app scoped to user
			FarmApp(root, user.get("username"))
			root.deiconify()
		except Exception:
			root.deiconify()

	try:
		open_auth(root, on_login=on_login)
	except Exception:
		# Fallback: show app without user context
		FarmApp(root)
		root.deiconify()
	root.mainloop()


if __name__ == "__main__":
	main()
