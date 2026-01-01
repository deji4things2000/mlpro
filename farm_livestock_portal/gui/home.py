import tkinter as tk
from tkinter import ttk

from gui.styles import apply_base_styles


def open_home(parent, user=None):
	"""Open the Home window with a styled Sign Out button.
	Deferred import used in sign-out to avoid circular dependencies.
	"""
	win = tk.Toplevel(parent)
	win.title("Home")
	win.geometry("600x300")
	win.resizable(True, True)

	try:
		apply_base_styles(win)
	except Exception:
		pass

	header_text = f"Welcome{f', {user}' if user else ''}"
	ttk.Label(win, text=header_text, style="Header.TLabel").pack(pady=(15, 10))

	def sign_out():
		try:
			# Avoid circular import by importing inside handler
			from gui.auth import open_auth
		except Exception:
			open_auth = None
		try:
			win.destroy()
		except Exception:
			pass
		if open_auth:
			try:
				open_auth(parent)
			except Exception:
				pass

	ttk.Button(win, text="Sign Out", command=sign_out, style="Danger.TButton").pack(pady=10)

	return win
