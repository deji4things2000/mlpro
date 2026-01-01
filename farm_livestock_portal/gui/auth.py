import tkinter as tk
from tkinter import ttk, messagebox

from gui.styles import apply_base_styles
from services.user_service import ensure_users_table, insert_user, verify_user, ensure_default_admin
from gui.home import open_home


def open_auth(parent, on_login=None):
	"""Open authentication window with Login and Sign Up tabs."""
	ensure_users_table()
	try:
		ensure_default_admin()
	except Exception:
		pass
	win = tk.Toplevel(parent)
	win.title("Sign In / Sign Up")
	win.geometry("500x420")
	win.resizable(True, True)

	try:
		apply_base_styles(win)
	except Exception:
		pass

	nb = ttk.Notebook(win)
	nb.pack(fill="both", expand=True, padx=10, pady=10)

	font_style = ("Helvetica", 11)

	# Login tab
	login = ttk.Frame(nb)
	nb.add(login, text="Login")

	ttk.Label(login, text="Username or Email", font=font_style).pack(pady=6)
	identifier_var = tk.StringVar()
	ttk.Entry(login, textvariable=identifier_var, font=font_style).pack(fill="x", padx=20)

	ttk.Label(login, text="Password", font=font_style).pack(pady=6)
	password_var = tk.StringVar()
	ttk.Entry(login, textvariable=password_var, font=font_style, show="*").pack(fill="x", padx=20)

	def do_login():
		ident = (identifier_var.get() or "").strip()
		pwd = password_var.get() or ""
		if not ident or not pwd:
			messagebox.showerror("Error", "Please enter identifier and password")
			return
		user = verify_user(ident, pwd)
		if not user:
			messagebox.showerror("Error", "Invalid credentials")
			return
		try:
			win.destroy()
		except Exception:
			pass
		# Pass user to callback if provided
		if on_login:
			try:
				on_login(user)
			except Exception:
				pass
		try:
			open_home(parent, user.get("username"))
		except Exception:
			pass

	ttk.Button(login, text="Sign In", command=do_login, style="Primary.TButton").pack(pady=12)

	# Sign Up tab
	signup = ttk.Frame(nb)
	nb.add(signup, text="Sign Up")

	ttk.Label(signup, text="Username", font=font_style).pack(pady=6)
	su_username = tk.StringVar()
	ttk.Entry(signup, textvariable=su_username, font=font_style).pack(fill="x", padx=20)

	ttk.Label(signup, text="Email", font=font_style).pack(pady=6)
	su_email = tk.StringVar()
	ttk.Entry(signup, textvariable=su_email, font=font_style).pack(fill="x", padx=20)

	ttk.Label(signup, text="Password", font=font_style).pack(pady=6)
	su_password = tk.StringVar()
	ttk.Entry(signup, textvariable=su_password, font=font_style, show="*").pack(fill="x", padx=20)

	ttk.Label(signup, text="Confirm Password", font=font_style).pack(pady=6)
	su_confirm = tk.StringVar()
	ttk.Entry(signup, textvariable=su_confirm, font=font_style, show="*").pack(fill="x", padx=20)

	def do_signup():
		username = (su_username.get() or "").strip()
		email = (su_email.get() or "").strip()
		pwd = su_password.get() or ""
		conf = su_confirm.get() or ""
		if not username or not email or not pwd:
			messagebox.showerror("Error", "All fields are required")
			return
		if pwd != conf:
			messagebox.showerror("Error", "Passwords do not match")
			return
		ok = insert_user(username, email, pwd)
		if ok:
			messagebox.showinfo("Success", "Account created. Please sign in.")
			nb.select(0)
		else:
			messagebox.showerror("Error", "Failed to create account")

	ttk.Button(signup, text="Create Account", command=do_signup, style="Primary.TButton").pack(pady=12)

	return win
