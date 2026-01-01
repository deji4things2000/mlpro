import os, uuid, subprocess
from typing import Optional, List
from dotenv import load_dotenv
from PyQt5 import QtWidgets
from sqlalchemy.orm import Session
from .db import engine, SessionLocal, Base
from .models import Animal, Event
from .barcodes import generate_barcode
from .reports import create_pdf

load_dotenv()
OPEN_CMD = os.getenv("OPEN_CMD", "open")

def init_db():
    Base.metadata.create_all(bind=engine)

def new_tag_id() -> str:
    return "A-" + uuid.uuid4().hex[:8].upper()

class AddAnimalDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Animal")
        form = QtWidgets.QFormLayout(self)
        self.species = QtWidgets.QLineEdit()
        self.breed = QtWidgets.QLineEdit()
        self.sex = QtWidgets.QLineEdit()
        self.dob = QtWidgets.QLineEdit()
        self.weight = QtWidgets.QDoubleSpinBox(); self.weight.setRange(0, 10000); self.weight.setDecimals(2)
        self.location = QtWidgets.QLineEdit()
        self.notes = QtWidgets.QTextEdit()
        form.addRow("Species", self.species)
        form.addRow("Breed", self.breed)
        form.addRow("Sex", self.sex)
        form.addRow("DOB (YYYY-MM-DD)", self.dob)
        form.addRow("Weight (kg)", self.weight)
        form.addRow("Location", self.location)
        form.addRow("Notes", self.notes)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        form.addRow(btns)
    def get_data(self):
        return {
            "species": self.species.text().strip(),
            "breed": self.breed.text().strip(),
            "sex": self.sex.text().strip(),
            "dob": self.dob.text().strip(),
            "weight_kg": float(self.weight.value()),
            "location": self.location.text().strip(),
            "notes": self.notes.toPlainText().strip(),
        }

class AddEventDialog(QtWidgets.QDialog):
    def __init__(self, tag_id: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Add Event for {tag_id}")
        form = QtWidgets.QFormLayout(self)
        self.event_type = QtWidgets.QLineEdit()
        self.event_date = QtWidgets.QLineEdit()
        self.details = QtWidgets.QTextEdit()
        form.addRow("Type", self.event_type)
        form.addRow("Date (YYYY-MM-DD)", self.event_date)
        form.addRow("Details", self.details)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        form.addRow(btns)
    def get_data(self):
        return {
            "event_type": self.event_type.text().strip(),
            "event_date": self.event_date.text().strip(),
            "details": self.details.toPlainText().strip(),
        }

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Livestock Tracker"); self.resize(1000, 600)
        self.table = QtWidgets.QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["Tag", "Species", "Breed", "Sex", "DOB", "Weight(kg)", "Location", "Notes"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

        toolbar = QtWidgets.QToolBar("Actions"); self.addToolBar(toolbar)
        add_act = QtWidgets.QAction("Add Animal", self); add_act.triggered.connect(self.add_animal)
        refresh_act = QtWidgets.QAction("Refresh", self); refresh_act.triggered.connect(self.load_animals)
        barcode_act = QtWidgets.QAction("Generate Barcode", self); barcode_act.triggered.connect(self.make_barcode)
        event_act = QtWidgets.QAction("Add Event", self); event_act.triggered.connect(self.add_event)
        report_act = QtWidgets.QAction("Create PDF Report", self); report_act.triggered.connect(self.make_report)
        del_act = QtWidgets.QAction("Delete", self); del_act.triggered.connect(self.delete_animal)
        toolbar.addActions([add_act, refresh_act, barcode_act, event_act, report_act, del_act])

        container = QtWidgets.QWidget(); layout = QtWidgets.QVBoxLayout(container)
        layout.addWidget(self.table); self.setCentralWidget(container)
        self.load_animals()

    def load_animals(self):
        self.table.setRowCount(0)
        with SessionLocal() as db:
            rows: List[Animal] = db.query(Animal).order_by(Animal.tag_id).all()
            for a in rows:
                r = self.table.rowCount(); self.table.insertRow(r)
                self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(a.tag_id))
                self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(a.species))
                self.table.setItem(r, 2, QtWidgets.QTableWidgetItem(a.breed))
                self.table.setItem(r, 3, QtWidgets.QTableWidgetItem(a.sex))
                self.table.setItem(r, 4, QtWidgets.QTableWidgetItem(a.dob))
                self.table.setItem(r, 5, QtWidgets.QTableWidgetItem(f"{a.weight_kg:.2f}"))
                self.table.setItem(r, 6, QtWidgets.QTableWidgetItem(a.location))
                self.table.setItem(r, 7, QtWidgets.QTableWidgetItem(a.notes))

    def current_tag(self) -> Optional[str]:
        r = self.table.currentRow()
        if r < 0: return None
        item = self.table.item(r, 0)
        return item.text() if item else None

    def add_animal(self):
        dlg = AddAnimalDialog(self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            data = dlg.get_data()
            with SessionLocal() as db:
                tag = new_tag_id()
                a = Animal(tag_id=tag, **data)
                db.add(a); db.commit()
            self.load_animals()
            QtWidgets.QMessageBox.information(self, "Added", f"Animal {tag} added.")

    def delete_animal(self):
        tag = self.current_tag()
        if not tag:
            QtWidgets.QMessageBox.warning(self, "No selection", "Select an animal row."); return
        with SessionLocal() as db:
            a = db.query(Animal).filter(Animal.tag_id == tag).first()
            if not a:
                QtWidgets.QMessageBox.warning(self, "Not found", "Animal not found."); return
            db.delete(a); db.commit()
        self.load_animals()
        QtWidgets.QMessageBox.information(self, "Deleted", f"Animal {tag} deleted.")

    def add_event(self):
        tag = self.current_tag()
        if not tag:
            QtWidgets.QMessageBox.warning(self, "No selection", "Select an animal row."); return
        dlg = AddEventDialog(tag, self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            data = dlg.get_data()
            with SessionLocal() as db:
                evt = Event(tag_id=tag, **data)
                db.add(evt); db.commit()
            QtWidgets.QMessageBox.information(self, "Event", f"Event added for {tag}.")

    def make_barcode(self):
        tag = self.current_tag()
        if not tag:
            QtWidgets.QMessageBox.warning(self, "No selection", "Select an animal row."); return
        path = generate_barcode(tag)
        QtWidgets.QMessageBox.information(self, "Barcode", f"Saved: {path}")
        try: subprocess.run([OPEN_CMD, path], check=False)
        except Exception: pass

    def make_report(self):
        with SessionLocal() as db:
            path = create_pdf(db)
        QtWidgets.QMessageBox.information(self, "Report", f"Saved: {path}")
        try: subprocess.run([OPEN_CMD, path], check=False)
        except Exception: pass
