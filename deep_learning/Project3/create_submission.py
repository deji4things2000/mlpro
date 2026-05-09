from os import path, getcwd
from zipfile import ZipFile

submission_name = f"{path.basename(getcwd())}.zip"

with ZipFile(submission_name, 'w') as zipfile:
    try:
        zipfile.write("basic_rnn_cell.py")
        zipfile.write("lstm_cell.py")
        zipfile.write("create_rnn.py")
        zipfile.write("text_generation.py")
        zipfile.write("basic_rnn.pt")
        zipfile.write("lstm_rnn.pt")
        zipfile.write("train.py")
        zipfile.write("sample.py")
        zipfile.write("create_sequence.py")
        zipfile.write("shakespeare.txt")
    except FileNotFoundError as e:
        print(f"You are missing a required file: {e}")

