
import tkinter as tk
from tkinter import filedialog
from Modules.text_file_processing import text_file_processing

root = tk.Tk()
root.withdraw()

#user select the text file to process
corridor_text_file = filedialog.askopenfilename()
text_file_processing(corridor_text_file)


#if the file is the original, run.
# if __name__ == "__main__"
