import tkinter as tk
from PIL import ImageTk, Image
import subprocess
import sys
import os

def run_file(file_path):
    # Activate the virtual environment
    venv_python = os.path.join(os.path.dirname(sys.executable), "python.exe")
    command = f"{venv_python} {file_path}"

    subprocess.call(command, shell=True)


# Define the file paths and corresponding button names
file_paths_and_names = [
    ("C:/Users/Dell/PycharmProjects/SnapchatFilters/Main.py", "Snapchat Filters"),
    ("C:/Users/Dell/PycharmProjects/SnapchatFilters/project.py", "Dinosaur Game"),
]            

# Create the main window and set its title
root = tk.Tk()
root.title("Python File Execution")

# Load the background image
background_image = Image.open(r'C:\Users\Dell\PycharmProjects\SnapchatFilters\process.jpg')
resized_background_image = background_image.resize((400, 400))  # Resize the image as needed
background_image_tk = ImageTk.PhotoImage(resized_background_image)

# Create a label to display the background image
background_label = tk.Label(root, image=background_image_tk)
background_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Create buttons for each file path
for file_path, button_name in file_paths_and_names:
    button = tk.Button(root, text=button_name, command=lambda file_path=file_path: run_file(file_path), font=("Arial", 12))
    button.pack(side=tk.TOP, pady=10)

# Start the main event loop
root.mainloop()
