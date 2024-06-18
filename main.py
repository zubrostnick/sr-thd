import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

# constants
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 400
WINDOW_TITLE = "Super Resolution App"
MODELS = ["srcnn", "edsr", "vdsr", "srgan"] 


def super_resolution(image):
    """
    TODO: set up the logic to load NN and upscale the image either as a single function or several ones.
    """
    pass


class SuperResolutionApp:
    """
    TODO: Class docstring; methods descriptions (?)
    """
    def __init__(self, root):
        self.root = root

        # create a Frame as a container for the buttons and combobox
        top_frame = tk.Frame(root)
        top_frame.pack(side="top", fill="x", padx=5, pady=5)

        # add "Choose Image" button
        self.btn_choose_image = tk.Button(top_frame, text="Choose Image", command=self.choose_image)
        self.btn_choose_image.grid(row=0, column=0, padx=5, pady=5)

        # add Combobox for model selection
        self.model_selector = ttk.Combobox(top_frame, values=MODELS, state="readonly")
        self.model_selector.set("Choose a model")
        self.model_selector.grid(row=0, column=1, padx=5, pady=5)

        # add "Upscale" button
        self.btn_upscale = tk.Button(top_frame, text="Upscale", command=self.upscale_image)
        self.btn_upscale.grid(row=0, column=2, padx=5, pady=5)

        # add labels to display images
        self.lbl_original_image = tk.Label(root)
        self.lbl_original_image.pack(side="left", padx=10, pady=10)

        self.lbl_upscaled_image = tk.Label(root)
        self.lbl_upscaled_image.pack(side="right", padx=10, pady=10)

        self.original_image = None
        self.upscaled_image = None

    def choose_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_image = Image.open(file_path)
            self.display_image(self.original_image, self.lbl_original_image)

    def display_image(self, image, label):
        image_tk = ImageTk.PhotoImage(image)
        label.config(image=image_tk)
        label.image = image_tk

    def upscale_image(self):
        if self.original_image:
            self.upscaled_image = super_resolution(self.original_image)
            self.display_image(self.upscaled_image, self.lbl_upscaled_image)
        else:
            messagebox.showwarning("Warning", "Please choose an image first.")


def instantiate_window():
    """
    Creates and returns a Tkinter window object.

    Returns:
        Tk: The root Tkinter window object.
    """
    root = tk.Tk()

    root.title(WINDOW_TITLE)
    root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")

    app = SuperResolutionApp(root)

    return root


if __name__ == "__main__":
    root = instantiate_window()
    root.mainloop()