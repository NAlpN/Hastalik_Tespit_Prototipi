import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
import os

class YOLOv8Interface:
    def __init__(self, master):
        self.predict_count = 2
        
        self.master = master
        master.title("YOLOv8 Arayüzü")
        master.geometry("800x600")
        master.configure(bg="#145363")

        self.style = ttk.Style()
        self.style.configure('TButton', font=('Arial Rounded MT Bold', 12))
        self.style.configure('TLabel', font=('Impact', 40), foreground='#FFFFFF')

        self.label_ata_ait = ttk.Label(master, text="Sağlıkta AIT", style='TLabel', background="#145363", foreground="#FFFFFF")
        self.label_ata_ait.pack(pady=20)

        self.select_button = ttk.Button(master, text="Görüntü Gir", style='TButton', command=self.select_image)
        self.select_button.pack(pady=20, ipadx=10, ipady=10)

        self.predict_button = ttk.Button(master, text="Neyim Var?", style='TButton', command=self.predict_image)
        self.predict_button.pack(pady=20, ipadx=10, ipady=10)

        self.image_label = ttk.Label(master)
        self.image_label.pack(side="left", anchor="nw", padx=20, pady=10)

        self.image_label2 = ttk.Label(master)
        self.image_label2.pack(side="right", anchor="ne", padx=20, pady=10)

        self.model_path = "C:/Users/alpnn/Masaüstü/Prototip/ultralytics/runs/classify/train/weights/best.pt"
        self.model = YOLO(self.model_path)
        self.selected_image_path = None

    def select_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.selected_image_path = file_path
            self.image = Image.open(file_path)
            self.display_image()

    def predict_image(self):
        if self.selected_image_path:
            selected_image = Image.open(self.selected_image_path)
            results = self.model.predict(source=selected_image, save=True, name='predict')

            selected_image_name = os.path.basename(self.selected_image_path)
            deneme_image_path = os.path.join("C:/Users/alpnn/Masaüstü/Prototip/ultralytics/runs/classify", f'predict{self.predict_count}\\{selected_image_name}')
            deneme_image = Image.open(deneme_image_path)
            deneme_image = deneme_image.resize((400, 300))
            deneme_photo = ImageTk.PhotoImage(image=deneme_image)
            self.image_label2.configure(image=deneme_photo)
            self.image_label2.image = deneme_photo

            self.predict_count += 1
        else:
            print("Lütfen önce bir resim seçin.")

    def display_image(self):
        image = self.image.resize((400, 300))
        photo = ImageTk.PhotoImage(image=image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo

def main():
    root = tk.Tk()
    app = YOLOv8Interface(root)
    root.mainloop()

if __name__ == "__main__":
    main()
