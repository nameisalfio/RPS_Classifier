import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import torch
from torchvision.transforms import transforms
from RPS_Classifier import RPSClassifier

class RPSClassifierApp:
    
    def __init__(self, root):
        self.root = root
        self.root.title("RPS Classifier Demo")
        self.class_names = {0: "Scissors", 1: "Rock", 2: "Paper"}  
        self.root.geometry("800x700")  
        self.root.configure(bg='#708090')  

        self.model = RPSClassifier()
        self.model.load_state_dict(torch.load(os.path.join("models", "rps_experiment_1-50.pth")))
        self.model.eval()

        self.label_input = tk.Label(root, text="Input Gesture:", font=("Arial", 20), bg='#708090')
        self.label_input.pack(pady=10)

        empty_image = Image.new('RGB', (240, 240), 'white')
        empty_image_tk = ImageTk.PhotoImage(empty_image)

        self.input_image = tk.Label(root, image=empty_image_tk, borderwidth=5, relief="solid")  
        self.input_image.image = empty_image_tk
        self.input_image.pack(pady=10)

        self.button_frame = tk.Frame(root, bg='#708090')
        self.button_frame.pack(pady=10)

        self.label_load = tk.Label(self.button_frame, text="Load Image", font=("Arial", 15), bg='#708090')
        self.label_load.grid(row=0, column=0, pady=10)

        load_icon_pil = Image.open(os.path.join("icons", "load_icon.png"))
        load_icon_pil = load_icon_pil.resize((50, 50))  
        load_icon = ImageTk.PhotoImage(load_icon_pil)

        self.btn_load_image = tk.Button(self.button_frame, image=load_icon, command=self.load_image) 
        self.btn_load_image.image = load_icon  
        self.btn_load_image.grid(row=0, column=1, padx=20, pady=10)

        self.label_classify = tk.Label(self.button_frame, text="Classify Image", font=("Arial", 15), bg='#708090')
        self.label_classify.grid(row=1, column=0, pady=10)

        classify_icon_pil = Image.open(os.path.join("icons", "classify_icon.png"))
        classify_icon_pil = classify_icon_pil.resize((50, 50))  
        classify_icon = ImageTk.PhotoImage(classify_icon_pil)

        self.btn_classify = tk.Button(self.button_frame, image=classify_icon, command=self.classify_image)  
        self.btn_classify.image = classify_icon  
        self.btn_classify.grid(row=1, column=1, padx=20, pady=10)

        self.btn_reset = tk.Button(root, text="Reset", command=self.reset, font=("Arial", 15, "bold"), bg='#2b2e50', fg='#506070')
        self.btn_reset.pack(pady=10)

        self.label_result = tk.Label(root, text="", font=("Arial", 20), bg='#708090', fg='white')  
        self.label_result.pack(pady=10)
    
    def load_image(self):
        try:
            file_path = filedialog.askopenfilename(initialdir=os.path.join("data", "test"), title="Select Image", filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
            
            image = Image.open(file_path)
            image_display = image.resize((240, 240))  
            image_tk = ImageTk.PhotoImage(image_display)
            self.input_image.config(image=image_tk)
            self.input_image.image = image_tk

            image_model = image.resize((48, 48)) 
            transform = transforms.Compose([transforms.ToTensor()])
            self.image_tensor = transform(image_model)
        except Exception as e:
            print(f"Error loading image: {e}")

    def classify_image(self):
        try:
            if self.image_tensor is None:
                print("No image loaded")
                return

            output = self.model(self.image_tensor.unsqueeze(0))  
            _, predicted_class = torch.max(output, 1) 

            self.label_result.config(text=f"Predicted class: {self.class_names[predicted_class.item()]}", font=("Arial", 20, 'bold'))  # Set font to bold
        except Exception as e:
            print(f"Error classifying image: {e}")

    def reset(self):
        empty_image = Image.new('RGB', (240, 240), 'white')
        empty_image_tk = ImageTk.PhotoImage(empty_image)
        self.input_image.config(image=empty_image_tk)
        self.input_image.image = empty_image_tk
        self.label_result.config(text="")

if __name__ == "__main__":
    print("Starting RPS Classifier App...")
    
    root = tk.Tk()
    app = RPSClassifierApp(root)
    root.mainloop()
    
    print("Closing RPS Classifier App...")