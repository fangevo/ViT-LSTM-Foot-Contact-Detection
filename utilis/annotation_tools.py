import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageTk
import os

class ImageLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Dataset Labeler")
        
        # Path settings
        self.label_dir = r"D:\IPP project\data_long_jump\annotates_combine_new"
        self.frame_base_dir = r"D:\IPP project\frames"
        
        # Current state variables
        self.current_video = 1
        self.current_frame = 0
        self.labels = None
        self.max_frame = 0
        
        # Create GUI components
        self.create_widgets()
        
        # Load initial video
        self.load_video(1)

    def create_widgets(self):
        # Video selection
        tk.Label(self.root, text="Video:").grid(row=0, column=0, padx=5, pady=5)
        self.video_var = tk.IntVar(value=1)
        video_spin = tk.Spinbox(self.root, from_=1, to=26, textvariable=self.video_var, 
                              command=self.change_video, width=5)
        video_spin.grid(row=0, column=1, padx=5, pady=5)

        # Frame selection
        tk.Label(self.root, text="Frame:").grid(row=0, column=2, padx=5, pady=5)
        self.frame_var = tk.IntVar(value=0)
        self.frame_spin = tk.Spinbox(self.root, from_=0, to=0, textvariable=self.frame_var, 
                                   command=self.change_frame, width=10)
        self.frame_spin.grid(row=0, column=3, padx=5, pady=5)

        # Image display
        self.image_label = tk.Label(self.root)
        self.image_label.grid(row=1, column=0, columnspan=4, padx=5, pady=5)

        # Current label display
        tk.Label(self.root, text="Current Label:").grid(row=2, column=0, padx=5, pady=5)
        self.current_label_var = tk.StringVar(value="0")
        tk.Label(self.root, textvariable=self.current_label_var).grid(row=2, column=1, padx=5, pady=5)

        # New label input
        tk.Label(self.root, text="New Label:").grid(row=2, column=2, padx=5, pady=5)
        self.new_label_var = tk.StringVar(value="0")
        self.label_entry = tk.Entry(self.root, textvariable=self.new_label_var, width=5)
        self.label_entry.grid(row=2, column=3, padx=5, pady=5)

        # Buttons
        tk.Button(self.root, text="Previous", command=self.prev_frame).grid(row=3, column=0, padx=5, pady=5)
        tk.Button(self.root, text="Next", command=self.next_frame).grid(row=3, column=1, padx=5, pady=5)
        tk.Button(self.root, text="Save", command=self.save_label).grid(row=3, column=2, padx=5, pady=5)
        tk.Button(self.root, text="Quit", command=self.root.quit).grid(row=3, column=3, padx=5, pady=5)

    def load_video(self, video_num):
        try:
            # Update current video number
            self.current_video = video_num
            
            # Load label file
            label_path = os.path.join(self.label_dir, f"{video_num:02d}.npy")
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label file not found: {label_path}")
                
            self.labels = np.load(label_path)
            self.max_frame = len(self.labels) - 1
            
            # Update frame selection range
            self.frame_spin.config(to=self.max_frame)
            self.current_frame = 0
            self.frame_var.set(0)
            
            # Display the first frame
            self.update_display()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video {video_num}: {str(e)}")

    def change_video(self):
        video_num = self.video_var.get()
        self.load_video(video_num)

    def change_frame(self):
        new_frame = self.frame_var.get()
        if 0 <= new_frame <= self.max_frame:
            self.current_frame = new_frame
            self.update_display()

    def prev_frame(self):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.frame_var.set(self.current_frame)
            self.update_display()

    def next_frame(self):
        if self.current_frame < self.max_frame:
            self.current_frame += 1
            self.frame_var.set(self.current_frame)
            self.update_display()

    def update_display(self):
        try:
            # Load and display image
            frame_path = os.path.join(self.frame_base_dir, f"{self.current_video:02d}", 
                                    f"{self.current_frame:05d}.png")
            if os.path.exists(frame_path):
                img = Image.open(frame_path)
                # Resize image to fit the window
                img = img.resize((960, 540), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.image_label.config(image=photo)
                self.image_label.image = photo  # Keep reference
                
                # Update current label display
                self.current_label_var.set(str(int(self.labels[self.current_frame])))
            else:
                self.image_label.config(image=None)
                self.current_label_var.set("No image")
                messagebox.showwarning("Warning", f"Frame not found: {frame_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update display: {str(e)}")

    def save_label(self):
        try:
            new_label = int(self.new_label_var.get())
            self.labels[self.current_frame] = new_label
            label_path = os.path.join(self.label_dir, f"{self.current_video:02d}.npy")
            np.save(label_path, self.labels)
            self.current_label_var.set(str(new_label))
            messagebox.showinfo("Success", "Label saved successfully!")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer label!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save label: {str(e)}")

def main():
    root = tk.Tk()
    app = ImageLabeler(root)
    root.mainloop()

if __name__ == "__main__":
    main()