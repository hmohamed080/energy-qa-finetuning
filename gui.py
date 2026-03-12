import tkinter as tk
from tkinter import messagebox

class GPT2ComparisonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GPT-2 Model Comparison")

        # Original Model Section
        self.original_label = tk.Label(root, text="Original GPT-2 Model")
        self.original_label.pack()

        self.original_text = tk.Text(root, height=10, width=50)
        self.original_text.pack()

        # Fine-tuned Model Section
        self.fine_tuned_label = tk.Label(root, text="Fine-tuned GPT-2 Model")
        self.fine_tuned_label.pack()

        self.fine_tuned_text = tk.Text(root, height=10, width=50)
        self.fine_tuned_text.pack()

        # Compare Button
        self.compare_button = tk.Button(root, text="Compare", command=self.compare_models)
        self.compare_button.pack()

    def compare_models(self):
        original_output = self.original_text.get("1.0", tk.END)
        fine_tuned_output = self.fine_tuned_text.get("1.0", tk.END)

        # Simple comparison logic (you can enhance this)
        if original_output == fine_tuned_output:
            messagebox.showinfo("Comparison Result", "The outputs are identical.")
        else:
            messagebox.showinfo("Comparison Result", "The outputs are different.")

if __name__ == '__main__':
    root = tk.Tk()
    app = GPT2ComparisonApp(root)
    root.mainloop()