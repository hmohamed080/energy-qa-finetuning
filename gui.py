import os
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext

FINE_TUNED_MODEL_DIR = './fine_tuned_model'


class EnergyQAApp:
    """GUI application that compares original and fine-tuned GPT-2 responses."""

    def __init__(self, root):
        self.root = root
        self.root.title("Energy QA – Model Comparison")
        self.root.geometry("900x650")
        self.root.resizable(True, True)

        self.tokenizer = None
        self.original_model = None
        self.fine_tuned_model = None

        self._build_ui()
        self._load_models_async()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ── Question input ──────────────────────────────────────────────
        input_frame = tk.Frame(self.root, padx=12, pady=10)
        input_frame.pack(fill=tk.X)

        tk.Label(input_frame, text="Enter an energy question:",
                 font=("Arial", 12, "bold")).pack(anchor="w")

        self.question_entry = tk.Entry(input_frame, font=("Arial", 12))
        self.question_entry.pack(fill=tk.X, pady=4)
        self.question_entry.insert(0, "What is renewable energy?")
        self.question_entry.bind("<Return>", lambda _e: self.compare_models())

        btn_frame = tk.Frame(input_frame)
        btn_frame.pack(anchor="w", pady=4)

        self.compare_btn = tk.Button(
            btn_frame, text="Compare Models", command=self.compare_models,
            font=("Arial", 11), bg="#1565C0", fg="white",
            padx=10, state=tk.DISABLED,
        )
        self.compare_btn.pack(side=tk.LEFT)

        self.status_label = tk.Label(btn_frame, text="  Loading models…",
                                     fg="gray", font=("Arial", 10))
        self.status_label.pack(side=tk.LEFT)

        # ── Results area ────────────────────────────────────────────────
        results_frame = tk.Frame(self.root, padx=12, pady=6)
        results_frame.pack(fill=tk.BOTH, expand=True)

        # Original model column
        left_frame = tk.Frame(results_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        tk.Label(left_frame, text="Original GPT-2",
                 font=("Arial", 11, "bold"), fg="#1565C0").pack(anchor="w")
        self.original_text = scrolledtext.ScrolledText(
            left_frame, height=18, wrap=tk.WORD, font=("Arial", 10),
        )
        self.original_text.pack(fill=tk.BOTH, expand=True)

        # Fine-tuned model column
        right_frame = tk.Frame(results_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(6, 0))
        tk.Label(right_frame, text="Fine-tuned GPT-2",
                 font=("Arial", 11, "bold"), fg="#2E7D32").pack(anchor="w")
        self.fine_tuned_text = scrolledtext.ScrolledText(
            right_frame, height=18, wrap=tk.WORD, font=("Arial", 10),
        )
        self.fine_tuned_text.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models_async(self):
        threading.Thread(target=self._load_models, daemon=True).start()

    def _load_models(self):
        try:
            from transformers import GPT2Tokenizer, GPT2LMHeadModel

            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token

            self.original_model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.original_model.eval()

            if os.path.isdir(FINE_TUNED_MODEL_DIR):
                self.fine_tuned_model = GPT2LMHeadModel.from_pretrained(
                    FINE_TUNED_MODEL_DIR)
                self.fine_tuned_model.eval()
                status_msg = "Models loaded."
                status_color = "green"
            else:
                status_msg = ("Fine-tuned model not found – "
                              "run fine_tune.py first.")
                status_color = "orange"
        except Exception as exc:  # pragma: no cover
            status_msg = f"Error loading models: {exc}"
            status_color = "red"

        self.root.after(0, lambda: self.status_label.config(
            text=f"  {status_msg}", fg=status_color))
        self.root.after(0, lambda: self.compare_btn.config(state=tk.NORMAL))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _generate(self, question, model, max_new_tokens=120):
        import torch

        input_ids = self.tokenizer.encode(question, return_tensors='pt')
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                # Sampling produces more natural responses in the GUI.
                # Tests use deterministic decoding (do_sample=False) for
                # reproducibility; the GUI uses sampling for variety.
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Compare action
    # ------------------------------------------------------------------

    def compare_models(self):
        question = self.question_entry.get().strip()
        if not question:
            messagebox.showwarning("Input Required",
                                   "Please enter an energy question.")
            return

        if self.tokenizer is None or self.original_model is None:
            messagebox.showerror("Not Ready",
                                 "Models are still loading – please wait.")
            return

        self.compare_btn.config(state=tk.DISABLED, text="Generating…")
        self.original_text.delete("1.0", tk.END)
        self.fine_tuned_text.delete("1.0", tk.END)
        self.original_text.insert(tk.END, "Generating…")
        self.fine_tuned_text.insert(tk.END, "Generating…")

        def run():
            # Original model
            try:
                orig = self._generate(question, self.original_model)
            except Exception as exc:
                orig = f"[Error] {exc}"

            # Fine-tuned model
            if self.fine_tuned_model is not None:
                try:
                    fine = self._generate(question, self.fine_tuned_model)
                except Exception as exc:
                    fine = f"[Error] {exc}"
            else:
                fine = ("Fine-tuned model not available.\n"
                        "Run fine_tune.py to create it.")

            self.root.after(0, lambda: self._display_results(orig, fine))

        threading.Thread(target=run, daemon=True).start()

    def _display_results(self, original, fine_tuned):
        self.original_text.delete("1.0", tk.END)
        self.original_text.insert(tk.END, original)
        self.fine_tuned_text.delete("1.0", tk.END)
        self.fine_tuned_text.insert(tk.END, fine_tuned)
        self.compare_btn.config(state=tk.NORMAL, text="Compare Models")


if __name__ == '__main__':
    root = tk.Tk()
    app = EnergyQAApp(root)
    root.mainloop()