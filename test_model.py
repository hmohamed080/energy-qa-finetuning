import os
import time
import unittest


class TestModelGeneration(unittest.TestCase):
    """Tests for original and fine-tuned GPT-2 model generation."""

    @classmethod
    def setUpClass(cls):
        from transformers import GPT2Tokenizer, GPT2LMHeadModel

        cls.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.original_model = GPT2LMHeadModel.from_pretrained('gpt2')
        cls.original_model.eval()

        cls.fine_tuned_model_dir = './fine_tuned_model'
        if os.path.isdir(cls.fine_tuned_model_dir):
            cls.fine_tuned_model = GPT2LMHeadModel.from_pretrained(
                cls.fine_tuned_model_dir)
            cls.fine_tuned_model.eval()
        else:
            cls.fine_tuned_model = None

    def _generate(self, question, model, max_new_tokens=80):
        import torch

        input_ids = self.tokenizer.encode(question, return_tensors='pt')
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Original model tests
    # ------------------------------------------------------------------

    def test_original_model_generates_text(self):
        question = "What is renewable energy?"
        answer = self._generate(question, self.original_model)
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), len(question),
                           "Model should produce tokens beyond the prompt.")

    def test_original_model_response_time(self):
        question = "What is solar energy?"
        start = time.time()
        self._generate(question, self.original_model)
        elapsed_ms = (time.time() - start) * 1000
        # 60 s is intentionally permissive to accommodate CPU-only CI runners
        self.assertLessEqual(elapsed_ms, 60_000,
                              "Original model response time is too slow.")

    # ------------------------------------------------------------------
    # Fine-tuned model tests (skipped if model has not been trained yet)
    # ------------------------------------------------------------------

    def test_fine_tuned_model_generates_text(self):
        if self.fine_tuned_model is None:
            self.skipTest("Fine-tuned model not found; run fine_tune.py first.")
        question = "What is energy efficiency?"
        answer = self._generate(question, self.fine_tuned_model)
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), len(question))

    def test_fine_tuned_model_response_time(self):
        if self.fine_tuned_model is None:
            self.skipTest("Fine-tuned model not found; run fine_tune.py first.")
        question = "How does wind energy work?"
        start = time.time()
        self._generate(question, self.fine_tuned_model)
        elapsed_ms = (time.time() - start) * 1000
        # 60 s is intentionally permissive to accommodate CPU-only CI runners
        self.assertLessEqual(elapsed_ms, 60_000,
                              "Fine-tuned model response time is too slow.")

    # ------------------------------------------------------------------
    # Dataset / infrastructure tests
    # ------------------------------------------------------------------

    def test_energy_dataset_exists(self):
        self.assertTrue(
            os.path.isfile('data/energy_qa.txt'),
            "data/energy_qa.txt not found.",
        )

    def test_energy_dataset_has_content(self):
        with open('data/energy_qa.txt', 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
        self.assertGreater(len(lines), 0,
                           "energy_qa.txt should contain Q&A pairs.")


if __name__ == '__main__':
    unittest.main()